import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import lib
import time
import visdom
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset

# 将每个片段的前step长度作为测试集
# 模型采用Res_GRU

start = time.time()
data = pd.read_csv('dataset.csv')

# 添加6个特征
extra_feats = {
     'i_s': lambda x: np.sqrt(x['i_d']**2 + x['i_q']**2),  # Current vector norm
     'u_s': lambda x: np.sqrt(x['u_d']**2 + x['u_q']**2),  # Voltage vector norm
     'S_el': lambda x: x['i_s']*x['u_s'],                  # Apparent power
     'P_el': lambda x: x['i_d'] * x['u_d'] + x['i_q'] *x['u_q'],  # Effective power
     'i_s_x_w': lambda x: x['i_s']*x['motor_speed'],
     'S_x_w': lambda x: x['S_el']*x['motor_speed'],
}
data = data.assign(**extra_feats)

data2 = data.copy()
key = list(data2)[2:]
data2[key] = data[key].apply(lambda x: (x - x.min()) / (x.max() - x.min()))     # 最大最小归一化
group2 = data2.groupby(by='session_id')
group1 = data.groupby(by='session_id')
X, Y = [], []
for i in range(len(group2)):
    X.append(group2.get_group(i).loc[:, key].to_numpy())
    Y.append(group1.get_group(i).loc[:, 'pm'].to_numpy())
step1, step2 = 25, 12                               # LSTM的输入和输出步长
data_X, data_Y, predict_X = [], [], []              # predict_X为需要提交结果的输入数据，对应每个片段的前step1行数据
predict_X2 = []
test_X, test_y = [], []
for x, y in zip(X, Y):
    for i in range(step2, len(x) - step1 + 1, 1):
        if i - step2 > 15:
            x1 = torch.tensor(x[i:i + step1, :], dtype=torch.float32)
            y1 = torch.tensor(y[i - step2:i], dtype=torch.float32)
            data_X.append(torch.flip(x1, dims=[0]))
            data_Y.append(torch.flip(y1, dims=[0]))
        else:
            x1 = torch.tensor(x[i:i + step1, :], dtype=torch.float32)
            y1 = torch.tensor(y[i - step2:i], dtype=torch.float32)
            test_X.append(torch.flip(x1, dims=[0]))
            test_y.append(torch.flip(y1, dims=[0]))
    pd_x = []
    for j in range(step2):
        pd_x.append(torch.flip(torch.tensor(x[j:step1+j, :], dtype=torch.float32), dims=[0]))
    predict_X.append(torch.stack(pd_x))
    predict_X2.append(torch.flip(torch.tensor(x[step2:step1 + step2, :], dtype=torch.float32), dims=[0]))
train_X, train_y, predict_X = torch.stack(data_X), torch.stack(data_Y), torch.stack(predict_X)
test_X, test_y = torch.stack(test_X), torch.stack(test_y)
predict_X2= torch.stack(predict_X2)

index = list(np.arange(len(train_X)))
random.shuffle(index)
train_X, train_y = train_X[index], train_y[index]

net = lib.Res_GRU(input_dim=len(key), num_hidden=64, input_num=step1, output_dim=step2, dropout=0.05)
lr, num_epochs = 0.004, 200
device = lib.try_gpu()
net = net.to(device)
net.apply(lib.init_weights)

loss = nn.MSELoss()
weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or 'bn' in name)
parameters = [{'params':weight_decay_list},
              {'params':no_decay_list, 'weight_decay':0}]
updater = torch.optim.SGD(parameters, lr, momentum=0.6, weight_decay=0.001)
# 自适应学习率设置
scheduler = torch.optim.lr_scheduler.LambdaLR(updater, lambda x: np.cos(x % 30 / 20) / ((x // 20) * 4 + 1))

# 根据测试集上的损失值动态调整学习率
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(updater, mode='min', factor=0.8, patience=5, threshold=0.05, threshold_mode='abs', min_lr=0, eps=1e-08, verbose=False)

batch_size = 128

# 绘制观察曲线
wind = visdom.Visdom(env='main')
wind.line([0.], [0.], win='train', opts=dict(title='epoch_loss', legend=['train_loss']))     # 图像的标例
wind.line([[0., 0.]], [0.], win='test', opts = dict(title='error', legend=['train_error', 'test_error']))

train_data = TensorDataset(train_X, train_y)
data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

test_X = test_X.to(device)
epoch_loss = []                 # 存储每一代的训练误差
LR_list = []                    # 存储每一代的学习率
best_MAE = 2.5
for epoch in range(num_epochs):
    LR_list.append(lr)
    temp_loss = []
    net.train()
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        l = loss(y_hat.reshape(y.shape), y).requires_grad_()
        temp_loss.append(l.cpu().detach().numpy())
        updater.zero_grad()
        l.backward()
        lib.grad_clipping(net, 20)            # 梯度裁剪
        # print([x.grad for x in updater.param_groups[0]['params']])
        updater.step()
    print('第{}次训练损失值为：{}'.format(epoch, np.mean(temp_loss)))
    wind.line([np.mean(temp_loss)], [epoch], win='train', update='append')
    # 在测试集上评估
    if epoch % 2 == 0:
        net.eval()
        with torch.no_grad():
            predict1 = [net(x[0].to(device)).detach() for x in data_loader]
            MAE0 = lib.MAE(torch.concat(predict1, dim=0).cpu(), train_y)
            print('第%s次训练的训练集上MAE结果为:%.3f'%(epoch, MAE0))

            train_data1 = TensorDataset(test_X)
            data_loader1 = DataLoader(train_data1, batch_size=batch_size, shuffle=False)
            predict1 = [net(x[0].to(device)).detach() for x in data_loader1]
            MAE1 = lib.MAE(torch.concat(predict1, dim=0).cpu(), test_y)
            print('第%s次训练的测试集上MAE结果为:%.3f'%(epoch, MAE1))
        if MAE0 < best_MAE:
            best_MAE = MAE0
            torch.save(net.state_dict(), 'best_model1')
        # 绘制曲线
        wind.line([[MAE0, MAE1]], [epoch], win='test', update='append')
    scheduler.step()
    epoch_loss.append(np.mean(temp_loss))

# 预测结果
net.load_state_dict(torch.load('best_model1'))
net.eval()
with torch.no_grad():
    predict1 = [net(x.to(device)).detach().cpu().numpy() for x in predict_X]
    res = lib.get_res(predict1, step2)

# 分别为 session_id, rank, pm
column1, column2, column3 = [], [], []
for i in range(len(group1)):
    column1.extend([i] * 12)
    column2.extend(list(range(1, 13)))
    column3.extend(res[i][::-1])
df = pd.DataFrame({'session_id':column1, 'rank':column2, 'pm':column3})
df.to_csv('data.csv', index=False, header=True)            # 保留列名但不要行索引

end = time.time()
print('程序运行时间为：{}'.format(end - start))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].plot(range(num_epochs), epoch_loss, marker='o', markersize=3)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('mean_loss')

ax[1].plot(range(num_epochs), LR_list, marker='o', markersize=3)
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('lr')
plt.savefig('results.png', dpi=600, bbox_inches='tight')   # 指定分辨率
plt.show()