import pandas as pd
import torch
from torch import nn
import lib
import time
import visdom
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
# 将每个片段的前step长度作为测试集
# 采用bagging的集成学习策略。每次重采样生成一个新的训练集

data = pd.read_csv('dataset.csv')
# del_keys = ['stator_winding', 'stator_yoke', 'torque']    # 与其他特征相关系数大于0.95,可以删除
# data2 = data.drop(del_keys, axis=1)

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
step1, step2 = 30, 12                               # LSTM的输入和输出步长
data_X, data_Y, predict_X = [], [], []              # predict_X为需要提交结果的输入数据，对应每个片段的前step1行数据
for x, y in zip(X, Y):
    for i in range(step2, len(x) - step1 + 1, 1):
        x1 = torch.tensor(x[i:i + step1, :], dtype=torch.float32)
        y1 = torch.tensor(y[i - step2:i], dtype=torch.float32)
        data_X.append(torch.flip(x1, dims=[0]))      # 翻转数据，保证有意义的输入顺序
        data_Y.append(torch.flip(y1, dims=[0]))
    # 仅使用最后step1长度数据做预测
    predict_X.append(torch.flip(torch.tensor(x[:step1, :], dtype=torch.float32), dims=[0]))
predict_X = torch.stack(predict_X)
data_X, data_Y = torch.stack(data_X), torch.stack(data_Y)

lr, num_epochs, batch_size = 0.004, 150, 128
device = lib.try_gpu()
loss = nn.MSELoss()
for final_epoch in range(15, 16):
    start = time.time()
    index = np.random.choice(np.arange(len(data_X)), size=len(data_X), replace=True)     # 有放回采用的bagging策略
    train_X, train_y = data_X[index], data_Y[index]
    # 残差连接的GRU网络
    net = lib.Res_GRU(input_dim=len(key), num_hidden=64, input_num=step1, output_dim=step2, dropout=0.05)
    net = net.to(device)
    net.apply(lib.init_weights)

    weight_decay_list = (param for name, param in net.named_parameters() if name[-4:] != 'bias' and 'bn' not in name)
    no_decay_list = (param for name, param in net.named_parameters() if name[-4:] == 'bias' or 'bn' in name)
    parameters = [{'params':weight_decay_list},
                  {'params':no_decay_list, 'weight_decay':0}]
    updater = torch.optim.SGD(parameters, lr, momentum=0.6, weight_decay=0.001)
    # 自适应学习率设置
    scheduler = torch.optim.lr_scheduler.LambdaLR(updater, lambda x: np.cos(x % 20 / 20) / ((x // 20) * 4 + 1))
    train_data = TensorDataset(train_X, train_y)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    best_MAE = 0.5
    for epoch in range(num_epochs):
        net.train()
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat.reshape(y.shape), y).requires_grad_()
            updater.zero_grad()
            l.backward()
            lib.grad_clipping(net, 20)            # 梯度裁剪,防止梯度爆炸
            updater.step()
        # 在测试集上评估
        if epoch > 10 and epoch % 1 == 0:
            net.eval()
            with torch.no_grad():
                predict1 = [net(x[0].to(device)).detach() for x in data_loader]
                MAE0 = lib.MAE(torch.concat(predict1, dim=0).cpu(), train_y)
                print('第%s次训练的训练集上MAE结果为:%.3f'%(epoch, MAE0))
            if MAE0 < best_MAE:
                best_MAE = MAE0
                torch.save(net.state_dict(), 'best_model_epoch/f_best_model%d' % final_epoch)
        scheduler.step()

    # 预测结果
    net.load_state_dict(torch.load('best_model_epoch/f_best_model%d' % final_epoch))
    net.eval()
    train_data = TensorDataset(predict_X)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        res = [net(x[0].to(device)).detach().cpu() for x in data_loader]
        res = torch.concat(res, dim=0).numpy()

    # 分别为 session_id, rank, pm
    column1, column2, column3 = [], [], []
    for i in range(len(group1)):
        column1.extend([i] * 12)
        column2.extend(list(range(1, 13)))
        column3.extend(res[i][::-1])
    df = pd.DataFrame({'session_id':column1, 'rank':column2, 'pm':column3})
    df.to_csv('submit_results/final_data%d.csv'%final_epoch, index=False, header=True)              # 保留列名但不要行索引

    end = time.time()
    print('程序运行时间为：{}'.format(end - start))

# 合并各个模型的预测结果
list, res = [], []
for i in range(16):
    data = pd.read_csv('submit_results/final_data%d.csv'%i)
    list.append(data['pm'].to_numpy())
list = np.stack(list)     # 大小为 16 * (id_num * 12)

for i in range(len(list[0])):
    temp = list[:, i]
    max, min = np.max(temp), np.min(temp)
    res.append((np.sum(temp) - max - min) / 14)    # 去除一个最大值和最小值

data['pm'] = pd.Series(res)
data.to_csv('final_data_test0.csv', index=False)