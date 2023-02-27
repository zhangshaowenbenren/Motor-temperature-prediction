# 基于AdaBoost的集成学习模型
# 基模型采用GRU_NN

import random
import pandas as pd
import torch
import lib
import time
import numpy as np
import pickle


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
step1, step2 = 30, 12                     # LSTM的输入和输出步长
weight = torch.tensor([1.])
data_X, data_Y, predict_X = [], [], []    # predict_X为需要提交结果的输入数据，对应每个片段的前step1行数据
test_X, test_y = [], []
for x, y in zip(X, Y):
    for i in range(step2, len(x) - step1 + 1, 1):
        if i - step2 > 15:
            x1 = torch.tensor(x[i:i + step1, :], dtype=torch.float32)
            y1 = torch.tensor(y[i - step2:i], dtype=torch.float32)
            y1 = torch.concat((y1, weight))
            data_X.append(torch.flip(x1, dims=[0]))
            data_Y.append(torch.flip(y1, dims=[0]))           # data_Y的第一列为权重
        else:
            x1 = torch.tensor(x[i:i + step1, :], dtype=torch.float32)
            y1 = torch.tensor(y[i - step2:i], dtype=torch.float32)
            y1 = torch.concat((y1, weight))
            test_X.append(torch.flip(x1, dims=[0]))
            test_y.append(torch.flip(y1, dims=[0]))  # data_Y的第一列为权重
    predict_X.append(torch.flip(torch.tensor(x[:step1, :], dtype=torch.float32), dims=[0]))

train_X, train_y, predict_X = torch.stack(data_X), torch.stack(data_Y), torch.stack(predict_X)
test_X, test_y = torch.stack(test_X), torch.stack(test_y)
# 样本权重归一化
train_y[:, 0] = 1 / torch.sum(train_y[:, 0]) * train_y[:, 0]
# 打乱样本顺序
index = list(np.arange(len(train_X)))
random.shuffle(index)
train_X, train_y = train_X[index], train_y[index]

net = lib.GRU_NN(input_dim=len(key), num_hidden=64, output_num=step2, dropout=0.1)
lr, num_epochs, batch_size = 0.01, 100, 256
n_estimators = 6                                # AdaBoost基学习器的数量
device = lib.try_gpu()

adaBoost = lib.AdaBoostRegressor(net, n_estimators=n_estimators)
adaBoost.fit(train_X, train_y, test_X, test_y, lib.loss, lr, num_epoch=num_epochs, batch_size=batch_size, device=device)

predict_X = predict_X.to(device)
res = adaBoost.predict(predict_X, device, n_estimators - 1)

with open('adaBoost.pkl', 'wb') as fp:
    pickle.dump(adaBoost, fp)

print('AdaBoost的每个学习的最后几次的测试误差：')
print(adaBoost.models_train_loss[:][-5:])
print('AdaBoost的各学习器集成的测试误差：')
print(adaBoost.AdaBoost_error)

# 分别为 session_id, rank, pm
column1, column2, column3 = [], [], []
for i in range(len(group1)):
    column1.extend([i] * 12)
    column2.extend(list(range(1, 13)))
    column3.extend(torch.flip(res[i], dims=[0]).numpy())           # 翻转输出值
df = pd.DataFrame({'session_id':column1, 'rank':column2, 'pm':column3})
df.to_csv('data.csv', index=False, header=True)            # 保留列名但不要行索引
end = time.time()
print('程序运行时间为：{}'.format(end - start))