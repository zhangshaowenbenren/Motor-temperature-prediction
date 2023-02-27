from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import copy
from torch.nn import functional as F
import random

def evaluate_baseline(data):
    scaler = StandardScaler()

    scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    trainset = scaled_data.loc[:, [c for c in data.columns if c not in ['profile_id', 'pm']]]
    target = scaled_data.loc[:, 'pm']

    # Use k-fold CV to measure generalizability
    ols = LinearRegression(fit_intercept=False)
    print('Start fitting OLS...')
    scores = cross_val_score(ols, trainset, target, cv=5, scoring='neg_mean_squared_error')
    print(f'OLS MSE: {-scores.mean():.4f} (+/- {scores.std() * 2:.3f})\n')  # mean and 95% confidence interval

    rf = RandomForestRegressor(n_estimators=20, n_jobs=-1)
    print('Start fitting RF...')
    scores = cross_val_score(rf, trainset, target, cv=5, scoring='neg_mean_squared_error')
    print(f'RF MSE: {-scores.mean():.4f} (+/- {scores.std() * 2:.3f})\n')  # mean and 95% confidence interval


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def getTrainAndTest(X, Y, step1=(12, 30), step2=12):
    data_X, data_Y, predict_X = [], [], []  # predict_X为需要提交结果的输入数据，对应每个片段的前step1行数据
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
        predict_X.append(torch.flip(torch.tensor(x[:step1, :], dtype=torch.float32), dims=[0]))
    train_X, train_y, predict_X = torch.stack(data_X), torch.stack(data_Y), torch.stack(predict_X)
    test_X, test_y = torch.stack(test_X), torch.stack(test_y)
    # 依次返回训练集特征、标签；测试集特征、标签以及最终的预测输入
    return train_X, train_y, test_X, test_y, predict_X


# 计算训练样本上的平均百分误差
def MAE(hat, value):
    # hat = hat.reshape(value.shape)
    # Union_set = np.array((value, hat)).T
    mape = torch.mean(torch.abs(value - hat))
    return mape


# GRU + 残差链接
class Res_GRU(nn.Module):
    def __init__(self, input_dim=11, num_hidden=64, input_num=25, output_dim=12, dropout=0.1, **kwargs):
        super(Res_GRU, self).__init__(**kwargs)
        self.Linear = nn.Sequential(nn.Linear(input_dim, num_hidden),nn.ReLU())
        self.GRU1 = nn.GRU(num_hidden, num_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.Dropout1 = nn.Dropout(dropout)
        self.mid_layer1 =  nn.Sequential(nn.Linear(num_hidden,num_hidden*2),nn.ReLU())     # 第一个残差块的分支
        self.mid_layer2 = nn.Sequential(nn.Linear(num_hidden*2,num_hidden*2), nn.ReLU())   # 第二个残差块的分支
        #加全连接层有效
        num_hidden = num_hidden*2
        self.Linear1 = nn.Sequential(nn.Linear(num_hidden, num_hidden//2 ) ,nn.ReLU(),
                                    nn.Linear(num_hidden // 2, num_hidden), nn.ReLU(),
                                    nn.Linear(num_hidden, num_hidden), nn.ReLU())

        self.GRU2 = nn.GRU(num_hidden, num_hidden//2,num_layers=1,bidirectional=True, batch_first=True)
        self.Linear2 = nn.Sequential(nn.Linear(num_hidden, num_hidden//2 ),nn.ReLU(),
                                    nn.Linear(num_hidden // 2, num_hidden), nn.ReLU(),
                                    nn.Linear(num_hidden, num_hidden), nn.ReLU(),
                                    nn.AdaptiveAvgPool1d(1))
        self.Dropout2 = nn.Dropout(dropout)
        self.output_LR = nn.Linear(input_num, output_dim)


    def forward(self, inputs):
        # inputs的形状为 (批量 * 循环 * 采样点)
        y = self.Linear(inputs)
        y0, h = self.GRU1(y)
        y1 = self.mid_layer1(y) + y0
        y2 = self.Dropout1(y1)               # 取最后output_dim个步长对应的数据

        y2 = self.Linear1(y2)
        y2, h = self.GRU2(y2, h)

        y3 = self.mid_layer2(y1) + y2
        y3 = self.Dropout2(y3)               # 取最后output_dim个步长对应的数据
        y3 = self.Linear2(y3)
        output = self.output_LR(y3.reshape(y3.shape[0], -1))    # 线性层输入最后一个时间步对应的output
        return output


# LSTM + NNs
class GRU_NN(nn.Module):
    def __init__(self, input_dim=11, num_hidden=64, output_num=12, dropout=0.2, num_layers=1, **kwargs):
        super(GRU_NN, self).__init__(**kwargs)
        self.GRU = nn.GRU(input_dim, num_hidden, num_layers=num_layers)
        self.LR0 = nn.Sequential(nn.Linear(num_hidden, num_hidden // 2), nn.ReLU(),
                                     nn.Linear(num_hidden // 2, num_hidden), nn.ReLU())

        self.GRU1 = nn.GRU(num_hidden, num_hidden, num_layers=num_layers)
        self.Dropout = nn.Dropout(dropout)
        self.Linear = nn.Sequential(nn.Linear(num_hidden, num_hidden // 2), nn.ReLU(),
                                    nn.Linear(num_hidden // 2, num_hidden // 4), nn.ReLU(),
                                    nn.Linear(num_hidden // 4, num_hidden // 8), nn.ReLU(),
                                    nn.Linear(num_hidden // 8, 1), nn.ReLU())
        self.output_dim = output_num
        self.num_hidden = num_hidden


    def forward(self, inputs):
        # inputs的形状为 (批量 * 循环 * 采样点)
        y = torch.transpose(inputs, 0, 1)
        y, _ = self.GRU(y)
        y = self.LR0(y)
        y, _ = self.GRU1(y)
        y = self.Dropout(y[-self.output_dim:])     # 取最后output_dim个步长对应的数据
        y = y.reshape((-1, self.num_hidden))       # 转化为二维数据
        output = self.Linear(y)                    # 线性层输入最后一个时间步对应的output
        return output.reshape((self.output_dim, -1)).T


def init_weights(m):
    if type(m) == nn.Linear:
        # nn.init.xavier_normal_(m.weight)
        nn.init.xavier_uniform_(m.weight)


def data_iter(data_X, data_Y, bs=128):
    for X, Y in zip(data_X, data_Y):
        lens = len(X)
        l = X.shape[0]
        for i in range(int(np.ceil(lens / bs))):
            train_x = X[i * bs:min((i + 1) * bs, l), :]
            train_y = Y[i * bs:min((i + 1) * bs, l)]
            yield train_x, train_y


def data_iter2(X, Y, step1, step2, batch_size):
    data_X, data_Y = [], []
    for x, y in zip(X, Y):
        gap = np.random.randint(step2)         # 跳过随机步长
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        l = len(y)
        for i in range(2 * step2 + gap, l - step1[0], step2):
            for j in range(*step1):
                # 采用0填充使各预测步长相同
                if i + j > l:
                    break
                x0 = torch.zeros((step1[1] - 1), x.shape[1])
                x0[-j:] = torch.flip(x[i:i+j], dims=[0])
                data_X.append(x0)
                data_Y.append(torch.flip(y[i-step2:i], dims=[0]))
    data_X, data_Y = torch.stack(data_X), torch.stack(data_Y)        # 分别为三维和二维
    train_data = TensorDataset(data_X, data_Y)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return data_X, data_Y, data_loader


def loss(y_hat, y):
    weight, y1 = y[:, 0].reshape((-1, 1)), y[:, 1:]       # weight是权重，y1是理论值
    y_hat = y_hat.reshape(y1.shape)
    temp = torch.sum((y_hat - y1) ** 2, dim=1)
    weight = weight / torch.mean(weight)                  # 权重是归一化的，因此损失值非常小。适当放大损失值
    # weight = weight.repeat(1, temp.shape[1])
    return torch.mean(temp * weight)


def train_net(net, train_X, train_y, loss, lr, num_epoch, batch_size, device):
    net = net.to(device)
    net.apply(init_weights)
    train_data = TensorDataset(train_X, train_y)
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    epoch_error = []                  # 存储每代的预测损失值
    updater = torch.optim.SGD(net.parameters(), lr)
    # 自适应学习率设置
    scheduler = torch.optim.lr_scheduler.StepLR(updater, step_size=10, gamma=0.8)
    for epoch in range(num_epoch):
        net.train()
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y).requires_grad_()        # 将y的第一列作为样本的权重
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 10)  # 梯度裁剪
            updater.step()
        scheduler.step()
        if epoch % 2 == 0:
            net.eval()
            with torch.no_grad():
                predict1 = [net(x[0].to(device)).detach() for x in data_loader]
                MAE0 = MAE(torch.concat(predict1, dim=0).to(torch.device('cpu')), train_y[:, 1:])
            epoch_error.append(MAE0)
            print('epoch:%s, MAE:%s' % (epoch, MAE0))
    return torch.tensor(epoch_error)


class AdaBoostRegressor():
    def __init__(self, base_estimator, n_estimators=10):
        if isinstance(isinstance, (list, tuple)):
            self.models = base_estimator                 # 存储每次循环的学习器
        else:
            base_estimator.apply(init_weights)
            self.models = [base_estimator]
            for i in range(n_estimators - 1):
                net0 = copy.deepcopy(base_estimator)
                net0.apply(init_weights)
                self.models.append(net0)
        self.models_weight = [1] * n_estimators                 # 每次迭代的基学习器的权重
        self.models_train_loss = [0] * n_estimators             # 每次迭代的基学习器的训练误差
        self.AdaBoost_error = []                                # 集成学习器相应数量学习器对应的测试误差


    # 训练第i个模型
    def fit(self, train_X, train_y, test_X, test_y, loss, lr, num_epoch, batch_size, device):
        for i, model in enumerate(self.models):
            print('开始训练第%s个基学习器...'%(i))
            epoch_loss = train_net(model, train_X, train_y, loss, lr, num_epoch, batch_size, device)    # 1.训练模型，返回每隔n（2）代的测试误差
            self.models_train_loss[i] = epoch_loss
            model.eval()
            with torch.no_grad():
                train_data = TensorDataset(train_X)
                data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
                y_hat = torch.concat([model(x[0].to(device)).detach() for x in data_loader], dim=0)
            erros = y_hat.to(torch.device('cpu')) - train_y[:, 1:]                   # 训练集中样本的测试误差
            Et = torch.max(torch.sum(torch.abs(erros), dim=1))                       # 2.model在训练集上的最大误差。每个样本多个误差值绝对值相加
            rel_error = torch.sum(torch.abs(erros), dim=1) ** 2 / (Et ** 2)          # 3.计算相对平方误差
            error_rate = torch.dot(train_y[:, 0], rel_error)       # 4.当前模型的误差率
            weight = error_rate / (1 - error_rate)                 # 5.当前模型的权重
            self.models_weight[i] = weight
            Zt = torch.dot(train_y[:, 0], torch.pow(weight, 1 - rel_error))
            train_y[:, 0] = train_y[:, 0] / Zt * torch.pow(weight, 1 - rel_error)
            test_error = MAE(self.predict(test_X, device, i), test_y[:, 1:])
            self.AdaBoost_error.append(test_error.numpy())


    # 对样本数据集进行回归预测, L为当前训练的学习器数量（从0计数）
    def predict(self, X, device, L):
        res = []
        for i, model in enumerate(self.models):
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                res.append(model(X.to(device)).detach().to(torch.device('cpu')))
            if i == L:
                break
        temp = 0
        for i, data in enumerate(res):
            temp += data * self.models_weight[i]
        return temp / sum(self.models_weight[:L+1])


def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
        # for p in params:
        #     print(p.grad)
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def get_res(data_set, l):
    res = []
    for data in data_set:      # data大小为 l * l
        temp = data[0]
        for i in range(l):
            n_value = []
            for j in range(min(l - i, len(data))):
                n_value.append(data[j, i + j])
            temp[i] = np.median(n_value)           # 取中位数
        res.append(temp)
    return res