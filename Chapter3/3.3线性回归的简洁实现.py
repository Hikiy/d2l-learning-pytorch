# coding = utf-8
# 线性回归的简洁实现

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 1. 生成数据


# 这个和3.2的函数仙童，根据W b 生成数据
def synthetic_data(w, b, num_example):
    # normal返回从单独的正态分布中提取的随机数的张量，mean：正态分布的均值，std:标准差。
    X = torch.normal(0, 1, (num_example, len(w)))
    # matmul是tensor的乘法，输入可以是高维的。 当输入是都是二维时，就是普通的矩阵乘法，和tensor.mm函数用法相同。
    y = torch.matmul(X, w) + b
    # 增加随机噪音
    y += torch.normal(0, 0.01, y.shape)

    # X 每一行包含一个二维数据样本， 第二个返回值每一行包含一维标签值, -1指行数根据列数决定
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 2.调用框架现有API来读取数据


# 构造pytorch迭代器
def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
# 转换为python的
next(iter(data_iter))

# 3.使用框架定义好的层

# ‘nn’是神经网络的缩写
from torch import nn
# Sequential 理解为一个容器，一层层放神经网络， Linear 线性神经网络，输入2，输出1
net = nn.Sequential(nn.Linear(2, 1))

# 初始化设置W和b，原本就会进行随机初始化
print(net[0].weight.data)
print(net[0].bias.data)
net[0].weight.data.normal_(0, 0.001)
net[0].bias.data.fill_(0)
print(net[0].weight.data)
print(net[0].bias.data)

# 4.定义损失函数
loss = nn.MSELoss()

# 5.定义优化函数
# SGD：stochastic Gradient Descent 随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 6.训练

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # 计算当前参数下损失
        l = loss(net(X), y)
        # 清空梯度，否则会叠加
        trainer.zero_grad()
        # 计算梯度
        l.backward()
        # step：执行单个优化步骤
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

