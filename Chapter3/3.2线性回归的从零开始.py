# coding = utf-8
# 线性回归的从零开始

import random
import torch
from d2l import torch as d2l


# 1.生成数据集
def synthetic_data(w, b, num_example):
    # normal返回从单独的正态分布中提取的随机数的张量，mean：正态分布的均值，std:标准差。
    X = torch.normal(0, 1, (num_example, len(w)))
    # matmul是tensor的乘法，输入可以是高维的。 当输入是都是二维时，就是普通的矩阵乘法，和tensor.mm函数用法相同。
    y = torch.matmul(X, w) + b
    # 增加随机噪音
    y += torch.normal(0, 0.01, y.shape)

    print(y)
    print("----------------")
    print(y.reshape((-1, 1)))
    print("----------------")
    print(y.reshape(-1, 1))

    # X 每一行包含一个二维数据样本， 第二个返回值每一行包含一维标签值, -1指行数根据列数决定
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
ture_b = 4.2
features, labels = synthetic_data(true_w, ture_b, 1000)

d2l.set_figsize()
# 画图，x,y,s     s：表示的是大小，是一个标量或者是一个shape大小为(n,)的数组
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# d2l.plt.show()

# 2.读取数据集

# 随机输出对应的特征和标签， batch_size为每次输出的大小， yield 是用于多次调用时输出的值
def data_iter(batch_size, features, labels):
    num_example = len(features)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in range(0, num_example, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_example)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 3.初始化模型参数

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 线性回归模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# 损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 优化算法
def sgd(params, lr, batch_size):
    # 小批量随机梯度下降
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03               # 学习率
num_epochs = 3          # 迭代周期个数
net = linreg
loss = squared_loss
batch_size = 10
print(w, b)
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
