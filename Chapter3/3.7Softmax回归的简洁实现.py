import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# Pytorch不会隐式调整输入形状， 所以需要调用展平层flatten（）在线性层前调整网络输入形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


# m为当前Linear
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# 会将net中每一层跑一边此函数
net.apply(init_weights)

loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
