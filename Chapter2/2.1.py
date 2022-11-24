# coding = utf-8

import torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

x = x.reshape(3, 4)
print(x)

print(torch.zeros(2, 3, 4))

print(torch.tensor([[1, 2, 3], [5, 7, 9]]))

# 与numpy转化
y = x.numpy()
z = torch.tensor(y)
print(type(y), type(z))