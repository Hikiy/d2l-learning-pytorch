# coding = utf-8
# 线性代数

import numpy as np
import torch

# 矩阵 a X b 就是将 a每一行乘以 b每一列    比如 a第一行每个数字乘以b第一列每个数字然后加起来，作为新的矩阵的第一行第一列，
# 然后a第一行每个数字乘以b第二列每个数字加起来作为新的矩阵的第一行第二列

A = np.arange(20).reshape(5, 4)
print(A)

# 转置，对角线转置
A = A.T
print(A)

X = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print(X)

# 克隆重新分配内存
B = X.clone()
print(B)

# Hadamard积 按元素相乘
print("Hadamard积:")
print(X * B)

# 元素和   按照哪个轴求和，这个轴就被抽走了，比如原本shape为 2,5,4 按照axis=1求和，
# 结果shape为 2,4 ，如果保留维度则是 2,1,4
print("元素和：")
print(B.sum())
print(B.sum(axis=0))
# 不丢失维度
print("元素和 不丢失维度：")
print(B.sum(axis=0, keepdims=True))
print(B.sum(axis=[0, 1]))
# 累积总和
print("累积总和：")
print(B)
print(B.cumsum(axis=0))

# 平均值
print("平均值:")
print(B.mean())
print(B.sum() / B.numel())

# 点积 相同位置的按元素乘积的和
print("点积:")
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(x, y, torch.dot(x, y))
# 也可以乘了后求和，结果一样
print(torch.sum(x*y))
print((x*y).sum())
# 向量积 matrix vector
print("向量积:")
e = torch.arange(10).reshape(2, 5)
g = torch.arange(10).reshape(5, 2)
f = torch.arange(5)
print(e, f, torch.mv(e, f))
# 矩阵相乘
print("矩阵相乘:")
print(e, g, torch.mm(e, g))
# 矩阵 a X b 就是将 a每一行乘以 b每一列    比如 a第一行每个数字乘以b第一列每个数字然后加起来，作为新的矩阵的第一行第一列，

# L2范数， 向量元素平方和的平方根
print("L2范数:")
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))

# L1范数， 向量元素的绝对值之和
print("L1范数:")
print(torch.abs(u).sum())

# F范数, 矩阵元素的平方和的平方根
print("F范数：")
print(torch.norm(torch.ones((4, 9))))


A = torch.arange(8, dtype=torch.float32).reshape(2, 4)

# 矩阵向量积
print(torch.mv(A, x), torch.mv(A, x).shape)

# 按元素乘
print(A * x)



