# coding = utf-8
# 自动求导

import torch

x = torch.arange(4.0)
print(x)
x.requires_grad_(True)
print(x.grad)

y = 2* torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
print(x.grad == 4*x)

# 默认情况下 pytorch会累积梯度，需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x*x
y.sum().backward()
print(x.grad)

x.grad.zero_()
y = x*x
u = y.detach()
z = u*x
z.sum().backward()
print(x.grad == u)

def f(a):
    b = a*2
    while b.norm() < 1000:
        b = b*2
    if b.sum() > 0:
        c = b
    else:
        c = 100*b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d/a)
