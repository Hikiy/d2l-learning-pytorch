# coding = utr-8

import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data = pd.read_csv(data_file)
# data = pd.read_csv("../data./house_tiny.csv")
print(data)

# iloc： index location
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

# 非数值的统计方法
# 方案1 新建类型
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

import torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)
print(X.dim())
