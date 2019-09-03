# -*- coding: utf-8 -*-

"""
我们将建立一个逻辑回归模型来预测一个学生是否被大学录取。
假设你是一个大学系的管理员，你想根据两次考试的结果来决定每个申请人的录取机会。
你有以前的申请人的历史数据，你可以用它作为逻辑回归的训练集。
对于每一个培训例子，你有两个考试的申请人的分数和录取决定。
为了做到这一点，我们将建立一个分类模型，根据考试成绩估计入学概率。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1.加载数据
path = '../datas/LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
# print(pdData.head())
# print(pdData.shape)
"""
      Exam 1     Exam 2  Admitted
0  34.623660  78.024693         0
1  30.286711  43.894998         0
2  35.847409  72.902198         0
3  60.182599  86.308552         1
4  79.032736  75.344376         1
(100, 3)
"""

# 2.数据展示
positive = pdData[pdData['Admitted'] == 1]  # 返回Admitted为1的数据
negative = pdData[pdData['Admitted'] == 0]

fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')  # 横坐标
ax.set_ylabel('Exam 2 Score')  # 纵坐标
# plt.show()


# 3.目标：建立分类器（求解出三个参数θ0、θ1、θ2）
# 需要完成的模块
"""
sigmoid : 映射到概率的函数
model : 返回预测结果值
cost : 根据参数计算损失
gradient : 计算每个参数的梯度方向
descent : 进行参数更新
accuracy: 计算精度
"""

# 3.1 sigmoid函数
def sigmoid(z):

	return 1/(1 + np.exp(-z))
# 画出sigmoid图
nums = np.arange(-10, 10, step=1)  # 生成-10到10的向量（含头不含尾），步伐为1，即[-10,-9,-8,...,8,9]
# print(nums)
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(nums, sigmoid(nums), 'r')
# plt.show()

# 3.2 model
def model(X, theta):
    
    return sigmoid(np.dot(X, theta.T))

pdData.insert(0, 'Ones', 1)
# print(pdData)
# 获取X,y
orig_data = pdData.as_matrix()  # 将panda表示的数据转换为数组，以便进行下一步计算
cols = orig_data.shape[1]  # 数据的列数
X = orig_data[:,0:cols-1]  # 前cols-1列为特征
y = orig_data[:,cols-1:cols]  # 最后一列为标签
# 初始化theta数组
theta = np.zeros([1, 3])
print(X[:5])