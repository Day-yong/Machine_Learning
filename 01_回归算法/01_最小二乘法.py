# -*- coding: utf-8 -*-
# 引入所需要的全部包
from sklearn.model_selection import train_test_split # 数据划分的类

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

# 加载数据
# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
path1='../datas/household_power_consumption_1000.txt'
df = pd.read_csv(path1, sep=';', low_memory=False)  # 没有混合类型的时候可以通过low_memory=False调用更多内存，加快效率

## 功率和电流之间的关系
X = df.iloc[:,2:4]
Y2 = df.iloc[:,5]

## 数据分割
X2_train,X2_test,Y2_train,Y2_test = train_test_split(X, Y2, test_size=0.2, random_state=0)

# 将X和Y转换为矩阵的形式
X = np.mat(X2_train)
Y = np.mat(Y2_train).reshape(-1,1)

# 计算θ
theta = (X.T * X).I * X.T * Y

# 对测试集合进行测试
y_hat = np.mat(X2_test) * theta

# 画图看看
# 功率与电流关系
t=np.arange(len(X2_test))
plt.figure(facecolor='w')
plt.plot(t, Y2_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc = 'lower right')
plt.title(u"线性回归预测功率与电流之间的关系", fontsize=20)
plt.grid(b=True)
plt.show()