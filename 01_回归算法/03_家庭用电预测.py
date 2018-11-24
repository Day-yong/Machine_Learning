# -*- coding: utf-8 -*-

"""
功率与电流的线性关系
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split  # 数据划分的类
from sklearn.linear_model import LinearRegression  # 线性回归的类
from sklearn.preprocessing import StandardScaler  # 数据标准化的类


## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


# 1.加载数据
# 数据内容：日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
path1='../datas/household_power_consumption_1000.txt'
df = pd.read_csv(path1, sep=";", low_memory=False)  # 没有混合类型数据的时候可以通过low_memory=False调用更多内存，加快效率


# 2.数据预处理--异常数据处理
new_df = df.replace('?', np.nan)  # 替换非法字符'?'为np.nan
datas = new_df.dropna(axis=0, how = 'any')
# axis : {0 or 'index', 1 or 'columns'}
#    * 0, or 'index' : 只要行中有一个数据为空，就删除此行
#    * 1, or 'columns' : 只要列中有一个数据为空，就删除此列
# how : {'any', 'all'}
#    * 'any' : 只要有一个为空，就删除
#    * 'all' : 全部为空，就删除


# 3.数据准备
# 需求：功率与电流的关系，可以认为：特征属性为功率；目标属性为电流值
# 获取X和Y
X = datas.iloc[:, 2:4]  # 获取功率，第2、3列
Y = datas.iloc[:, 5]# 获取电流，第5列

# 4.数据分割
# X：特征矩阵(类型一般是DataFrame)
# Y：特征对应的Label标签(类型一般是Series)；Series结构是基于NumPy的ndarray结构，是一个一维的标签矩阵
# test_size: 对X/Y进行划分的时候，测试集合的数据占比, 是一个(0,1)之间的float类型的值
# random_state: 数据分割是基于随机器进行分割的，该参数给定随机数种子；给一个值(int类型)的作用就是保证每次分割所产生的数据集是完全相同的
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# 5.数据归一化
scaler = StandardScaler()  # 创建数据标准化模型
# ss.fit(X_train) # 模型训练
# X_train = xx.transform(X_train) # 对训练集合数据进行转换
X_train = scaler.fit_transform(X_train)  # 训练模型并转换训练集
X_test = scaler.transform(X_test)  # 对测试集进行转换


# 6.模型训练(训练集上)
lr = LinearRegression()  # 线性回归模型构建
lr.fit(X_train, Y_train)  # 训练模型


# 7.预测结果(测试集上)
y_predict = lr.predict(X_test)


# 8.模型评估
print("电流预测准确率：", lr.score(X_test, Y_test))
print("电流参数：", lr.coef_)


# 9.绘图
t = np.arange(len(X_test))
plt.figure(facecolor='w')  # 设置背景色
plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')  # 红色画真实值
plt.plot(t, y_predict, 'b-', linewidth=2, label=u'预测值')  # 蓝色画预测值
plt.legend(loc = 'upper left')  # 设置label位置
plt.title(u"线性回归预测功率与电流之间的关系", fontsize=20)  # 设置标题
plt.grid(b=True)  # 加网格
plt.show()