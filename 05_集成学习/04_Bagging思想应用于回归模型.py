# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings  #警告处理

from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.model_selection import train_test_split  # 数据分割
from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.ensemble import BaggingRegressor  # Bagging回归


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 警告处理
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


# 1.读取数据
names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
path = '../datas/boston_housing.data'
# 由于每条数据的格式不统一，所以可以先按一行一条记录的方式来读取，然后再进行数据预处理
fd = pd.read_csv(path, header = None)  # header = None表示没有数据对应的名称，可以给数据加上


# 2.数据处理
data = np.empty((len(fd), 14))  # 生成形状为[len(fd), 14]的空数组
# 用于预处理数据
def notEmpty(s):
    return s != ''
# 对每条记录依次处理
for i, d in enumerate(fd.values):  # enumerate生成一列索引i(表示fd中的每一条记录), d为其元素(此处d就是fd的一条记录内容)
    d = map(float, filter(notEmpty, d[0].split(' '))) # filter一个函数，一个list
    """
	d[0].split(' ')：将每条记录按空格切分，生成list，可迭代
	notEmpty:调用前面的自定义的函数，将空格表示为False，非空格表示为True
	filter(function,iterable)：将迭代器传入函数中
	map(function,iterable)：对迭代器进行function操作，这里表示根据filter结果是否为真，来过滤list中的空格项
    """
    # map操作后的类型为map类型，转为list类型，并将该条记录存在之前定义的空数组中
    data[i] = list(d)
    # 遍历完所有数据，数据也就处理好了


# 3.划分数据
X, Y = np.split(data, (13,), axis=1)  # 前13个数据划为X，最后一个划为Y
# 将Y拉直为一个扁平的数组
Y = Y.ravel()


# 4.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)


# 5.模型构建及训练
# a.线性回归模型
lr = LinearRegression()
lr.fit(X_train, Y_train)
print("训练集上R^2:%.5f" % lr.score(X_train, Y_train))
print("测试集上R^2:%.5f" % lr.score(X_test, Y_test))
# b.使用Bagging集成线性回归
bg = BaggingRegressor(LinearRegression(), n_estimators=50, max_samples=0.7, max_features=0.8, random_state=0)
bg.fit(X_train, Y_train)
print("训练集上R^2:%.5f" % bg.score(X_train, Y_train))
print("测试集上R^2:%.5f" % bg.score(X_test, Y_test))