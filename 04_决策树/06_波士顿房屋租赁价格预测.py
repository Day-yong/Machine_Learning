# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings  # 警告处理

from sklearn.linear_model.coordinate_descent import ConvergenceWarning  # 警告处理
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV  # 回归模型
from sklearn.tree import DecisionTreeRegressor  # 回归决策树模型
from sklearn.model_selection import train_test_split  # 数据分割
from sklearn.preprocessing import MinMaxScaler  # 数据归一化 



# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
# 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)

# 1.读取数据
names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
path = "../datas/boston_housing.data"
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
# 查看下数据
# print(y.shape)
# print ("样本数据量:%d, 特征个数：%d" % x.shape)
# print ("target样本数据量:%d" % y.shape[0])


# 4.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=14)
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (X_train.shape[0], X_test.shape[0]))


# 5.数据归一化
ss = MinMaxScaler()  # 创建归一化模型
X_train = ss.fit_transform(X_train, Y_train)  # 训练模型并转换数据
X_test = ss.transform(X_test)  # 转换数据
print ("原始数据各个特征属性的调整最小值:",ss.min_)
print ("原始数据各个特征属性的缩放数据值:",ss.scale_)


# 6.模型构建，并训练，预测，评估
## a.决策树模型
dtr = DecisionTreeRegressor(criterion='mae',max_depth=7)  # 构建回归决策树模型，使用平均绝对误差
dtr.fit(X_train, Y_train)  # 模型训练
dtr_y_hat = dtr.predict(X_test)  # 模型预测
dtr_score = dtr.score(X_test, Y_test)  # 模型评估
print("回归决策树正确率：%.2f%%" % (dtr_score * 100))
## b.线性回归模型
lr = LinearRegression() # 构建线性回归模型
lr.fit(X_train, Y_train)  # 模型训练
lr_y_hat = lr.predict(X_test)  # 模型预测
lr_score = lr.score(X_test, Y_test)  # 模型评估
print("线性回归正确率：%.2f%%" % (lr_score * 100))
## c.Lasso回归模型
ls = LassoCV(alphas=np.logspace(-3,1,20)) # 构建LASSO模型
ls.fit(X_train, Y_train)  # 模型训练
ls_y_hat = ls.predict(X_test)  # 模型预测
ls_score = ls.score(X_test, Y_test)  # 模型评估
print("Lasso回归正确率：%.2f%%" % (ls_score * 100))
## d.Ridge回归模型
rg = RidgeCV(alphas=np.logspace(-3,1,20)) # 构建LASSO模型
rg.fit(X_train, Y_train)  # 模型训练
rg_y_hat = rg.predict(X_test)  # 模型预测
rg_score = rg.score(X_test, Y_test)  # 模型评估
print("Ridge回归正确率：%.2f%%" % (rg_score * 100))


# 7.画图
plt.figure(figsize=(12,6), facecolor='w')  # 大小为(12,6)的白画板
ln_x_test = range(len(X_test))
plt.plot(ln_x_test, Y_test, 'r-', lw=2, label=u'真实值')
plt.plot(ln_x_test, lr_y_hat, 'b-', lw=2, label=u'Linear回归，$R^2$=%.3f' % lr_score)
plt.plot(ln_x_test, ls_y_hat, 'y-', lw=2, label=u'Lasso回归，$R^2$=%.3f' % ls_score)
plt.plot(ln_x_test, rg_y_hat, 'c-', lw=2, label=u'Ridge回归，$R^2$=%.3f' % rg_score)
plt.plot(ln_x_test, dtr_y_hat, 'g-', lw=4, label=u'回归决策树预测值，$R^2$=%.3f' % dtr_score)
plt.xlabel(u'数据编码')
plt.ylabel(u'租赁价格')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.title(u'波士顿房屋租赁数据预测')
plt.show()

