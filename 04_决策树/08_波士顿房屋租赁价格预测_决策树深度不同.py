# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings  # 警告处理

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor  # 回归决策树模型
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import MinMaxScaler  # 数据归一化
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.pipeline import Pipeline  # 管道
from sklearn.model_selection import GridSearchCV  # 网格搜索交叉验证，用于选择最优的参数


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)

# 用于预处理数据
def notEmpty(s):
    return s != ''

# 1.读取数据
names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
path = "../datas/boston_housing.data"
# 由于每条数据的格式不统一，所以可以先按一行一条记录的方式来读取，然后再进行数据预处理
fd = pd.read_csv(path, header = None)  # header = None表示没有数据对应的名称，可以给数据加上


# 2.数据处理
data = np.empty((len(fd), 14))  # 生成形状为[len(fd), 14]的空数组
# 对每条记录依次处理
for i, d in enumerate(fd.values):  # enumerate生成一列索引i(表示fd中的每一条记录), d为其元素(此处d就是fd的一条记录内容)
    d = map(float, filter(notEmpty, d[0].split(' '))) # filter一个函数，一个list
    data[i] = list(d)
    # 遍历完所有数据，数据也就处理好了


# 3.划分数据
X, Y = np.split(data, (13,), axis=1)  # 前13个数据划为X，最后一个划为Y
Y = Y.ravel()


# 4.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=14)
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (X_train.shape[0], X_test.shape[0]))


# 5.模型训构建，训练，评估
depths = np.arange(1, 20)
err_list = []
for d in depths:
    clf = DecisionTreeRegressor(criterion='mse', max_depth=d)  # max_depth决策树深度
    clf.fit(X_train, Y_train)
    
    score1 = clf.score(X_test, Y_test)
    err = 1 - score1
    err_list.append(err)
    print ("%d深度，正确率%.5f" % (d, score1))

## 画图
plt.figure(facecolor='w')
plt.plot(depths, err_list, 'ro-', lw=3)
plt.xlabel(u'决策树深度', fontsize=16)
plt.ylabel(u'错误率', fontsize=16)
plt.grid(True)
plt.title(u'决策树层次太多导致的拟合问题(欠拟合和过拟合)', fontsize=18)
plt.show()