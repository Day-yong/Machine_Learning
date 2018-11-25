# -*- coding: utf-8 -*-

"""
基于原始数据前3列比较一下决策树在不同深度的情况下错误率
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings  # 警告处理

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier  # 分类树
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import MinMaxScaler  # 数据归一化
from sklearn.feature_selection import SelectKBest  # 特征选择
from sklearn.feature_selection import chi2  # 卡方统计量
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.pipeline import Pipeline  # 管道
from sklearn.model_selection import GridSearchCV  # 网格搜索交叉验证，用于选择最优的参数


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)


# 1.读取数据
path = '../datas/iris.data'
df = pd.read_csv(path, header=None)


# 2.划分数据
X = df[list(range(4))]  # 获取X变量
Y = pd.Categorical(df[4]).codes # 获取Y，并将其转换为1,2,3类型


# 3.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=14) # random_state随机数生成种子
# DecisionTreeClassifier是分类算法，要求Y必须是int类型
Y_train = Y_train.astype(np.int)
Y_test = Y_test.astype(np.int)


# 4.模型训练
depths = np.arange(1, 15)  # 决策树深度可选值
err_list = []  # 错误率
for d in depths:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=d, min_samples_split=2)
    # 仅设置了这二个参数，没有对数据进行特征选择和降维，所以跟前面得到的结果不同
    clf.fit(X_train, Y_train)
    
    # 计算的是在测试集上的模型预测能力
    score = clf.score(X_test, Y_test)
    err = 1 - score
    err_list.append(err)
    print("%d深度，测试集上正确率%.5f" % (d, clf.score(X_train, Y_train)))
    print("%d深度，训练集上正确率%.5f\n" % (d, score))


# 5.画图
plt.figure(facecolor='w')
plt.plot(depths, err_list, 'ro-', lw=3)
plt.xlabel(u'决策树深度', fontsize=16)
plt.ylabel(u'错误率', fontsize=16)
plt.grid(True)
plt.title(u'决策树层次太多导致的拟合问题(欠拟合和过拟合)', fontsize=18)
plt.show()