# -*- coding: utf-8 -*-

"""
数据格式如下：
1000025, 5,  1,  1, 1, 2,  1, 3, 1, 1, 2
1017122, 8, 10, 10, 8, 7, 10, 9, 7, 1, 4
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings  # 警告处理

from sklearn.linear_model.coordinate_descent import ConvergenceWarning  # 警告处理
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.externals import joblib # 模型保存与加载


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 拦截异常
warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)


# 1.读取数据
# 1000025, 5, 1, 1, 1, 2, 1, 3, 1, 1, 2
path = '../datas/breast-cancer-wisconsin.data'
names = ['id','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
         'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei',
         'Bland Chromatin','Normal Nucleoli','Mitoses','Class']
df = pd.read_csv(path, header=None, names=names)


# 2.异常数据处理
datas = df.replace('?', np.nan)  # 将非法字符"?"替换为np.nan
datas = datas.dropna(how = 'any')  # 将行中有空值的行删除


# 3.数据提取以及数据分割
# 提取
# 第一列数据为id，对分类决策没有帮助，所以无需作为特征数据
X = datas[names[1:10]]
Y = datas[names[10]]


# 4.数据分割
# test_size：测试集所占比例
# random_state：保证每次分割所产生的数据集是完全相同的
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# 5.模型加载
oss = joblib.load("result/ss.model")
olr = joblib.load("result/lr.model")


# 6.模型评估
r = olr.score(X_train, Y_train)
print ("R值（准确率）：", r)
print ("稀疏化特征比率：%.2f%%" % (np.mean(olr.coef_.ravel() == 0) * 100))
print ("参数：",olr.coef_)
print ("截距：",olr.intercept_)
print(olr.predict_proba(X_test))  # 获取sigmoid函数返回的概率值


# 7.数据预测
# a. 预测数据格式化(归一化)
X_test = oss.transform(X_test) # 使用模型进行归一化操作
# b. 结果数据预测
Y_predict = olr.predict(X_test)


# 8.画图
x_len = range(len(X_test))
plt.figure(figsize=(14,7), facecolor='w')
plt.ylim(0,6)
plt.plot(x_len, Y_test, 'ro',markersize = 8, zorder=3, label=u'真实值')
plt.plot(x_len, Y_predict, 'go', markersize = 14, zorder=2, label=u'预测值,$R^2$=%.3f' % olr.score(X_test, Y_test))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'乳腺癌类型', fontsize=18)
plt.title(u'Logistic回归算法对数据进行分类', fontsize=20)
plt.show()