# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from sklearn.linear_model import LogisticRegressionCV  # 逻辑回归模型
from sklearn.linear_model.coordinate_descent import ConvergenceWarning  # 警告处理
from sklearn.model_selection import train_test_split  # 数据划分 
from sklearn.preprocessing import MinMaxScaler  # 数据标准化
from sklearn.preprocessing import label_binarize
from sklearn import metrics


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
# 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)
warnings.filterwarnings(action = 'ignore', category=UserWarning)


# 1.读取数据
path1 = "../datas/winequality-red.csv"
df1 = pd.read_csv(path1, sep=";")
df1['type'] = 1  # 设置数据类型为红葡萄酒

path2 = "../datas/winequality-white.csv"
df2 = pd.read_csv(path2, sep=";")
df2['type'] = 2  # 设置数据类型为白葡萄酒


# 2.数据预处理
# a.合并两个df
df = pd.concat([df1,df2], axis=0)
# b.自变量名称
names = ["fixed acidity","volatile acidity","citric acid",
         "residual sugar","chlorides","free sulfur dioxide",
         "total sulfur dioxide","density","pH","sulphates",
         "alcohol", "type"]
# c.因变量名称
quality = "quality"


# d.异常数据处理
new_df = df.replace('?', np.nan)
datas = new_df.dropna(how = 'any') # 只要有空，就删除所在行

# e.提取自变量和因变量
X = datas[names]
Y = datas[quality]
Y = Y.ravel()  # Y拉长为扁平的数组

# 3.数据分割
# test_size：测试集所占比例
# random_state：保证每次分割所产生的数据集是完全相同的
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# 4.数据格式化(归一化)
# 将数据缩放到[0,1]
ss = MinMaxScaler()
X_train = ss.fit_transform(X_train)  # 训练模型及归一化数据


# 5.模型构建及训练
"""
multi_class: 分类方式参数
	参数可选: ovr(默认)、multinomial
		这两种方式在二元分类问题中，效果是一样的；在多元分类问题中，效果不一样
	ovr: one-vs-rest， 对于多元分类的问题，先将其看做二元分类，分类完成后，再迭代对其中一类继续进行二元分类
	multinomial: many-vs-many（MVM）,即Softmax分类效果
# 对于多元分类问题，如果模型有T类，我们每次在所有的T类样本里面选择两类样本出来，
# 不妨记为T1类和T2类，把所有的输出为T1和T2的样本放在一起，把T1作为正例，T2作为负例，
# 进行二元逻辑回归，得到模型参数。我们一共需要T(T-1)/2次分类

penalty: 过拟合解决参数,l1或者l2
solver: 参数优化方式
	当penalty为l1的时候，参数只能是：liblinear(坐标轴下降法)；
nlbfgs和cg都是关于目标函数的二阶泰勒展开
	当penalty为l2的时候，参数可以是：lbfgs(拟牛顿法)、newton-cg(牛顿法变种)，seg(minibatch)
维度<10000时，lbfgs法比较好；维度>10000时，cg法比较好；显卡计算的时候，lbfgs和cg都比seg快
"""
lr = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-5, 1, 100), 
                          multi_class='multinomial', penalty='l2', solver='lbfgs')
lr.fit(X_train, Y_train)


# 6.模型评估
r = lr.score(X_train, Y_train)
print("R值：", r)
print("特征稀疏化比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))
print("参数：",lr.coef_)
print("截距：",lr.intercept_)
print("概率：", lr.predict_proba(X_test)) # 获取sigmoid函数返回的概率值


# 7.数据预测
# a. 预测数据格式化(归一化)
X_test = ss.transform(X_test) # 使用模型进行归一化操作
# b. 结果数据预测
Y_predict = lr.predict(X_test)


# 8. 图表展示
x_len = range(len(X_test))
plt.figure(figsize=(14,7), facecolor='w')
plt.ylim(-1,11)
plt.plot(x_len, Y_test, 'ro',markersize = 8, zorder=3, label=u'真实值')
plt.plot(x_len, Y_predict, 'go', markersize = 12, zorder=2, label=u'预测值,$R^2$=%.3f' % lr.score(X_train, Y_train))
plt.legend(loc = 'upper left')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'葡萄酒质量', fontsize=18)
plt.title(u'葡萄酒质量预测统计', fontsize=20)
plt.show()