# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings # 警告处理

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV  # 回归模型
from sklearn.preprocessing import PolynomialFeatures  # 模型特征的构造
from sklearn.pipeline import Pipeline  # 管道
from sklearn.linear_model.coordinate_descent import ConvergenceWarning  # 警告处理
from sklearn.model_selection import train_test_split  # 数据集划分


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

X = datas[names]
Y = datas[quality]
Y = Y.ravel()  # Y拉长为扁平的数组


# 3.管道
# 创建模型列表
models = [
    Pipeline([
            ('Poly', PolynomialFeatures()), # 模型特征的构造
            ('Linear', LinearRegression())  # 线性回归
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', RidgeCV(alphas=np.logspace(-4, 2, 20)))  # RidgeCV模型，alphas学习率
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', LassoCV(alphas=np.logspace(-4, 2, 20)))  # LassoCV模型
        ]),
    Pipeline([
            ('Poly', PolynomialFeatures()),
            ('Linear', ElasticNetCV(alphas=np.logspace(-4,2, 20), l1_ratio=np.linspace(0, 1, 5)))  # ElasticNetCV模型
        ])
]



# 4.划分数据
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.01, random_state=0)


# 5.模型训练，画图

# 画板
plt.figure(figsize=(16,8), facecolor='w')
# 标题
titles = u'线性回归预测', u'Ridge回归预测', u'Lasso回归预测', u'ElasticNet预测'
ln_x_test = range(len(X_test))

# 给定阶以及颜色
d_pool = np.arange(1,4,1) # 1 2 3 阶
m = len(d_pool)
clrs = [] # 颜色
for c in np.linspace(5570560, 255, m):
    clrs.append('#%06x' % int(c))

# 模型训练
for t in range(4):
    plt.subplot(2, 2, t + 1)  # 子图
    model = models[t]
    plt.plot(ln_x_test, Y_test, c='r', lw=2, alpha=0.75, zorder=10, label=u'真实值')
    for i,d in enumerate(d_pool):
        # 设置参数
        model.set_params(Poly__degree=d)
        # 模型训练
        model.fit(X_train, Y_train)
        # 模型预测及计算R^2
        Y_pre = model.predict(X_test)
        R = model.score(X_train, Y_train)
        # 输出信息
        lin = model.get_params('Linear')['Linear']
        output = u"%s:%d阶, 截距:%d, 系数:" % (titles[t], d, lin.intercept_)
        print(output, lin.coef_)
        # 图形展示
        plt.plot(ln_x_test, Y_pre, c=clrs[i], lw=2,alpha=0.75, zorder=i, label=u'%d阶预测值,$R^2$=%.3f' % (d,R))
    plt.legend(loc = 'upper left')
    plt.grid(True)
    plt.title(titles[t], fontsize=18)
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
plt.suptitle(u'葡萄酒质量预测', fontsize=22)
plt.show()