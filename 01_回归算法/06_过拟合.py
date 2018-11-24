# -*- coding: utf-8 -*-

"""
过拟合样例
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.preprocessing import PolynomialFeatures # 特征的构造
from sklearn.pipeline import Pipeline # 管道


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False


# 创建模拟数据
np.random.seed(100)  # 随机数种子，是每次生成的随机数相同
# 显示方式设置，linewidth每行的字符数用于插入换行符，suppress是否使用科学计数法
np.set_printoptions(linewidth=1000, suppress=True) 
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)  # 0-6中等分生成10个数，并加上随机数
y = 1.8*x**3 + x**2 - 14*x - 7 + np.random.randn(N) 
# 将其设置为矩阵
x.shape = -1, 1
y.shape = -1, 1

# 管道
models = [
    Pipeline([
            ('Poly', PolynomialFeatures(include_bias=False)),
            ('Linear', LinearRegression(fit_intercept=False))
        ])
]


# 模型训练，画图（只画1,4,7,10阶）
plt.figure(figsize=(12,8), facecolor='w')
degree = np.arange(1,12,2) # 1,3,5,7,9,11阶
dm = degree.size
colors = [] # 颜色
for c in np.linspace(16711680, 255, dm):
    colors.append('#%06x' % int(c))

model = models[0]


for i,d in enumerate(degree):
    plt.subplot(int(np.ceil(dm/2.0)),2,i+1)  # 两行，两列，正序画

    plt.plot(x, y, 'ro', ms=10, zorder=N)  # 真实值，红色的点

    # 设置阶数
    model.set_params(Poly__degree = d)
    # 模型训练
    model.fit(x, y.ravel())
    
    lin = model.get_params()['Linear']
    output = u'%d阶，系数为：' % (d)
    # 判断lin对象中是否有对应的属性
    if hasattr(lin, 'alpha_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'alpha=%.6f, ' % lin.alpha_) + output[idx:]
    if hasattr(lin, 'l1_ratio_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'l1_ratio=%.6f, ' % lin.l1_ratio_) + output[idx:]
    print (output, lin.coef_.ravel())
    
    x_hat = np.linspace(x.min(), x.max(), num=100) # 产生模拟数据
    x_hat.shape = -1,1
    # 模型预测
    y_hat = model.predict(x_hat)
    # 模型评估
    s = model.score(x, y)
    
    # 画图
    z = N - 1 if (d == 2) else 0
    label = u'%d阶, 正确率=%.3f' % (d,s)
    plt.plot(x_hat, y_hat, color=colors[i], lw=2, alpha=0.75, label=label, zorder=z)
    
    plt.legend(loc = 'upper left')
    plt.grid(True)
    plt.xlabel('X', fontsize=10)
    plt.ylabel('Y', fontsize=10)

plt.tight_layout(1, rect=(0,0,1,0.95))
plt.suptitle(u'线性回归过拟合显示', fontsize=15)
plt.show()
