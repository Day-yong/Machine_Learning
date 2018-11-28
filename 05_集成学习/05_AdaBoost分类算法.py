# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier  # 分类AdaBoost
from sklearn.tree import DecisionTreeClassifier  # 分类决策树
from sklearn.datasets import make_gaussian_quantiles  # 造数据


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


# 1.创建数据
# 创建符合高斯分布的数据集
"""
mean:多维正态分布的均值，默认为(0,0)
cov:方差
n_samples：样本数，默认为100
n_features：每个样本的特征数，默认为2
n_classes：样本类别数，默认为3
random_state：随机数种子
"""
# 以(0,0)为均值，方差为2，样本数为200，类别为2，随机数种子为0
X1, Y1 = make_gaussian_quantiles(cov=2, n_samples=200, n_classes=2, random_state=0)
# 以(3,3)为均值，方差为1.5，样本数为300，类别为2，样本特征数为2(相当于二维的点)，随机数种子为0
X2, Y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1)

# 2.数据划分
X = np.concatenate((X1, X2))
Y = np.concatenate((Y1, - Y2 + 1))
# 本来生成的数据分布类似，标签也类似，即里面的标签和外面的标签是对应的
# -Y2 + 1就将原先为1的变为0，原先为0的变为1，从图中就可以看出
# 也可以尝试改为Y2，再看看运行结果，结果将不好划分


# 3.构建AdaBoost模型
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME.R", n_estimators=200)
bdt.fit(X, Y)  #训练模型


# 4.设置画图网格
plot_step = 0.02
X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
Y_min, Y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
XX, YY = np.meshgrid(np.arange(X_min, X_max, plot_step), np.arange(Y_min, Y_max, plot_step))

# 5.预测
Z = bdt.predict(np.c_[XX.ravel(), YY.ravel()])
"""
np.c_[np.array([1,2,3]), np.array([4,5,6])]
输出为：	
array([[1, 4],
       [2, 5],
       [3, 6]])
"""
# 设置维度
Z = Z.reshape(XX.shape)


# 6.画图
plot_colors = "br"  # 类别颜色
class_names = "AB"  # 类别名

plt.figure(figsize=(10, 5), facecolor='w')  # 画板

plt.subplot(121)  # 子图，1行2列画在第一列
plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)  # XX,YY可用于指定四边形的角；Z标量二维数组，值将进行颜色映射
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(Y == i)  
    plt.scatter(X[idx, 0], X[idx, 1], c=c, cmap=plt.cm.Paired, label=u"类别%s" % n)
plt.xlim(X_min, X_max)
plt.ylim(Y_min, Y_max)
plt.legend(loc='upper right')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(u'AdaBoost分类结果,正确率为:%.2f%%' % (bdt.score(X, Y) * 100))

# 获取决策函数的数值
twoclass_output = bdt.decision_function(X)
# 获取范围
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)  # 子图，1行2列画在第二列
for i, n, c in zip(range(2), class_names, plot_colors):
# 直方图
    plt.hist(twoclass_output[Y == i],
             bins=20,
             range=plot_range,
             facecolor=c,
             label=u'类别 %s' % n,
             alpha=.5)
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel(u'样本数')
plt.xlabel(u'决策函数值')
plt.title(u'AdaBoost的决策值')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()