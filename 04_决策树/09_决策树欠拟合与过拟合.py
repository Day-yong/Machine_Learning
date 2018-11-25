# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor  # 回归决策树


# 1.数据准备
rng = np.random.RandomState(1)  # 随机数种子，使每次执行程序的时候，随机参数的随机数都一样
X = np.sort(10 * rng.rand(80, 1), axis=0)  # 原数据X
y = np.sin(X).ravel()  # 原数据X对应的Y值
y[::5] += 3 * (0.5 - rng.rand(16))  # 加上噪声

# 2.构造不同深度的回归决策树
clf_0 = DecisionTreeRegressor(max_depth=1)
clf_1 = DecisionTreeRegressor(max_depth=2)
clf_2 = DecisionTreeRegressor(max_depth=3)
clf_3 = DecisionTreeRegressor(max_depth=5)

# 3.训练模型
clf_0.fit(X, y)
clf_1.fit(X, y)
clf_2.fit(X, y)
clf_3.fit(X, y)

# 4.创建预测模拟数据
X_test = np.arange(0.0, 10, 0.01)[:, np.newaxis]  # 测试数据
y_0 = clf_0.predict(X_test)
y_1 = clf_1.predict(X_test)
y_2 = clf_2.predict(X_test)
y_3 = clf_3.predict(X_test)

# 5.画图
plt.figure(figsize=(16,9),dpi=80,facecolor='w')
plt.scatter(X, y, c="k", s=10,label="data")
plt.plot(X_test, y_0, c="y", label="max_depth=1,$R^2$=%.3f" % (clf_0.score(X, y)), linewidth=2)
plt.plot(X_test, y_1, c="g", label="max_depth=2,$R^2$=%.3f" % (clf_1.score(X, y)), linewidth=2)
plt.plot(X_test, y_2, c="r", label="max_depth=3,$R^2$=%.3f" % (clf_2.score(X, y)), linewidth=2)
plt.plot(X_test, y_3, c="b", label="max_depth=5,$R^2$=%.3f" % (clf_3.score(X, y)), linewidth=2)
plt.xlabel("X", horizontalalignment="left")
plt.ylabel("Y")
plt.title("Decision Tree Regression")
plt.legend(loc = 'lower right')
plt.show()