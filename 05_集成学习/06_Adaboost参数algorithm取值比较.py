# -*- coding: utf-8 -*- 


import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.datasets import make_gaussian_quantiles  # 数据构造
from sklearn.ensemble import AdaBoostClassifier  # 分类AdaBoost
from sklearn.metrics import accuracy_score  # 计算roc和auc
from sklearn.tree import DecisionTreeClassifier  # 分类决策树
from sklearn.externals.six.moves import zip

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 1.创建模拟数据
X, Y = make_gaussian_quantiles(n_samples=13000, n_features=10, n_classes=3, random_state=1)


# 2.划分数据集
n_split = 3000
X_train, X_test = X[:n_split], X[n_split:]
Y_train, Y_test = Y[:n_split], Y[n_split:]


# 3.建立两个模型，algorithm算法不同，bdt_real选择的是samme.r
bdt_real = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1)
bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=600, learning_rate=1, algorithm="SAMME")


# 4.模型训练
bdt_real.fit(X_train, Y_train)
bdt_discrete.fit(X_train, Y_train)


# 5.模型预测，评估
# staged_predict()：输入样本的预测类别被计算为集合中分类器的加权平均预测
# 获得预测的准确率，accuracy_score，是单个分类器的准确率。
# 预测的误差率estimator_errors_
real_test_errors = []  # 第一个模型每一个分类器的误差率
discrete_test_errors = []  # 第二个模型每一个分类器的误差率

for real_test_predict, discrete_train_predict in zip(bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
    real_test_errors.append(1. - accuracy_score(real_test_predict, Y_test))  # 1-准确率=误差率
    discrete_test_errors.append(1. - accuracy_score(discrete_train_predict, Y_test))

n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)

discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]


# 6.画图
plt.figure(figsize=(15, 5), facecolor='w')

plt.subplot(131)  # 子图，1行3列第1列
plt.plot(range(1, n_trees_discrete + 1), discrete_test_errors, c='g', label='SAMME')
plt.plot(range(1, n_trees_real + 1), real_test_errors, c='r', linestyle='dashed', label='SAMME.R')
plt.legend()
plt.ylim(0.18, 0.62)
plt.ylabel(u'测试数据的预测错误率')
plt.xlabel(u'弱分类器数目')

plt.subplot(132)  # 子图，1行3列第2列
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors, "b", label='SAMME', alpha=.5)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,"r", label='SAMME.R', alpha=.5)
plt.legend()
plt.ylabel(u'模型实际错误率')
plt.xlabel(u'弱分类器数目')
plt.ylim((.2, max(real_estimator_errors.max(), discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))

plt.subplot(133)  # 子图，1行3列第3列
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights, "b", label='SAMME')
plt.legend()
plt.ylabel(u'权重')
plt.xlabel(u'弱分类器编号')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))

# 显示
plt.subplots_adjust(wspace=0.25)
plt.show()