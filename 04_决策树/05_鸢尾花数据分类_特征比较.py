# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier  # 决策树模型

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 1.读取数据
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'  # 特征名称
path = '../datas/iris.data'  # 数据路径
data = pd.read_csv(path, header=None)


# 2.划分数据
X_prime = data[list(range(4))]  # 特征数据
Y = pd.Categorical(data[4]).codes  # 把Y转换成分类型的0,1,2


# 3.特征比较
feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]  # 特征组合
plt.figure(figsize=(9, 6), facecolor='w') # 大小为(9, 6)的白色画板
for i, pair in enumerate(feature_pairs):  # 遍历feature_pairs，并返回索引以及对应的元素值，如 索引：0 ，值：[0, 1]
	# 截取两个特征，重新构成X特征数据
	X = X_prime[pair]
	# 决策树模型构建，训练
	model = DecisionTreeClassifier(criterion='gini', max_depth = 5)
	model.fit(X, Y)
	# 模型预测
	y_hat = model.predict(X)
	# 模型评估
	score = model.score(X, Y)
	Y2 = Y.reshape(-1)
	result = np.count_nonzero(y_hat == Y)  # 统计预测正确的个数
	print('特征：',iris_feature[pair[0]], '+' , iris_feature[pair[1]])
	print('预测正确数目：', result)
	print('准确率：%.2f%%' % (score * 100) )

	N = 500  # 横纵各采样多少值
	x1_min, x2_min = X.min()
	x1_max, x2_max = X.max()
	t1 = np.linspace(x1_min, x1_max, N)
	t2 = np.linspace(x2_min, x2_max, N)
	x1, x2 = np.meshgrid(t1, t2)  # 生成网络采样点
	x_test = np.dstack((x1.flat, x2.flat))[0]  # 测试点

	plt_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
	plt_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])  # 数据点颜色
	y_hat = model.predict(x_test)  # 预测
	y_hat = y_hat.reshape(x1.shape)

	plt.subplot(2, 3, i+1)  # 子图，2行，每行3个
	plt.pcolormesh(x1, x2, y_hat, cmap=plt_light)  # 测试数据
	plt.scatter(X[pair[0]], X[pair[1]], c=Y, edgecolors='k', cmap=plt_dark)  # 训练数据
	plt.xlabel(iris_feature[pair[0]], fontsize=10)
	plt.ylabel(iris_feature[pair[1]], fontsize=10)
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	plt.grid()
	plt.title(u'准确率:%.2f%%' % (score * 100), fontdict={'fontsize':15})

plt.suptitle(u'鸢尾花数据在决策树中两两特征属性对目标属性的影响', fontsize=18, y = 1)
plt.tight_layout(2)
plt.subplots_adjust(top=0.92)
plt.show()