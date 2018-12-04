# -*- coding:utf-8 -*-

import numpy as np

from sklearn.datasets import make_blobs  # 产生模拟数据的方法
from sklearn.cluster import KMeans  # K-means模型


# 1、产生模拟数据
N = 1000
centers = 4
"""
n_samples：样本数
n_features：特征数，维度
centers：簇个数
random_state：随机数种子
"""
X, Y = make_blobs(n_samples=N, n_features=2, centers=centers, random_state=0)


# 2、模型构建及训练
km = KMeans(n_clusters=centers, init="random", random_state=0)
km.fit(X)


# 3、模型预测
y_hat = km.predict(X[:10])
print(y_hat)


# 4、打印一些信息
print("所有样本距离所属簇中心点的总距离和为:%.5f" % km.inertia_)
print("所有样本距离所属簇中心点的平均距离为:%.5f" % (km.inertia_ / N))

print("所有的中心点聚类中心坐标:")
cluter_centers = km.cluster_centers_
print(cluter_centers)

print("score其实就是所有样本点离所属簇中心点距离和的相反数:")
print(km.score(X))