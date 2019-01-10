# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import multivariate_normal  # 多元正态分布
from sklearn.mixture import GaussianMixture  # GMM Gaussian Mixture Model
from sklearn.metrics.pairwise import pairwise_distances_argmin  # 计算一个点和一组点之间的最小距离


# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 使用scikit携带的EM算法或者自己实现的EM算法
def trainModel(style, x):
    if style == 'sklearn':
        print("sklearn")
        # 对象创建
        g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000, init_params='kmeans')
        # 模型训练
        g.fit(x)
        # 效果输出
        print('类别概率:\t', g.weights_)
        print('均值:\n', g.means_, '\n')
        print('方差:\n', g.covariances_, '\n')
        print('似然函数的值:\n', g.lower_bound_)
        mu1, mu2 = g.means_
        sigma1, sigma2 = g.covariances_
        # 返回数据
        return (mu1, mu2, sigma1, sigma2)
    else:
        # 自己实现一个EM算法
        num_iter = 100  # 迭代次数
        n, d = data.shape  # 数据维度
        
        # 初始化均值和方差正定矩阵(sigma叫做协方差矩阵)
        mu1 = data.min(axis=0)
        mu2 = data.max(axis=0)
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        gamma = 0.5
        print("随机初始的期望为:")
        print(mu1)
        print(mu2)
        print("随机初始的方差为:")
        print(sigma1)
        print(sigma2)
        print("随机初始的γ为:")
        print([gamma, 1-gamma])
        
        # 实现EM算法
        for i in range(num_iter):
            # E Step
            # 1. 计算获得多元高斯分布的概率密度函数
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            # 2. 计算概率值
            tau1 = gamma * norm1.pdf(data)
            tau2 = (1 - gamma) * norm2.pdf(data)
            # 3. 概率值均一化（即E步：公式中的γ帽子）
            hat_gamma = tau1 / (tau1 + tau2)
            
            # M Step
            # 1. 计算更新后的均值
            mu1 = np.dot(hat_gamma, data) / np.sum(hat_gamma)
            mu2 = np.dot((1 - hat_gamma), data) / np.sum((1 - hat_gamma))
            # 2. 计算更新后的方差
            sigma1 = np.dot(hat_gamma * (data - mu1).T, data - mu1) / np.sum(hat_gamma)
            sigma2 = np.dot((1 - hat_gamma) * (data - mu2).T, data - mu2) / np.sum(1 - hat_gamma)
            # 3. 计算更新后的π值
            gamma = np.sum(hat_gamma) / n
            
            # 输出信息
            j = i + 1
            if j % 10 == 0:
                print (j, ":\t", mu1, mu2)
        
        # 效果输出
        print ('类别概率:\t', gamma)
        print ('均值:\t', mu1, mu2)
        print ('方差:\n', sigma1, '\n\n', sigma2, '\n')
        
        # 返回结果
        return (mu1, mu2, sigma1, sigma2)


# 创建模拟数据（3维数据）
np.random.seed(1)  # 随机数种子
N = 500
M = 250

# 根据给定的均值和协方差矩阵构建数据
mean1 = (0, 0, 0)
cov1 = np.diag((1, 2, 3))
## 产生500条数据
data1 = np.random.multivariate_normal(mean1, cov1, N)

## 产生一个数据分布不均衡的数据集， 250条
mean2 = (2, 2, 1)
cov2 = np.array(((3, 1, 0), (1, 3, 0), (0, 0, 3)))
data2 = np.random.multivariate_normal(mean2, cov2, M)

## 合并data1和data2这两个数据集
data = np.vstack((data1, data2))

## 产生数据对应的y值
y1 = np.array([True] * N + [False] * M)
y2 = ~y1


# 预测结果(得到概率密度值)
#style = 'sklearn'
style = 'self'
mu1, mu2, sigma1, sigma2 = trainModel(style, data)
# 预测分类（根据均值和方差对原始数据进行概率密度的推测）
norm1 = multivariate_normal(mu1, sigma1)
norm2 = multivariate_normal(mu2, sigma2)
tau1 = norm1.pdf(data)
tau2 = norm2.pdf(data)


## 计算均值的距离，然后根据距离得到分类情况
dist = pairwise_distances_argmin([mean1, mean2], [mu1, mu2], metric='euclidean')
print ("距离:", dist)
if dist[0] == 0:
    c1 = tau1 > tau2
else:
    c1 = tau1 < tau2
c2 = ~c1
## 计算准确率
acc = np.mean(y1 == c1)
print (u'准确率：%.2f%%' % (100*acc))



# 画图
fig = plt.figure(figsize=(12, 6), facecolor='w')

## 添加一个子图，设置为3d的
ax = fig.add_subplot(1, 2, 1, projection='3d')
## 点图
ax.scatter(data[y1, 0], data[y1, 1], data[y1, 2], c='r', s=30, marker='o', depthshade=True)
ax.scatter(data[y2, 0], data[y2, 1], data[y2, 2], c='g', s=30, marker='^', depthshade=True)
## 标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
## 标题
ax.set_title(u'原始数据', fontsize=16)

## 添加一个子图，设置为3d
ax = fig.add_subplot(1, 2, 2, projection='3d')
# 画点
ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
# 设置标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# 设置标题
ax.set_title(u'EM算法分类', fontsize=16)

# 设置总标题
plt.suptitle(u'EM算法的实现,准备率：%.2f%%' % (acc * 100), fontsize=20)
plt.subplots_adjust(top=0.9)
plt.show()