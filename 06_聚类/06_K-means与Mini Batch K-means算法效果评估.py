import time
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib as mpl

from sklearn.cluster import MiniBatchKMeans, KMeans  # 模型
from sklearn import metrics  # 评估
from sklearn.metrics.pairwise import pairwise_distances_argmin  # 距离计算
from sklearn.datasets.samples_generator import make_blobs  # 模拟数据创建

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 1.创建模拟数据
# 初始化三个中心
centers = [[1, 1], [-1, -1], [1, -1]] 
clusters = len(centers)  # 聚类的数目为3    
# 产生3000组二维的数据，中心是意思三个中心点，标准差是0.7
X, Y = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7, random_state=28)
# Y在实际工作中是人工给定的，专门用于判断聚类的效果的一个值
# 实际工作中，我们假定聚类算法的模型都是比较可以，最多用轮廓系数/模型的score api返回值进行度量；
# 其它的效果度量方式一般不用
# 原因：其它度量方式需要给定数据的实际的Y值 ===> 当我给定Y值的时候，其实我可以直接使用分类算法了，不需要使用聚类


# 2.模型构建，训练
# a.KMeans模型
k_means = KMeans(init='k-means++', n_clusters=clusters, random_state=28)
t0 = time.time() 
k_means.fit(X)  
km_batch = time.time() - t0  
print ("K-Means算法模型训练消耗时间:%.4fs" % km_batch)
# b.MiniBatchKMeans模型
batch_size = 100
mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=batch_size, random_state=28)  
t0 = time.time()  
mbk.fit(X)  
mbk_batch = time.time() - t0  
print ("Mini Batch K-Means算法模型训练消耗时间:%.4fs" % mbk_batch)


# 3.样本所属的类别
km_y_hat = k_means.labels_
mbkm_y_hat = mbk.labels_


# 4.获取聚类中心点并聚类中心点进行排序
k_means_cluster_centers = k_means.cluster_centers_  # 输出kmeans聚类中心点
mbk_means_cluster_centers = mbk.cluster_centers_  # 输出mbk聚类中心点
print ("K-Means算法聚类中心点:\ncenter=", k_means_cluster_centers)
print ("Mini Batch K-Means算法聚类中心点:\ncenter=", mbk_means_cluster_centers)
order = pairwise_distances_argmin(k_means_cluster_centers, mbk_means_cluster_centers)


# 5.效果评估
score_funcs = [
    metrics.adjusted_rand_score,  # ARI
    metrics.v_measure_score,  # 均一性和完整性的加权平均
    metrics.adjusted_mutual_info_score,  # AMI
    metrics.mutual_info_score,  # 互信息
]

# 迭代对每个评估函数进行评估操作
for score_func in score_funcs:
    t0 = time.time()
    km_scores = score_func(Y,km_y_hat)
    print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (score_func.__name__,km_scores, time.time() - t0))
    
    t0 = time.time()
    mbkm_scores = score_func(Y,mbkm_y_hat)
    print("Mini Batch K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs\n" % (score_func.__name__,mbkm_scores, time.time() - t0))