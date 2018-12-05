import time  
import numpy as np  
import matplotlib.pyplot as plt  
import matplotlib as mpl

from sklearn.cluster import MiniBatchKMeans, KMeans  # 模型
from sklearn.metrics.pairwise import pairwise_distances_argmin  # 计算距离
from sklearn.datasets.samples_generator import make_blobs  # 模拟数据


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 1.创建模拟数据
# 初始化三个中心
centers = [[1, 1], [-1, -1], [1, -1]] 
clusters = len(centers)  # 聚类的数目为3    
# 产生3000组二维的数据，中心是意思三个中心点，标准差是0.7
X, Y = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7, random_state=28)


# 2.模型构建
# a.构建kmeans算法
k_means = KMeans(init='k-means++', n_clusters=clusters, random_state=28)
t0 = time.time()  # 当前时间
k_means.fit(X)  # 训练模型
km_batch = time.time() - t0  # 使用kmeans训练数据的消耗时间
print ("K-Means算法模型训练消耗时间:%.4fs" % km_batch)

# b.构建MiniBatchKMeans算法
batch_size = 100
mbk = MiniBatchKMeans(init='k-means++', n_clusters=clusters, batch_size=batch_size, random_state=28)  
t0 = time.time()  
mbk.fit(X)  
mbk_batch = time.time() - t0  
print ("Mini Batch K-Means算法模型训练消耗时间:%.4fs" % mbk_batch)


# 3.模型预测
km_y_hat = k_means.predict(X)
mbkm_y_hat = mbk.predict(X)


# 4.获取聚类中心点并聚类中心点进行排序，方便后面画图
k_means_cluster_centers = k_means.cluster_centers_  # 输出kmeans聚类中心点
mbk_means_cluster_centers = mbk.cluster_centers_  # 输出mbk聚类中心点
print ("K-Means算法聚类中心点:\ncenter=", k_means_cluster_centers)
print ("Mini Batch K-Means算法聚类中心点:\ncenter=", mbk_means_cluster_centers)
order = pairwise_distances_argmin(k_means_cluster_centers, mbk_means_cluster_centers)


# 5.画图
plt.figure(figsize=(12, 6), facecolor='w')  # 画板
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)  # 调整子图布局
cm = mpl.colors.ListedColormap(['#FFC2CC', '#C2FFCC', '#CCC2FF'])  # 数据颜色列表
cm2 = mpl.colors.ListedColormap(['#FF0000', '#00FF00', '#0000FF'])  # 质心颜色列表
# 子图1：原始数据
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=Y, s=6, cmap=cm, edgecolors='none')
plt.title(u'原始数据分布图')
plt.xticks(())
plt.yticks(())
plt.grid(True)
# 子图2：K-Means算法聚类结果图
plt.subplot(222)
plt.scatter(X[:,0], X[:,1], c=km_y_hat, s=6, cmap=cm,edgecolors='none')
plt.scatter(k_means_cluster_centers[:,0], k_means_cluster_centers[:,1],c=range(clusters),s=60,cmap=cm2,edgecolors='none')
plt.title(u'K-Means算法聚类结果图')
plt.xticks(())
plt.yticks(())
plt.text(-3.5, 2.5,  '训练时间为: %.2fms' % (km_batch*1000))  
plt.grid(True)
# 子图3：Mini Batch K-Means算法聚类结果图
plt.subplot(223)
plt.scatter(X[:,0], X[:,1], c=mbkm_y_hat, s=6, cmap=cm,edgecolors='none')
plt.scatter(mbk_means_cluster_centers[:,0], mbk_means_cluster_centers[:,1],c=range(clusters),s=60,cmap=cm2,edgecolors='none')
plt.title(u'Mini Batch K-Means算法聚类结果图')
plt.xticks(())
plt.yticks(())
plt.text(-3.5, 2.5,  '训练时间为: %.2fms' % (mbk_batch*1000))  
plt.grid(True)


different = list(map(lambda x: (x!=0) & (x!=1) & (x!=2), mbkm_y_hat))  # 先标记mbkm_y_hat与km_y_hat不相同，即先标记为False
for k in range(clusters):  
	# 不相等就加True,相等加False,False+True=1,False+False=0
    different += ((km_y_hat == k) != (mbkm_y_hat == order[k]))  
identic = np.logical_not(different)  # 反转，False变为True,True变为False
different_nodes = len(list(filter(lambda x:x, different)))  # 计算不相同点的个数

plt.subplot(224)
# 两者预测相同的
plt.plot(X[identic, 0], X[identic, 1], 'w', markerfacecolor='#6495ED', marker='.')  
# 两者预测不相同的
plt.plot(X[different, 0], X[different, 1], 'w', markerfacecolor='r', marker='.')  
plt.title(u'Mini Batch K-Means和K-Means算法预测结果不同的点')  
plt.xticks(())  
plt.yticks(())
plt.text(-3.5, 2.5,  '不同的点数目为: %d' % (different_nodes))  

plt.show()