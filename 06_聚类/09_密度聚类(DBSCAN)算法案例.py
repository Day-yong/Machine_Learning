import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as ds  # 创建数据
import matplotlib.colors

from sklearn.cluster import DBSCAN  # DBSCAN模型
from sklearn.preprocessing import StandardScaler  # 数据标准化


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 1.创建模拟数据
N = 1000  # 数据数目
centers = [[1, 2], [-1, -1], [1, -1], [-1, 1]]  # 质心坐标
# 球形数据
data1, y1 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1,0.75, 0.5,0.25), random_state=0)
data1 = StandardScaler().fit_transform(data1)  # 标准化，构建模型，训练模型并转换数据
# 参数可选值
params1 = ((0.15, 5), (0.2, 10), (0.2, 15), (0.3, 5), (0.3, 10), (0.3, 15))

# 三个环形数据
t = np.arange(0, 2 * np.pi, 0.1)  # 角度0-2π
data2_1 = np.vstack((np.cos(t), np.sin(t))).T
data2_2 = np.vstack((2*np.cos(t), 2*np.sin(t))).T
data2_3 = np.vstack((3*np.cos(t), 3*np.sin(t))).T
data2 = np.vstack((data2_1, data2_2, data2_3))
y2 = np.vstack(([0] * len(data2_1), [1] * len(data2_2), [2] * len(data2_3))).ravel()
# 参数可选值
params2 = ((0.5, 3), (0.5, 5), (0.5, 10), (1., 3), (1., 10), (1., 20))

datasets = [(data1, y1, params1), (data2, y2, params2)]



# 2.模型构建，训练，画图
# 扩展图边界
def expandBorder(a, b):
    d = (b - a) * 0.1
    return a-d, b+d

colors = ['r', 'g', 'b', 'y', 'c']  # 颜色列表
cm = mpl.colors.ListedColormap(colors)

for i,(X, y, params) in enumerate(datasets):  # 遍历两个数据
    x1_min, x2_min = np.min(X, axis=0)
    x1_max, x2_max = np.max(X, axis=0)
    x1_min, x1_max = expandBorder(x1_min, x1_max)
    x2_min, x2_max = expandBorder(x2_min, x2_max)
    
    plt.figure(figsize=(12, 8), facecolor='w')
    plt.suptitle(u'DBSCAN聚类-数据%d' % (i+1), fontsize=20)
    plt.subplots_adjust(top=0.9, hspace=0.5)  # 子图的间的距离
    
    for j, param in enumerate(params):
        eps, min_samples = param
        # 模型构建
        model = DBSCAN(eps=eps, min_samples=min_samples)
        # eps 半径，控制邻域的大小，值越大，越能容忍噪声点，值越小，相比形成的簇就越多
        # min_samples 原理中所说的M，控制哪个是核心点，值越小，越可以容忍噪声点，越大，就更容易把有效点划分成噪声点
        
        model.fit(X)  # 训练模型
        y_hat = model.labels_

        unique_y_hat = np.unique(y_hat)  # 类别
        n_clusters = len(unique_y_hat) - (1 if -1 in y_hat else 0)  # 簇数目
        print ("类别:",unique_y_hat,"；聚类簇数目:",n_clusters)
        
        core_samples_mask = np.zeros_like(y_hat, dtype=bool)  # 创建大小和y_hat相同的元素为False的数组
        core_samples_mask[model.core_sample_indices_] = True
        
        # 画图
        plt.subplot(3,3,j+1)
        for k, col in zip(unique_y_hat, colors):
            if k == -1:
                col = 'c'
                
            class_member_mask = (y_hat == k)  # 据聚类后所属簇索引和类别相等，值设为True
            xy = X[class_member_mask & core_samples_mask]  # 聚类后与之前的相同的(图中显示的圈大点)
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
            xy = X[class_member_mask & ~core_samples_mask] # 聚类后与之前的不相同的(图中显示的圈小点)
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.grid(True)
        plt.title('$\epsilon$ = %.1f  m = %d，聚类簇数目：%d' % (eps, min_samples, n_clusters), fontsize=14)
    # 原始数据显示
    plt.subplot(3,3,7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cm, edgecolors='none')
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.title('原始数据，聚类簇数目:%d' % len(np.unique(y)))
    plt.grid(True)
    plt.show()   