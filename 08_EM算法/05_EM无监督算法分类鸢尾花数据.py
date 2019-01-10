import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances_argmin

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def expand(a, b, rate=0.05):
    d = (b - a) * rate
    return a-d, b+d


## 数据加载
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
path = '../datas/iris.data'
data = pd.read_csv(path, header=None)
x_prime = data[np.arange(4)]
y = pd.Categorical(data[4]).codes


# 类别数量
n_components = 3
# 不同特征属性分类
feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
plt.figure(figsize=(10, 6), facecolor='w')
cm_light = mpl.colors.ListedColormap(['#FFA0A0', '#A0FFA0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['r', 'g', 'b'])

for k, pair in enumerate(feature_pairs):
    print (u"特征属性:", iris_feature[pair[0]], iris_feature[pair[1]])
    x = x_prime[pair]
    # 计算实际的均值
    mean = np.array([np.mean(x[y == i], axis=0) for i in range(3)])
    print (u"实际均值：\n",mean)
    
    # 模型构建
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=28)
    gmm.fit(x)
    
    # 效果参数输出
    print ("预测均值=\n", gmm.means_)
    print ("预测方差=\n", gmm.covariances_)
    
    # 获取预测结果
    y_hat = gmm.predict(x)
    order = pairwise_distances_argmin(mean, gmm.means_, axis=1, metric='euclidean')
    print ('顺序：\t', order)
    
    # 修改预测结果顺序
    n_sample = y.size
    n_types = 3
    change = np.empty((n_types, n_sample), dtype=np.bool)
    for i in range(n_types):
        change[i] = y_hat == order[i]
    for i in range(n_types):
        y_hat[change[i]] = i
    
    # 计算准确率
    acc = u'准确率：%.2f%%' % (100*np.mean(y_hat == y))
    print (acc)
    
    # 画图
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    grid_hat = gmm.predict(grid_test)

    change = np.empty((n_types, grid_hat.size), dtype=np.bool)
    for i in range(n_types):
        change[i] = grid_hat == order[i]
    for i in range(n_types):
        grid_hat[change[i]] = i
    grid_hat = grid_hat.reshape(x1.shape)
    
    # 子图
    plt.subplot(3, 2, k+1)
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[pair[0]], x[pair[1]], s=30, c=y, marker='o', cmap=cm_dark, edgecolors='k')
    xx = 0.9 * x1_min + 0.1 * x1_max
    yy = 0.2 * x2_min + 0.8 * x2_max
    plt.text(xx, yy, acc, fontsize=14)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.xlabel(iris_feature[pair[0]], fontsize=14)
    plt.ylabel(iris_feature[pair[1]], fontsize=14)
    plt.grid()
    print()
    print()
    
plt.tight_layout(2)
plt.suptitle(u'EM算法鸢尾花数据分类', fontsize=20)
plt.subplots_adjust(top=0.90)
plt.show()