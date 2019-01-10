import numpy as np
import pandas as pd
import itertools
from scipy import linalg
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture  # 高斯混合模型


# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


# 用于适当扩大画图时的边界
def expand(a, b, rate=0.05):
    d = (b - a) * rate
    return a-d, b+d


# 1、样本数据产生
n_samples = 500

np.random.seed(28)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C), .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]


# 2、构建模型：不同参数效果比较
lowest_bic = np.infty
bics = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']

for cv_type in cv_types:  # 协方差的类型
    for n_components in n_components_range:  # 混合模型的个数
        gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        if bics[-1] < lowest_bic:
            lowest_bic = bics[-1]
            best_gmm = gmm


# 3、获取相关参数以及最优算法
clf = best_gmm
Y_ = clf.predict(X)

print ("均值:\n", clf.means_)
print ("方差:\n", clf.covariances_)



# 4、画图
bics = np.array(bics)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])

## 画出效果比较
spl = plt.subplot(2, 1, 1)
bars = []
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bics[i * len(n_components_range):(i + 1) * len(n_components_range)], width=.2, color=color))

plt.xticks(n_components_range)
plt.ylim([bics.min() * 1.01 - .01 * bics.max(), bics.max()])
plt.title(u'不同模型参数下BIC的值')
xpos = np.mod(bics.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bics.argmin() / len(n_components_range))
plt.text(xpos, bics.min() * 0.97 + .03 * bics.max(), '*', fontsize=14)
spl.set_xlabel(u'类别数量')
spl.legend([b[0] for b in bars], cv_types)

# 画出分类效果图（可以看到最优分类是2）
splot = plt.subplot(2, 1, 2)
cm_light = mpl.colors.ListedColormap(['#FFA0A0', '#A0FFA0'])
cm_dark = mpl.colors.ListedColormap(['r', 'g'])

x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
x1_min, x1_max = expand(x1_min, x1_max)
x2_min, x2_max = expand(x2_min, x2_max)
x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
grid_test = np.stack((x1.flat, x2.flat), axis=1)
grid_hat = clf.predict(grid_test)
grid_hat = grid_hat.reshape(x1.shape)
if clf.means_[0][0] > clf.means_[1][0]:
    z = grid_hat == 0
    grid_hat[z] = 1
    grid_hat[~z] = 0

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y_, marker='o', cmap=cm_dark, edgecolors='k')
plt.xlim((x1_min, x1_max))
plt.ylim((x2_min, x2_max))
plt.title(u'最优参数下GMM算法效果')
plt.subplots_adjust(hspace=.35, bottom=.02)
plt.grid()

plt.show()