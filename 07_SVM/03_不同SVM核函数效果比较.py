import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

from sklearn.svm import SVC  # svm.SVC模型
from sklearn.model_selection import train_test_split  # 数据分割
from sklearn.metrics import accuracy_score  # 计算正确率


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


# 1.读取数据
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'
path = '../datas/iris.data'  # 数据路径
data = pd.read_csv(path, header=None)


# 2.划分数据
X, Y = data[list(range(4))], data[4]
# 把文本数据进行编码，比如a b c编码为 0 1 2; 可以通过pd.Categorical(y).categories获取index对应的原始值
Y = pd.Categorical(Y).codes
X = data[[0,1]]  # 获取第1列和第二列


# 3.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=0)


# 4.模型构建及训练
## 数据SVM分类器构建
svm1 = SVC(C=1, kernel='linear')
svm2 = SVC(C=1, kernel='rbf')
svm3 = SVC(C=1, kernel='poly')
svm4 = SVC(C=1, kernel='sigmoid')

## 模型训练
t0=time.time()
svm1.fit(X_train, Y_train)
t1=time.time()
svm2.fit(X_train, Y_train)
t2=time.time()
svm3.fit(X_train, Y_train)
t3=time.time()
svm4.fit(X_train, Y_train)
t4=time.time()


# 5.模型评估
svm1_score1 = accuracy_score(Y_train, svm1.predict(X_train))
svm1_score2 = accuracy_score(Y_test, svm1.predict(X_test))

svm2_score1 = accuracy_score(Y_train, svm2.predict(X_train))
svm2_score2 = accuracy_score(Y_test, svm2.predict(X_test))

svm3_score1 = accuracy_score(Y_train, svm3.predict(X_train))
svm3_score2 = accuracy_score(Y_test, svm3.predict(X_test))

svm4_score1 = accuracy_score(Y_train, svm4.predict(X_train))
svm4_score2 = accuracy_score(Y_test, svm4.predict(X_test))

# 画图1：评估结果比较
x_tmp = [0,1,2,3]
t_score = [t1 - t0, t2-t1, t3-t2, t4-t3]
y_score1 = [svm1_score1, svm2_score1, svm3_score1, svm4_score1]
y_score2 = [svm1_score2, svm2_score2, svm3_score2, svm4_score2]
plt.figure(facecolor='w', figsize=(12,6))

plt.subplot(121)
plt.plot(x_tmp, y_score1, 'r-', lw=2, label=u'训练集准确率')
plt.plot(x_tmp, y_score2, 'g-', lw=2, label=u'测试集准确率')
plt.xlim(-0.3, 3.3)
plt.ylim(np.min((np.min(y_score1), np.min(y_score2)))*0.9, np.max((np.max(y_score1), np.max(y_score2)))*1.1)
plt.legend(loc = 'lower left')
plt.title(u'模型预测准确率', fontsize=13)
plt.xticks(x_tmp, [u'linear-SVM', u'rbf-SVM', u'poly-SVM', u'sigmoid-SVM'], rotation=0)
plt.grid(b=True)

plt.subplot(122)
plt.plot(x_tmp, t_score, 'b-', lw=2, label=u'模型训练时间')
plt.title(u'模型训练耗时', fontsize=13)
plt.xticks(x_tmp, [u'linear-SVM', u'rbf-SVM', u'poly-SVM', u'sigmoid-SVM'], rotation=0)
plt.xlim(-0.3, 3.3)
plt.grid(b=True)

plt.suptitle(u'鸢尾花数据SVM分类器不同内核函数模型比较', fontsize=16)
plt.show()


# 6.画图2：分类结果比较
N = 500
x1_min, x2_min = X.min()
x1_max, x2_max = X.max()

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
grid_show = np.dstack((x1.flat, x2.flat))[0] # 测试点

## 获取各个不同算法的测试值
svm1_grid_hat = svm1.predict(grid_show)
svm1_grid_hat = svm1_grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

svm2_grid_hat = svm2.predict(grid_show)
svm2_grid_hat = svm2_grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

svm3_grid_hat = svm3.predict(grid_show)
svm3_grid_hat = svm3_grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

svm4_grid_hat = svm4.predict(grid_show)
svm4_grid_hat = svm4_grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

## 画图
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w', figsize=(14,7))

# 子图1：核函数为linear
plt.subplot(221)
## 区域图
plt.pcolormesh(x1, x2, svm1_grid_hat, cmap=cm_light)
## 所以样本点
plt.scatter(X[0], X[1], c=Y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
## 测试数据集
plt.scatter(X_test[0], X_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花Linear-SVM特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

# 子图2：核函数为rbf
plt.subplot(222)
## 区域图
plt.pcolormesh(x1, x2, svm2_grid_hat, cmap=cm_light)
## 所以样本点
plt.scatter(X[0], X[1], c=Y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
## 测试数据集
plt.scatter(X_test[0], X_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花rbf-SVM特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

# 子图3：核函数为poly
plt.subplot(223)
## 区域图
plt.pcolormesh(x1, x2, svm3_grid_hat, cmap=cm_light)
## 所以样本点
plt.scatter(X[0], X[1], c=Y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
## 测试数据集
plt.scatter(X_test[0], X_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花poly-SVM特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

# 子图4：核函数为sigmoid
plt.subplot(224)
## 区域图
plt.pcolormesh(x1, x2, svm4_grid_hat, cmap=cm_light)
## 所以样本点
plt.scatter(X[0], X[1], c=Y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
## 测试数据集
plt.scatter(X_test[0], X_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花sigmoid-SVM特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.show()