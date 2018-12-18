import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from sklearn.svm import SVC  # svm.SVC模型
from sklearn.linear_model import LogisticRegression,RidgeClassifier  # 逻辑回归，岭回归模型
from sklearn.neighbors import KNeighborsClassifier  # KNN模型
from sklearn.model_selection import train_test_split  # 数据分割
from sklearn.metrics import accuracy_score  # 计算正确率
from sklearn.exceptions import ChangedBehaviorWarning  # 警告处理


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 警告处理
warnings.filterwarnings('ignore', category=ChangedBehaviorWarning)


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
# 4.1：SVM分类器
svm = SVC(C=1, kernel='linear')
svm.fit(X_train, Y_train)
# 4.2：LogisticRegression逻辑回归
lr = LogisticRegression()
lr.fit(X_train, Y_train)
# 4.3：RidgeClassifier岭回归
rc = RidgeClassifier()
rc.fit(X_train, Y_train)
# 4.4：KNN
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)


# 5.模型评估
svm_score1 = accuracy_score(Y_train, svm.predict(X_train))
svm_score2 = accuracy_score(Y_test, svm.predict(X_test))

lr_score1 = accuracy_score(Y_train, lr.predict(X_train))
lr_score2 = accuracy_score(Y_test, lr.predict(X_test))

rc_score1 = accuracy_score(Y_train, rc.predict(X_train))
rc_score2 = accuracy_score(Y_test, rc.predict(X_test))

knn_score1 = accuracy_score(Y_train, knn.predict(X_train))
knn_score2 = accuracy_score(Y_test, knn.predict(X_test))

# 6.画图1：准确率比较
x_tmp = [0,1,2,3]
y_score1 = [svm_score1, lr_score1, rc_score1, knn_score1]
y_score2 = [svm_score2, lr_score2, rc_score2, knn_score2]

plt.figure(facecolor='w')
plt.plot(x_tmp, y_score1, 'r-', lw=2, label=u'训练集准确率')
plt.plot(x_tmp, y_score2, 'g-', lw=2, label=u'测试集准确率')
plt.xlim(0, 3)
plt.ylim(np.min((np.min(y_score1), np.min(y_score2)))*0.9, np.max((np.max(y_score1), np.max(y_score2)))*1.1)
plt.legend(loc = 'lower right')
plt.title(u'鸢尾花数据不同分类器准确率比较', fontsize=16)
plt.xticks(x_tmp, [u'SVM', u'Logistic', u'Ridge', u'KNN'], rotation=0)
plt.grid(b=True)
plt.show()


# 7.画图2：分类结果比较
N = 500
x1_min, x2_min = X.min()
x1_max, x2_max = X.max()

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
grid_show = np.dstack((x1.flat, x2.flat))[0] # 测试点

# 获取各个不同算法的测试值
svm_grid_hat = svm.predict(grid_show)
svm_grid_hat = svm_grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

lr_grid_hat = lr.predict(grid_show)
lr_grid_hat = lr_grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

rc_grid_hat = rc.predict(grid_show)
rc_grid_hat = rc_grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

knn_grid_hat = knn.predict(grid_show)
knn_grid_hat = knn_grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

# 画图
cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w', figsize=(14,7))

# 子图1：SVM
plt.subplot(221)
## 区域图
plt.pcolormesh(x1, x2, svm_grid_hat, cmap=cm_light)
## 所有样本点
plt.scatter(X[0], X[1], c=Y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
## 测试数据集
plt.scatter(X_test[0], X_test[1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

# 子图2：LogisticRegression
plt.subplot(222)
## 区域图
plt.pcolormesh(x1, x2, lr_grid_hat, cmap=cm_light)
## 所有样本点
plt.scatter(X[0], X[1], c=Y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
## 测试数据集
plt.scatter(X_test[0], X_test[1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花Logistic特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

# 子图3：RidgeClassifier
plt.subplot(223)
## 区域图
plt.pcolormesh(x1, x2, rc_grid_hat, cmap=cm_light)
## 所有样本点
plt.scatter(X[0], X[1], c=Y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
## 测试数据集
plt.scatter(X_test[0], X_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花Ridge特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

# 子图4：KNeighborsClassifier
plt.subplot(224)
## 区域图
plt.pcolormesh(x1, x2, knn_grid_hat, cmap=cm_light)
## 所有样本点
plt.scatter(X[0], X[1], c=Y, edgecolors='k', s=50, cmap=cm_dark)      # 样本
## 测试数据集
plt.scatter(X_test[0], X_test[1], s=120, facecolors='none', zorder=10)     # 圈中测试集样本
## lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花KNN特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)

plt.show()
