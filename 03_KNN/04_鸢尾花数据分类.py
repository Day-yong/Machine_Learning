# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.neighbors import KNeighborsClassifier  # KNN模型
from sklearn.preprocessing import label_binarize  # 转换数据为矩阵
from sklearn import metrics  # 模型评估，用于计算roc/auc

"""
数据格式如下：
5.1, 3.5, 1.4, 0.2, Iris-setosa
7.0, 3.2, 4.7, 1.4, Iris-versicolor
7.3, 2.9, 6.3, 1.8, Iris-virginica
数据有三类
可以通过df['cla'].value_counts()查看
"""

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


# 数据加载
path = '../datas/iris.data'
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df = pd.read_csv(path, header=None, names=names)


# 2.数据预处理
# 需要将最后一个特征进行哑编码
# a.自定义哑编码
def parseRecord(record):
    result=[]
    r = zip(names, record)
    for name,v in r:
        if name == 'cla':
            if v == 'Iris-setosa':
                result.append(1)
            elif v == 'Iris-versicolor':
                result.append(2)
            elif v == 'Iris-virginica':
                result.append(3)
            else:
                result.append(np.nan)
        else:
            result.append(float(v))
    return result

# b.数据转换
datas = df.apply(lambda r: pd.Series(parseRecord(r),index=names), axis=1)

# c.异常数据处理
datas = datas.dropna(how='any')

# d.数据获取
X = datas[names[0:-1]]
Y = datas[names[-1]]


# 3.数据划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


# 4.KNN模型构建
knn = KNeighborsClassifier(n_neighbors = 3)  # k为3，algorithm='auto'默认，也可指定为{'auto', 'ball_tree', 'kd_tree', 'brute'}
knn.fit(X_train, Y_train)  # 训练模型


# 5.模型评估
y_test_hot = label_binarize(Y_test,classes=(1,2,3))  # 数据转换为矩阵形式
knn_y_score = knn.predict_proba(X_test) # 得到预测属于某个类别的概率值
# 计算ROC
knn_fpr, knn_tpr, knn_threasholds = metrics.roc_curve(y_test_hot.ravel(), knn_y_score.ravel())  # ravel返回一个扁平数组
# 计算AUC
knn_auc = metrics.auc(knn_fpr, knn_tpr)
print ("KNN算法R值：", knn.score(X_train, Y_train))
print ("KNN算法AUC值：", knn_auc)

# 6.模型预测
knn_y_predict = knn.predict(X_test)

# 7.预测结果画图
x_test_len = range(len(X_test))
plt.figure(figsize=(12, 9), facecolor='w')
plt.ylim(0.5,3.5)
plt.plot(x_test_len, Y_test, 'ro', markersize = 6, zorder=3, label=u'真实值')  # 画真实值
plt.plot(x_test_len, knn_y_predict, 'yo', markersize = 16, zorder=1, label=u'KNN算法预测值,$R^2$=%.3f' % knn.score(X_test, Y_test))
plt.legend(loc = 'lower right')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'种类', fontsize=18)
plt.title(u'KNN算法对鸢尾花数据分类', fontsize=20)
plt.show()

# 8.画图：ROC曲线画图
plt.figure(figsize=(8, 6), facecolor='w')
plt.plot(knn_fpr,knn_tpr, c='g', lw=2, label=u'KNN算法,AUC=%.3f' % knn_auc)
plt.plot((0,1),(0,1), c='#a0a0a0', lw=2, ls='--')
plt.xlim(-0.01, 1.02)
plt.ylim(-0.01, 1.02)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate(FPR)', fontsize=16)
plt.ylabel('True Positive Rate(TPR)', fontsize=16)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'鸢尾花数据KNN算法的ROC/AUC', fontsize=18)
plt.show()