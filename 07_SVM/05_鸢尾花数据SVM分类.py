import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from sklearn import svm  # SVM模型
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


# 4.模型构建
'''
sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, 
				shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, 
				verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)
svm.SVC API说明：
# 功能：使用SVM分类器进行模型构建
# 参数说明：
# C: 误差项的惩罚系数，默认为1.0；一般为大于0的一个数字，C越大表示在训练过程中对于总误差的关注度越高，
	 也就是说当C越大的时候，对于训练集的表现会越好，但是有可能引发过度拟合的问题(overfiting)
# kernel：指定SVM内部函数的类型，可选值：linear、poly、rbf、sigmoid、precomputed(基本不用，有前提要求，要求特征属性数目和样本数目一样)；默认是rbf；
# degree：当使用多项式函数作为svm内部的函数的时候，给定多项式的项数，默认为3
# gamma：当SVM内部使用poly、rbf、sigmoid的时候，核函数的系数值，当默认值为auto的时候，实际系数为1/n_features
# coef0: 当核函数为poly或者sigmoid的时候，给定的独立系数，默认为0
# probability：是否启用概率估计，默认不启动，不太建议启动
# shrinking：是否开启收缩启发式计算，默认为True
# tol: 模型构建收敛参数，当模型的的误差变化率小于该值的时候，结束模型构建过程，默认值:1e-3
# cache_size：在模型构建过程中，缓存数据的最大内存大小，默认为空，单位MB
# class_weight：给定各个类别的权重，默认为空
# max_iter：最大迭代次数，默认-1表示不限制
# decision_function_shape: 决策函数，可选值：ovo和ovr，默认为None；推荐使用ovr；
'''
clf = svm.SVC(C=1, kernel='rbf', gamma=0.1)
# gamma值越大，训练集的拟合就越好，但是会造成过拟合，导致测试集拟合变差
# gamma值越小，模型的泛化能力越好，训练集和测试集的拟合相近，但是会导致训练集出现欠拟合问题，
# 从而，准确率变低，导致测试集准确率也变低。


# 5.模型训练
clf.fit(X_train, Y_train)


# 6.模型评估：计算模型的准确率/精度
print (clf.score(X_train, Y_train)) 
print ('训练集准确率：', accuracy_score(Y_train, clf.predict(X_train)))
print (clf.score(X_test, Y_test))
print ('测试集准确率：', accuracy_score(Y_test, clf.predict(X_test)))
# 计算决策函数的结构值以及预测值(decision_function计算的是样本x到各个分割平面的距离<也就是决策函数的值>)
print ('decision_function:\n', clf.decision_function(X_train))
print ('\npredict:\n', clf.predict(X_train))


# 7.画图
N = 500
x1_min, x2_min = X.min()
x1_max, x2_max = X.max()

t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
grid_show = np.dstack((x1.flat, x2.flat))[0] # 测试点


grid_hat = clf.predict(grid_show)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

cm_light = mpl.colors.ListedColormap(['#00FFCC', '#FFA0A0', '#A0A0FF'])
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
plt.figure(facecolor='w')
# 区域图
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
# 所有样本点
plt.scatter(X[0], X[1], c=Y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
# 测试数据集
plt.scatter(X_test[0], X_test[1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
# lable列表
plt.xlabel(iris_feature[0], fontsize=13)
plt.ylabel(iris_feature[1], fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'鸢尾花SVM特征分类', fontsize=16)
plt.grid(b=True, ls=':')
plt.tight_layout(pad=1.5)
plt.show()
