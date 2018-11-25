# -*- coding: utf-8 -*-

"""
数据格式：
5.1, 3.5, 1.4, 0.2, Iris-setosa
7.0, 3.2, 4.7, 1.4, Iris-versicolor
6.5, 3.2, 5.1, 2.0, Iris-virginica
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings  # 警告处理

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier  # 分类树
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import MinMaxScaler  # 数据归一化
from sklearn.feature_selection import SelectKBest  # 特征选择
from sklearn.feature_selection import chi2  # 卡方统计量
from sklearn.decomposition import PCA  # 主成分分析


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)


# 1.读取数据
path = '../datas/iris.data'
df = pd.read_csv(path, header=None)


# 2.划分数据
X = df[list(range(4))]  # 获取X变量
Y = pd.Categorical(df[4]).codes 
# 获取Y，并将其转换为1,2,3类型(注意这里没有使用哑编码方式，而是直接使用了python自带的)
# 关于哑编码方式，可参看之前的鸢尾花数据分类的案例
print("总样本数目：%d;特征属性数目:%d" % X.shape)


# 3.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # random_state随机数生成种子
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (X_train.shape[0], X_test.shape[0]))

# DecisionTreeClassifier是分类算法，要求Y必须是int类型
Y_train = Y_train.astype(np.int)
Y_test = Y_test.astype(np.int)


# 4.数据预处理
## a.数据归一化
"""
数据标准化
	StandardScaler (基于特征矩阵的列，将属性值转换至服从正态分布)
	标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下
	常用与基于正态分布的算法，比如回归

数据归一化
	MinMaxScaler （区间缩放，基于最大最小值，将数据转换到0,1区间上的）
	(old - min) / (max - min)
	提升模型收敛速度，提升模型精度
	常见用于神经网络

Normalizer （基于矩阵的行，将样本向量转换为单位向量）
	其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准
	常见用于文本分类和聚类、logistic回归中也会使用，有效防止过拟合
"""
ss = MinMaxScaler()  # 创建归一化模型
X_train = ss.fit_transform(X_train)  # 根据给定的数据训练模型(fit)，然后使用训练好的模型对给定的数据进行转换操作(transform)
X_test = ss.transform(X_test)  # 使用训练好的模型对给定的数据集(X)进行转换操作
print ("原始数据各个特征属性的调整最小值(min):",ss.min_)
print ("原始数据各个特征属性的缩放数据值(max - min):",ss.scale_)


## b.特征选择：从已有的特征中选择出影响目标值最大的特征属性
"""
常用方法：{ 分类：F统计量、卡方系数(SelectKBest)，互信息mutual_info_classif
		 { 连续：皮尔逊相关系数 F统计量 互信息mutual_info_classif

在当前的案例中，使用SelectKBest这个方法从4个原始的特征属性，选择出来3个
k 默认为10，如果指定了，那么就会返回你所想要的特征的个数
"""

ch2 = SelectKBest(chi2, k=3)  # 创建卡方系数模型
X_train = ch2.fit_transform(X_train, Y_train)  # 训练模型，并转换
X_test = ch2.transform(X_test)  # 在训练好的模型下转换
select_name_index = ch2.get_support(indices=True)
print ("对类别判断影响最大的三个特征属性分布是:",ch2.get_support(indices=False))


## c.降维
"""
降维：
因为对于数据而言，如果特征属性比较多，在构建过程中，会比较复杂，这个时候考虑将多维（高维）映射到低维的数据
常用方法：
	PCA：主成分分析（无监督）
	LDA：线性判别分析（有监督）
"""
pca = PCA(n_components = 2) # 构建一个pca模型对象，设置最终维度是2维
# 这里是为了后面画图方便，所以将数据维度设置了2维，一般用默认不设置参数就可以
X_train = pca.fit_transform(X_train)  # 训练模型并转换
X_test = pca.transform(X_test)  # 转换


# 5.模型构建
model = DecisionTreeClassifier(criterion='entropy', random_state=0)  # 另外也可选gini 
# 模型训练
model.fit(X_train, Y_train)
# 模型预测
y_test = model.predict(X_test)
# print(y_test_hat)


# 6.模型评估
# 方式1：计算得到
Y_test = Y_test.reshape(-1)  # 验证集真实标签，转换为扁平数组
result = (Y_test == y_test)  # 相等返回Ture，不等返回False
accuracy = np.mean(result) * 100  # 计算准确率，注意True为1，False为0
"""
result = [True,True,False,False,True] 即 [1,1,0,0,1]
准确率：Ture的个数/总个数=3/5
准确率：np.mean(result)求均值：(1+1+0+0+1) / 5
"""
print("计算得到准确率为：%.2f%%" % accuracy)


# 方式2：通过参数得到
print ("Score：", model.score(X_test, Y_test))  # 准确率
print ("Classes:", model.classes_)  # 类别
print("获取各个特征的权重:", end='')
print(model.feature_importances_)  # 特征权重


# 7.模型预测
# 前面我们将数据维度设置了2维，所以在降维后，还剩两个特征
N = 100 # 横纵各采样多少个值
x1_min = np.min((X_train.T[0].min(), X_test.T[0].min()))  # 获取特征1最小值
x1_max = np.max((X_train.T[0].max(), X_test.T[0].max()))  # 获取特征1最大值
x2_min = np.min((X_train.T[1].min(), X_test.T[1].min()))  # 获取特征2最小值
x2_max = np.max((X_train.T[1].max(), X_test.T[1].max()))  # 获取特征2最大值
t1 = np.linspace(x1_min, x1_max, N)  # 生成最小值到最大值等分的N个数
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)   # 生成网格采样点
x_show = np.dstack((x1.flat, x2.flat))[0]  # 测试点

"""
x1 = [-0.80794216 -0.79147488 -0.77500761 -0.75854034 -0.74207307 -0.72560579 ...] 100个数据
x2 = [-0.21512034 -0.19865307 -0.1821858  -0.16571852 -0.14925125 -0.13278398 ...] 100个数据
np.dstack((x1.flat, x2.flat))[0]：
[[-0.80794216 -0.21512034],[-0.79147488 -0.19865307],[-0.77500761 -0.28147536]...]也就是分别取x1和x2，组成一个数据点
"""
# 预测
y_show_hat = model.predict(x_show)

y_show_hat = y_show_hat.reshape(x1.shape) # 使之与输入的形状相同
# print(y_show_hat.shape)
# print(y_show_hat[0])

# 8.画图
plt_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])  # 决策区域颜色
plt_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])  # 类别颜色

plt.figure(facecolor='w') # 白色背景画板
plt.pcolormesh(x1, x2, y_show_hat, cmap = plt_light)
# 画测试数据的点信息
plt.scatter(X_test.T[0], X_test.T[1], c = Y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=plt_dark, marker='*')
# 画训练数据的点信息
plt.scatter(X_train.T[0], X_train.T[1], c = Y_train.ravel(), edgecolors='k', s=40, cmap=plt_dark)
plt.xlabel(u'特征属性1', fontsize=15)
plt.ylabel(u'特征属性2', fontsize=15)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(True)
plt.title(u'鸢尾花数据的决策树分类', fontsize=18)
plt.show()