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

# ch2 = SelectKBest(chi2, k=3)  # 创建卡方系数模型
# X_train = ch2.fit_transform(X_train, Y_train)  # 训练模型，并转换
# X_test = ch2.transform(X_test)  # 在训练好的模型下转换
# select_name_index = ch2.get_support(indices=True)
# print ("对类别判断影响最大的三个特征属性分布是:",ch2.get_support(indices=False))


# ## c.降维
# """
# 降维：
# 因为对于数据而言，如果特征属性比较多，在构建过程中，会比较复杂，这个时候考虑将多维（高维）映射到低维的数据
# 常用方法：
# 	PCA：主成分分析（无监督）
# 	LDA：线性判别分析（有监督）
# """
# pca = PCA(n_components = 2) # 构建一个pca模型对象，设置最终维度是2维
# # 这里是为了后面画图方便，所以将数据维度设置了2维，一般用默认不设置参数就可以
# X_train = pca.fit_transform(X_train)  # 训练模型并转换
# X_test = pca.transform(X_test)  # 转换


# 5.模型构建
model = DecisionTreeClassifier(criterion='entropy', random_state=0, min_samples_split=2)  # 另外也可选gini, min_samples_split剪枝操作
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



# 7.模型可视化
## a.方式一：输出形成dot文件，然后使用graphviz的dot命令将dot文件转换为pdf
# from sklearn import tree
# with open('iris.dot', 'w') as f:
#     # 将模型model输出到给定的文件中
#     f = tree.export_graphviz(model, out_file=f)
# # 命令行执行dot命令： dot -Tpdf iris.dot -o iris.pdf

## b.方式二：直接使用pydotplus插件生成pdf文件 pip install pydotplus
# from sklearn import tree
# import pydotplus 
# dot_data = tree.export_graphviz(model, out_file=None) 
# graph = pydotplus.graph_from_dot_data(dot_data) 
# graph.write_pdf("iris1.pdf") 


# ## c.方式三：直接生成图片(这里加了特征和标签，所以需要将前面的特征选择和降维去掉，不然特征数目不对应)
from sklearn import tree
import pydotplus
dot_data = tree.export_graphviz(model, out_file=None,
						feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],  
                        class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], 
                        filled=True, rounded=True,  
                        special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png("iris2.png")