# -*- coding: utf-8 -*-

import numpy as np  # 科学计算包Numpy
import operator  # 运算符模块

"""
对未知类别属性的数据集中的每个点依次执行一下操作：

（1）计算已知类别数据集中的点与当前点之间的距离 
（2）按照距离递增次序排序 
（3）选取与当前点距离最小的k个点 
（4）确定前k个点所在类别的出现频数 
（5）返回当前k个点出现频数最高的类别作为当前点的预测分类
"""

# k-近邻算法
def knn(inX, dataSet, labels, k):  # 输入向量，训练数据，标签，参数k
    dataSetSize = dataSet.shape[0]  # 数据个数
    # 计算差值
    """
	tile函数：将输入向量变成和训练数据一样的形状
	例如：输入向量为[1.2, 1]，
	训练数据为:
	   [[1.0, 1.1],
        [1.0, 1.0],
        [  0,   0],
        [  0, 0.1]]
    变换后为：
       [[1.2,   1],
        [1.2,   1],
        [1.2,   1],
        [1.2,   1]]
    """
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # 计算对应元素的差值
    sqDiffMat = diffMat ** 2  # 每个元素分别平方
    sqDistances = sqDiffMat.sum(axis=1)  # 按行求和
    distances = sqDistances ** 0.5  # 开根号  欧氏距离，求得每个训练数据到输入数据的距离
    sortedDistIndicies = distances.argsort()  # 返回数组值从小到大的索引
    classCount = {}  # 创建一个字典，用于记录每个实例对应的频数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 选择k个距离最小的点，对应标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 统计频数
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    """
	operator.itemgetter(1):返回前面classCount迭代器中的第1列
	sorted中key指定按value值排序
	reverse=True降序
    """
    return sortedClassCount[0][0]  # 返回最多的，多数表决法

# 1.数据获取
def createDataSet():
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# 查看数据
group, labels = createDataSet()
print("训练数据:", group)
print("标签:", labels)


# 预测
result = knn([1.2, 1], group, labels, 3)
print("预测标签为：", result)
