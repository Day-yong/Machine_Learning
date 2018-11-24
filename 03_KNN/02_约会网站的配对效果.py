# -*- coding: utf-8 -*-

import numpy as np  # 科学计算包Numpy
import operator  # 运算符模块
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
部分txt数据格式为：
| 每年获得的飞行常客里程数 | 玩视频游戏所耗时间百分比 | 每周消费的冰淇淋公升数| 标签|
|		40920			 |8.326976				  |0.953952			    |  3 |
|		14488			 |7.153469				  |1.673904				|  2 |
|		26052			 |1.441871				  |0.805124				|  1 |
"""


# k-近邻算法(具体可见01_k近邻算法.py)
def knn(inX, dataSet, labels, k):  # 输入向量，训练数据，标签，参数k
    dataSetSize = dataSet.shape[0]  # 数据个数
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

    return sortedClassCount[0][0]  # 返回最多的，多数表决法


# 从文本文件中解析数据为Numpy格式，便于计算
def file2matrix(filename):  # 传入文件名
	fr = open(filename)  # 打开文件
	arrayOLines = fr.readlines()  # 按行全部读取数据，返回list
	numberOfLines = len(arrayOLines)  # 得到文件行数
	returnMat = np.zeros((numberOfLines, 3))  # 创建0填充的Numpy矩阵，大小为(样本数目,特征数目)
	classLabelVector = []  # 创建空列表，用于存储标签数据
	index = 0
	for line in arrayOLines:  # 遍历每行数据，即每个样本
		line = line.strip()  # 使用strip函数，截取掉所有的回车字符
		listFromLine = line.split('\t')  # 将上一步得到的整行数据分割成一个元素列表
		returnMat[index, :] = listFromLine[0:3]  # 选取前3个元素，形成特征矩阵，index为样本索引
		classLabelVector.append(int(listFromLine[-1]))  # 索引值-1选取列表中的最后一列，将其存储到classLabelVector中（必须明确告知存储元素值为整型）
		index += 1
	return returnMat, classLabelVector


# 测试
# datingDataMat, datingLabels = file2matrix('datas/datingTestSet.txt') # 注意自己文件相对路径
# print(datingDataMat)
# print(datingLabels[0:20])


# 归一化特征
# 由于数据的范围不一样，所以在计算距离的时候，差值大的对计算结果影响比较大
# 例如40920 - 14488 和8.326976 - 7.153469平方后计算距离，可以认为后面一个可以忽略
# 处理方法：(每列原数据 - 每列最小值)/ (每列最大值 - 每列最小值)
def normal(dataSet):
	minVals = dataSet.min(0)  # 最小值 参数0是的函数可以从列中选取最小值
	maxVals = dataSet.max(0)  # 最大值
	ranges = maxVals - minVals  # 取值范围
	normDateSet = np.zeros(dataSet.shape)  # 创建与dataSet同型的0填充矩阵
	m = dataSet.shape[0]  # 每列数据个数
	normDateSet = dataSet - np.tile(minVals, (m, 1))  # 每列原来的数据减去每列最小值
	normDateSet = normDateSet / np.tile(ranges, (m, 1))  # 对应元素相除，除以取值范围
	return normDateSet, ranges, minVals


# 应用测试集测试约会网站预测函数的效果
def datingClassTest():
    hoRatio = 0.10  # 测试集比例
    datingDataMat, datingLabels = file2matrix('datas/datingTestSet.txt')  # 读取数据
    normMat = normal(datingDataMat)  # 归一化处理
    m = normMat.shape[0]  # 样本数目
    numTestVecs = int(m * hoRatio)  # 用于测试的数据
    errorCount = 0
    for i in range(numTestVecs):
    	# normMat[i, :]待分类的输入数据，normMat[numTestVecs:m, :]训练数据，datingLabels[numTestVecs:m]标签数据，k=3
        classifierResult = knn(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)  # 分类函数
        print("预测结果为 : %d, 真实值为 : %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1
    print("错误率为 : %f" % (errorCount / float(numTestVecs)))


# 针对新数据的预测
def classifyPerson():
    resultList = ['不喜欢的人', '魅力一般的人', '极具魅力的人']
    percentTats = float(input("玩视频游戏所耗时间百分比："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("每年消费冰淇淋公升数："))
    datingDataMat, datingLabels = file2matrix("datas/datingTestSet.txt")
    normMat, ranges, minVals = normal(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = knn((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("你对这个人的印象是：", resultList[classifierResult - 1])


# 测试
classifyPerson()