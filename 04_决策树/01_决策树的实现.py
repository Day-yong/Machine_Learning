# -*- coding: utf-8 -*-

import operator
from math import log


# 计算给定数据集的香农熵
# H(x) = -sum{p(i)log[p(i)]}
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)  # 计算数据集的数目
	labelCounts = {}  # 创建空字典，key为标签，value为数据集中为key标签的数据总数目
	for featVec in dataSet:  # 遍历每条数据
		currentLabel = featVec[-1]  # 获取当前数据的标签
		if currentLabel not in labelCounts.keys():  # 判断当前标签是否在字典中，不在字典中就扩展字典，并将value设为0
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1  # 如果字典中已经存在此标签，就将其value加1，即数目加1
	shannonEnt = 0.0  # 香农熵
	for key in labelCounts:  # 遍历字典，计算每个标签各占总数据集的比例
		prob = float(labelCounts[key]) / numEntries  # 获取该标签的value，除以数据集的数目
		shannonEnt -= prob * log(prob, 2)  # 按香农熵公式计算香农熵

	return shannonEnt

"""
得到熵后，按照最大信息增益的方法划分数据集，当然也可以按照信息增益率或基尼指数来划分
"""

# 划分数据集
def splitDataSet(dataSet, axis, value):  # 待划分的数据集、划分数据集特征、需要返回的特征的值
    retDataSet = []  # 创建新的list对象
    for featVec in dataSet:  # 抽取数据
        if featVec[axis] == value:  # 如果要划分的数据集特征和需要返回的特征的值相等
        	# 去掉该属性列
            reducedFeatVec = featVec[:axis]  # 取该属性的前面的属性
            reducedFeatVec.extend(featVec[axis + 1:]) # 然后拼接上该属性后面的属性
            retDataSet.append(reducedFeatVec)  # 得到每条数据与原先数据相比，少了该属性列
    return retDataSet  # 最后得到比原数据集少了一列属性并且只有删除的那列属性值为value的数据集
"""
执行过程：
[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
splitDataSet(myDat, 0, 1)
[[1, 'yes'], [1, 'yes'], [0, 'no']]
先判断下标为0的属性，如果属性值为1就留下来，并且将该属性去掉
"""

# 选择最好的数据集划分方式——实现选取特征，划分数据集，计算得出最好的划分数据集的特征
# Gain = H(D) - H(D|A)
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 计算特征属性的个数
    baseEntropy = calcShannonEnt(dataSet)  # 调用前面写好的calcShannonEnt函数，计算整个数据集的原始香农熵
    bestInfoGain = 0.0  # 记录信息增益
    bestFeature = -1  # 最优分裂属性
    for i in range(numFeatures):  # 遍历每个属性
        featList = [example[i] for example in dataSet]  # 将该属性下的每条数据的属性值写入新的list中
        uniqueVals = set(featList)  # 使用set数据类型，除去list中重复的，每个属性值只出现一次
        newEntropy = 0.0  # 新香农熵
        for value in uniqueVals:  # 遍历该属性中的每个可能的属性值
            subDataSet = splitDataSet(dataSet, i, value)  # 调用前面写好的splitDataSet函数，对每个属性值划分一次数据集
            prob = len(subDataSet) / float(len(dataSet))  # 计算该属性值为value的占整个数据集的比例
            # 计算该属性的一个可能属性值的新熵，通过遍历该属性中的每个可能的属性值，将每个可能属性值的新熵求和，得到该属性的新熵
            newEntropy += prob * calcShannonEnt(subDataSet)  
            
        infoGain = baseEntropy - newEntropy  # 求信息增益
        if infoGain > bestInfoGain:  # 如果此信息增益大于之前的记录的信息增益
            bestInfoGain = infoGain  # 更新最大的信息增益
            bestFeature = i  # 并且记录当前的属性索引
    return bestFeature  # 返回最好特征划分的索引值


# 多数表决方法
def majorityCnt(classList):
    classCount = {}  # 创建唯一值的数据字典，用于存储每个类标签出现的频率
    for vote in classList:
        if vote not in classCount.keys():  # 如果该标签不再字典中，就扩展字典，并将value值设为0
        	classCount[vote] = 0
        classCount[vote] += 1  # 否则就+1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  # 利用operator操作键值排序字典，reverse=True倒序

    return sortedClassCount[0][0]  # 取出第一个，即投票最多的


# 创建树的函数代码
def createTree(dataSet, labels):  # 数据集和标签列表
    classList = [example[-1] for example in dataSet]  # 获取数据集的标签（数据集每条记录的最后列数据）
    # 递归停止的第一个条件
    if classList.count(classList[0]) == len(classList):  # 类别完全相同就停止继续划分数据
        return classList[0]
    # 递归停止的第二个条件
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类别（无法简单地返回唯一的类标签，使用前面的多数表决方法）
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 找出最佳数据集划分的特征属性的索引
    bestFeatLabel = labels[bestFeat]  # 获取最佳索引对应的值
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])  # 删除最佳索引的列
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)  # 使用set集合去除重复的值，得到列表包含的所有属性值
    for value in uniqueVals:  # 遍历所有最佳划分的分类标签
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)  # 递归调用
    return myTree



# 创建数据集
def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['第一个特征', '第二个特征']
    return dataSet, labels


myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print("myTree:", myTree)
# myTree: {'第一个特征': {0: 'no', 1: {'第二个特征': {0: 'no', 1: 'yes'}}}}