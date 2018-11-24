# -*- coding: utf-8 -*-

import numpy as np  # 科学计算包Numpy
import operator  # 运算符模块
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# 将图像格式转化为向量 32*32 --> 1*1024
def img2vector(filename):
    returnVect = np.zeros((1, 1024))  # 创建1*1024的0填充向量矩阵
    fr = open(filename)  # 打开文件
    for i in range(32):  # 读取文件的前32行，前32列
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect  # 返回每个图像的向量

# 测试代码
# testVector = img2vector('datas/testDigits/0_13.txt')
# print(testVector[0, 0:31])
# print(testVector[0, 32:63])


# 手写数字识别系统的测试代码
def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('../datas/trainingDigits')  # 获取目录内容，返回文件名列表
    m = len(trainingFileList)  # 计算目录中的文件数
    trainingMat = np.zeros((m, 1024))  # 创建一个m行1024列的训练矩阵，每行数据存储一个图像
    for i in range(m):   # 从文件名中解析出分类数字，文件名命名是按照规则命名的
        fileNameStr = trainingFileList[i]  # 获取文件名，格式为0_0.txt
        fileStr = fileNameStr.split('.')[0]  # 按'.'拆分得到list['0_0','txt']并取出'0_0'
        classNumStr = int(fileStr.split('_')[0]) # 按'_'拆分得到list['0','0']并取出第一个0，为标签
        hwLabels.append(classNumStr)    # 存储到hwLabels向量中
        trainingMat[i, :] = img2vector('../datas/trainingDigits/%s' % fileNameStr) # 调用函数，每遍历一个文件就处理为二维数组中的一行
    testFileList = os.listdir('../datas/testDigits')  # 测试数据集
    errorCount = 0.0  # 错误率
    mTest = len(testFileList)  # 测试集数目
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('../datas/testDigits/%s' % fileNameStr)
        classifierResult = knn(vectorUnderTest, trainingMat, hwLabels, 3)  # 使用classify0函数测试该目录下的每一个文件
        print("预测结果为：%d，真实值为：%d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("预测错误的总数为：%d" % errorCount)
    print("手写数字识别系统的错误率为：%f" % (errorCount / float(mTest)))

# 测试手写数字识别系统
handwritingClassTest()