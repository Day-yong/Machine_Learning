from numpy import *
import numpy as np
import matplotlib.pyplot as plt

# 1、读取数据
def loadDateSet(fileName):
    dataMat = []
    fr = open(fileName)  # 打开文件
    for line in fr.readlines():  # 遍历文件的每一行(每行表示一个数据)
        curLine = line.strip().split('\t')  # 处理每行数据，返回字符串list
        fltLine = list(map(float, curLine))  # 使用float函数处理list中的字符串，使其为float类型
        dataMat.append(fltLine)  # 将该数据加入到数组中
    return dataMat

# 2、向量距离计算
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA-vecB, 2)))  # 欧氏距离


# 3、构建一个包含k个随机质心的集合
def randCent(dataSet, k):
    n = shape(dataSet)[1]  # 数据特征个数(即数据维度)
    # 创建一个0矩阵，其中zeros为创建0填充的数组，mat是转换为矩阵，用于存放k个质心
    centroids = mat(zeros((k, n)))
    for i in range(n):  # 遍历每个特征
        minI = min(dataSet[:,i])  # 获取最小值
        rangeI = float(max(dataSet[:, i]) - minI)  # 范围
        centroids[:, i] = minI + rangeI * random.rand(k, 1)  # 最小值+范围*随机数
    return centroids

# 4.K均值聚类算法
"""
dataSet:数据集
k:簇的个数
distMeas:距离计算
createCent:创建k个随机质心
关于距离计算方式与随机生成k个质心可以选择其他方法
"""
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]  # 数据数目
    clusterAssment = mat(zeros((m, 2)))  
    # 储存每个点的簇分配结果，第一列记录簇索引，第二列记录误差，
    # 误差指当前点到簇质心的距离，可用于评估聚类的效果
    centroids = createCent(dataSet, k)  # 质心生成
    clusterChanged = True  # 标记变量，为True则继续迭代
    while clusterChanged:
        clusterChanged = False
        # 1.寻找最近的质心
        for i in range(m):  # 遍历每个数据
            minDist = inf  # 最小距离
            minIndex = -1  # 最小距离的索引
            for j in range(k):  # 遍历每个质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])  # 计算该点到每个质心的距离
                if distJI < minDist:  # 与之前的最小距离比较
                    minDist = distJI  # 更新最小距离
                    minIndex = j  # 更新最小距离的索引
            # 到此，便得到了该点到哪个质心距离最近
            # =======================================
            if clusterAssment[i, 0] != minIndex:  # 如果之前记录的簇索引不等于目前最小距离的簇索引
                clusterChanged = True  # 设置为True，继续遍历，直到簇分配结果不再改变为止
            clusterAssment[i, :] = minIndex, minDist**2  # 记录新的簇索引和误差
        # print(centroids)
        # 2.更新质心的位置
        for cent in range(k):
            ptsInclust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获取给定簇的所有点
            """
            clusterAssment[:, 0].A == cent：表示clusterAssment第一列簇索引是否等于当前的簇
            nonzero：返回一个元祖，第一个元素为True所在的行，第二个元素为True所在的列，这里取为行，即取出给定簇的数据
            例如：
            a = mat([[1,1,0],[1,1,0],[1,0,3]])
            nonzero(a)  # 返回非0元素所在行和列
            (array([0, 0, 1, 1, 2, 2], dtype=int64), array([0, 1, 0, 1, 0, 2], dtype=int64))
            """
            centroids[cent, :] = mean(ptsInclust, axis=0)  # 然后计算均值，axis=0沿着列方向
    return centroids, clusterAssment  # 返回质心与点分配结果

# 5.二分K均值聚类算法
"""
dataSet:数据集
k:所期望的簇的个数
distMeas:距离计算
可以改变所使用的距离计算方法（这里使用的是欧氏距离）
"""
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]  # 数据数目
    clusterAssment = mat(zeros((m,2)))
    # 储存每个点的簇分配结果，第一列记录簇索引，第二列记录误差，
    # 误差指当前点到簇质心的距离，可用于评估聚类的效果 
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 计算整个数据集的质心(将所有点看成一个簇)
    centList =[centroid0]  # 创建一个列表用于保留所有的质心
    for j in range(m):  # 遍历每个数据
        # 计算每个数据到质心的距离(误差值)，记录在簇分配结果对应位置
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):  # 当质心数小于期望簇的数目
        lowestSSE = inf  # 最小SSE
        for i in range(len(centList)):  # 遍历列表中的每个簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]  # 获得簇对应的数据
            """
            调用kMeans，进行k=2的KMeans聚类
            生成两个质心（簇），同时给出两个簇的点分配结果(误差值)
            """
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])  # 划分后两个簇的总误差
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])  # 剩余簇(未进行划分的簇)的总误差
            print("划分后生产的两个簇的总误差和未划分的簇的总误差: ",sseSplit,sseNotSplit)  # 打印输出
            if (sseSplit + sseNotSplit) < lowestSSE:  # 如果划分后的总误差小于之前的最小SSE
                bestCentToSplit = i  # 记录当前被划分的簇
                bestNewCents = centroidMat  # 记录划分后的两个质心
                bestClustAss = splitClustAss.copy()  # copy划分后的两个簇的误差值
                lowestSSE = sseSplit + sseNotSplit  # 更新最小SSE
        # 更新划分后两个簇的索引(一个是质心列表的长度，另一个是当前被划分的簇的索引)
        # 例如一开始为两个簇[0,1]，然后选择1划分后得到[0,1,2]，其中0不动，1被删除被加入新的质心，索引为1和2
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('被划分的簇的索引：',bestCentToSplit)
        print('被划分的簇的数据数目：', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0] # 新的第一个质心添加到centList列表
        centList.append(bestNewCents[1,:].tolist()[0])  # 新的第二个质心添加到centList列表
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss  # 更新被重新划分的数据的分配结果
    return mat(centList), clusterAssment  # 返回质心列表和簇分配结果


# 测试
datMat = mat(loadDateSet('../datas/testSet2.txt'))
centList, myNewAssments = biKmeans(datMat, 3)
print(centList)


# 5.画图
marker = ['s', 'o', '^', '<']  # 散点图点的形状
color = ['b','m','c','g']  # 颜色
X = np.array(datMat)  # 数据点
CentX = np.array(centList)  # 质心点4个
Cents = np.array(myNewAssments[:,0])  # 每个数据点对应的簇
for i,Centroid in enumerate(Cents):  # 遍历每个数据对应的簇，返回数据的索引即其对应的簇
    plt.scatter(X[i][0], X[i][1], marker=marker[int(Centroid[0])],c=color[int(Centroid[0])])  # 按簇画数据点
plt.scatter(CentX[:,0],CentX[:,1],marker='*',c = 'r')  # 画4个质心
plt.show()