import random
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 加载数据集
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines(): # 解析文本文件中的数据
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])]) # 数据矩阵
        labelMat.append(float(lineArr[2]))  # 类标签
    return dataMat, labelMat

# 用于在某个区间范围内随机选择一个整数
def selectJrand(i, m):  # i为第一个alpha下标，m是所有alpha对应的数目
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

# 用于调整大于H或小于L的alpha值
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# 完整版Platt SMO的支持函数

# （不加核函数）构建一个仅包含init方法的optStruct类，实现其成员变量的填充
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))


# 计算E值并返回
# 不加核函数
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

# 内循环中的启发式方法，选择第二个alpha
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]  # 构建出一个非零表，函数nonzero返回一个列表

    # 在所有值上进行循环并选择其中使得该变量最大的那个值
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


# 计算误差值并存入缓存中，在对alpha值进行优化之后会用到这个值
def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

# 完整Platt SMO算法中的优化例程（与smoSimple()函数一样，只是使用了自己的数据结构，在参数oS中传递）
# innerL()函数用来选择第二个alpha，并在可能是对其进行优化处理，如果存在任意一对alpha值发生变化，那么返回1
# 不加核函数
def innerL(i, oS):
    Ei = calcEk(oS, i)  # 计算E值
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # 第二个alpha选择中的启发式方法
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)  # 计算误差值，并存入缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * \
             (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[j, :] * oS.X[j, :].T - oS.labelMat[j] * \
             (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

# 完整版Platt SMO的外循环代码
# 不加核函数
def smoP(dataMatIn, classLabels, C, toler, maxIter):  # 数据集、类别标签、常数C、容错率和退出前最大的循环次数
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler)  # 构建一个数据结构来容纳所有的数据

    # 初始化一些控制变量
    iter = 0
    entireSet = True
    alphaPairsChanged = 0

    # 代码主体
    # 退出循环的条件：
    # 1、迭代次数超过指定的最大值；
    # 2、历整个集合都没有对任意alpha值进行修改（即：alphaPairsChanged=0）
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # 遍历所有的值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 调用innerL()函数
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历非边界值（非边界值指的是那些不等于边界0或C的alpha值）
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


# 测试
dataArr, labelArr = loadDataSet('../datas/SVMtestSet.txt')
b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
print('b:',b)
print(alphas[alphas > 0])  # 观察元素大于0的

shape(alphas[alphas > 0])  # 得到支持向量的个数

for i in range(100):
    if alphas[i] > 0.0:
        print("完整版支持向量为:", dataArr[i], labelArr[i])


# 计算w的值
# 该程序最重要的是for循环，for循环中实现的仅仅是多个数的乘积，前面我们计算出的alpha值，大部分是为0
# 虽然遍历数据集中的所有数据，但是起作用的只有支持向量，其他对计算w毫无作用
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m): 
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


ws = calcWs(alphas, dataArr, labelArr)
print("w:",ws)

# 对数据进行分类处理
dataMat = mat(dataArr)
print(dataMat[0] * mat(ws) + b)
print(labelArr[0])

dataMat = mat(dataArr)
print(dataMat[1] * mat(ws) + b)
print(labelArr[1])

dataMat = mat(dataArr)
print(dataMat[2] * mat(ws) + b)
print(labelArr[2])