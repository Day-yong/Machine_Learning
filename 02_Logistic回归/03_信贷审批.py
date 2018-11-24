# -*- coding: utf-8 -*-

"""
信贷审批数据格式如下：
b, 30.83,   0, u, g, w,  v,  1.25, t, t, 01, f, g, 00202, 0, +
b, 32.33, 7.5, u, g, e, bb, 1.585, t, f,  0, t, s, 00420, 0, -
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings  # 警告处理

from sklearn.linear_model import LogisticRegressionCV  # Logistic回归
from sklearn.linear_model.coordinate_descent import ConvergenceWarning  # 警告处理
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.pipeline import Pipeline  # 管道


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 拦截异常
warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)


# 1.读取数据
path = '../datas/crx.data'
names = ['A1','A2','A3','A4','A5','A6','A7','A8',
         'A9','A10','A11','A12','A13','A14','A15','A16']
df = pd.read_csv(path, header=None, names=names)
# print ("数据条数:", len(df))

# 2.数据预处理
# a.异常数据过滤
datas = df.replace("?", np.nan)
datas = datas.dropna(how = 'any')
# print ("过滤后数据条数:", len(df))

# 查看信贷审批数据格式
# b, 30.83,   0, u, g, w,  v,  1.25, t, t, 01, f, g, 00202, 0, +
# 里面包括不是数值型的，我们需要将数据都表示为float类型，可以使用哑编码
# 通过df.info() 查看各个列的字符相关信息如下：
# A1     653 non-null object
# A2     653 non-null object
# A3     653 non-null float64
# A4     653 non-null object
# A5     653 non-null object
# A6     653 non-null object
# A7     653 non-null object
# A8     653 non-null float64
# A9     653 non-null object
# A10    653 non-null object
# A11    653 non-null int64
# A12    653 non-null object
# A13    653 non-null object
# A14    653 non-null object
# A15    653 non-null int64
# A16    653 non-null object
# 通过df.A1.value_counts()查看数据类型为object的A1取值情况，其他改变A1即可


# c.自定义的一个哑编码实现方式：将v变量转换成为一个向量/list集合的形式
def parse(v, l):
    # v是一个字符串，需要进行转换的数据
    # l是一个类别信息，其中v是其中的一个值
    return [1 if i == v else 0 for i in l]
# 哑编码实验
# print(parse('v', ['v', 'y', 'l'])) # [1, 0, 0]
# print(parse('y', ['v', 'y', 'l'])) # [0, 1, 0]
# print(parse('l', ['v', 'y', 'l'])) # [0, 0, 1]

# d.定义一个处理每条数据的函数
def parseRecord(record):
    result = []
    # 格式化数据，将离散数据转换为连续数据
    a1 = record['A1']
    for i in parse(a1, ('a', 'b')):
        result.append(i)
    
    result.append(float(record['A2']))
    result.append(float(record['A3']))
    
    # 将A4的信息转换为哑编码的形式; 对于DataFrame中，原来一列的数据现在需要四列来进行表示
    a4 = record['A4']
    for i in parse(a4, ('u', 'y', 'l', 't')):
        result.append(i)
    
    a5 = record['A5']
    for i in parse(a5, ('g', 'p', 'gg')):
        result.append(i)
    
    a6 = record['A6']
    for i in parse(a6, ('c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff')):
        result.append(i)
    
    a7 = record['A7']
    for i in parse(a7, ('v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o')):
        result.append(i)
    
    result.append(float(record['A8']))
    
    a9 = record['A9']
    for i in parse(a9, ('t', 'f')):
        result.append(i)
        
    a10 = record['A10']
    for i in parse(a10, ('t', 'f')):
        result.append(i)
    
    result.append(float(record['A11']))
    
    a12 = record['A12']
    for i in parse(a12, ('t', 'f')):
        result.append(i)
        
    a13 = record['A13']
    for i in parse(a13, ('g', 'p', 's')):
        result.append(i)
    
    result.append(float(record['A14']))
    result.append(float(record['A15']))
    
    a16 = record['A16']
    if a16 == '+':
        result.append(1)
    else:
        result.append(0)
        
    return result

# e.数据特征处理(将数据转换为数值类型的)
# 重新定义names
# 因为需要对A4进行哑编码操作，需要使用四列来表示一列的值
new_names =  ['A1_0', 'A1_1',
			'A2','A3',
			'A4_0','A4_1','A4_2','A4_3',
			'A5_0', 'A5_1', 'A5_2',
			'A6_0', 'A6_1', 'A6_2', 'A6_3', 'A6_4', 'A6_5', 'A6_6', 'A6_7', 'A6_8', 'A6_9', 'A6_10', 'A6_11', 'A6_12', 'A6_13',
			'A7_0', 'A7_1', 'A7_2', 'A7_3', 'A7_4', 'A7_5', 'A7_6', 'A7_7', 'A7_8',
			'A8',
			'A9_0', 'A9_1','A10_0',
			'A10_1',
			'A11',
			'A12_0', 'A12_1',
			'A13_0', 'A13_1', 'A13_2',
			'A14','A15','A16']
datas = datas.apply(lambda x: pd.Series(parseRecord(x), index = new_names), axis=1)


# 3.数据划分
X = datas[new_names[0:-1]]
Y = datas[new_names[-1]]


# 4.数据分割
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# 5.数据正则化操作(归一化)
ss = StandardScaler()
# 模型训练一定是在训练集合上训练的
X_train = ss.fit_transform(X_train)  # 训练正则化模型，并将训练数据归一化操作
X_test = ss.transform(X_test)  # 使用训练好的模型对测试数据进行归一化操作

# 6.模型构建与训练
"""
Logistic算法模型构建
LogisticRegression中，参数说明：
 	penalty => 惩罚项方式，即使用何种方式进行正则化操作(可选: l1或者l2)
 	C => 惩罚项系数，即L1或者L2正则化项中给定的那个λ系数(ppt上)
LogisticRegressionCV中，参数说明：
 	LogisticRegressionCV表示LogisticRegression进行交叉验证选择超参数(惩罚项系数C/λ)
 	Cs => 表示惩罚项系数的可选范围

solver: 参数优化方式
	当penalty为l1的时候，参数只能是：liblinear(坐标轴下降法)；
	nlbfgs和cg都是关于目标函数的二阶泰勒展开
		当penalty为l2的时候，参数可以是：lbfgs(拟牛顿法)、newton-cg(牛顿法变种)，seg(minibatch)
	维度<10000时，lbfgs法比较好；维度>10000时，cg法比较好；显卡计算的时候，lbfgs和cg都比seg快
multi_class: 分类方式参数
	参数可选: ovr(默认)、multinomial
		这两种方式在二元分类问题中，效果是一样的；在多元分类问题中，效果不一样
	ovr: one-vs-rest， 对于多元分类的问题，先将其看做二元分类，分类完成后，再迭代对其中一类继续进行二元分类
	multinomial: many-vs-many（MVM）,即Softmax分类效果
"""
lr = LogisticRegressionCV(Cs=np.logspace(-4,1,50), fit_intercept=True, 
						penalty='l2', solver='lbfgs', tol=0.01, multi_class='ovr')
lr.fit(X_train, Y_train)


# 7.Logistic算法效果输出
lr_r = lr.score(X_train, Y_train)
print ("Logistic算法R值（训练集上的准确率）：", lr_r)
print ("Logistic算法稀疏化特征比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))
print ("Logistic算法参数：",lr.coef_)
print ("Logistic算法截距：",lr.intercept_)


# 8.Logistic算法预测（预测所属类别）
y_predict = lr.predict(X_test)


# Logistic算法获取概率值(就是Logistic算法计算出来的结果值)
y1 = lr.predict_proba(X_test)
print(y1)

# 9.画图
x_len = range(len(X_test))
plt.figure(figsize=(14,7), facecolor='w')
plt.ylim(-0.1,1.1)
plt.plot(x_len, Y_test, 'ro',markersize = 6, zorder=3, label=u'真实值')
plt.plot(x_len, y_predict, 'go', markersize = 10, zorder=2, label=u'Logis算法预测值,$R^2$=%.3f' % lr.score(X_test, Y_test))
plt.legend(loc = 'center right')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'是否审批(0表示未通过，1表示通过)', fontsize=18)
plt.title(u'Logistic回归算法预测结果', fontsize=20)
plt.show()