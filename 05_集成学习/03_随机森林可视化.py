# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split  # 数据分割
from sklearn.preprocessing import Imputer  # 数据缺省值处理
from sklearn.preprocessing import MinMaxScaler  # 数据归一化
from sklearn.preprocessing import label_binarize  # 数据二值化
from sklearn.decomposition import PCA  # 降维
from sklearn.ensemble import RandomForestClassifier  # 分类随机森林模型



# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


# 1.读取数据
path = '../datas/risk_factors_cervical_cancer.csv'
df = pd.read_csv(path)
# print(df.columns) 获取特征名称
names = [u'Age', u'Number of sexual partners', u'First sexual intercourse',
       u'Num of pregnancies', u'Smokes', u'Smokes (years)', u'Smokes (packs/year)',
       u'Hormonal Contraceptives', u'Hormonal Contraceptives (years)', u'IUD',
       u'IUD (years)', u'STDs', u'STDs (number)', u'STDs:condylomatosis',
       u'STDs:cervical condylomatosis', u'STDs:vaginal condylomatosis',
       u'STDs:vulvo-perineal condylomatosis', u'STDs:syphilis',
       u'STDs:pelvic inflammatory disease', u'STDs:genital herpes',
       u'STDs:molluscum contagiosum', u'STDs:AIDS', 'STDs:HIV',
       u'STDs:Hepatitis B', u'STDs:HPV', u'STDs: Number of diagnosis',
       u'STDs: Time since first diagnosis', u'STDs: Time since last diagnosis',
       u'Dx:Cancer', u'Dx:CIN', u'Dx:HPV', u'Dx', 
       u'Hinselmann', u'Schiller', u'Citology', u'Biopsy']


# 2.划分数据
# 模型存在多个需要预测的y值，'Hinselmann', 'Schiller', 'Citology', 'Biopsy'
# 如果是这种情况下，简单来讲可以直接模型构建，
# 在模型内部会单独的处理每个需要预测的y值，相当于对每个y创建一个模型
X = df[names[0:-4]]  # 获取特征数据X
Y = df[names[-4:]]  # 获取4个标签数据Y
# print(X.head(1)) 查看下数据


# 3.数据处理
# 由于数据存在"?"
X = X.replace("?", np.NaN)  # 将"?"替换为np.NaN
# 然后np.nan使用Imputer给定缺省值进行数据填充，默认是以列/特征的均值填充
imputer = Imputer(missing_values='NaN')  # 构建模型
X = imputer.fit_transform(X, Y)  # 训练模型并转换数据


# 4.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# print ("训练样本数量:%d,特征属性数目:%d,目标属性数目:%d" % (X_train.shape[0],X_train.shape[1],Y_train.shape[1]))
# print ("测试样本数量:%d" % X_test.shape[0])


# 5.数据归一化
# 分类模型，经常使用的是minmaxscaler归一化，回归模型经常用standardscaler标准化
ss = MinMaxScaler()  # 构建归一化模型
X_train = ss.fit_transform(X_train, Y_train)  # 训练模型并转换数据
X_test = ss.transform(X_test)  # 转换数据


# 6.降维(此数据集的维度比较高，我们可以做降维处理)
pca = PCA(n_components=2)  # 创建PCA模型，指定维度为2
X_train = pca.fit_transform(X_train)  # 训练模型并转换数据
X_test = pca.transform(X_test)


# 7.随机森林模型构建及训练
# n_estimators=100决策树个数，max_depth=1每个决策树深度，random_state=0随机数种子
forest = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=1, random_state=0)
forest.fit(X_train, Y_train)  # 训练模型


# 8.模型可视化
from sklearn import tree
import pydotplus
k = 0
for clf in forest.estimators_:
    dot_data = tree.export_graphviz(clf, out_file=None,  
                         filled=True, rounded=True,  
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("foress_tree_%d.pdf" % k)
    k += 1
    if k == 5:  # 只可视化前5个决策树
        break