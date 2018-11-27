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
from sklearn import metrics  # ROC/AUC



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


# 5.构建不同个数及深度决策树的随机森林模型并训练、预测、评估
estimators = [1,50,100,500]  # 个数可选值
depth = [1,2,3,7,15]  # 深度可选值
err_list = []  # 错位率
for es in estimators:
       es_list = []
       for d in depth:
              rfc = RandomForestClassifier(n_estimators=es, criterion='gini', max_depth=d, max_features=None, random_state=0)
              rfc.fit(X_train, Y_train)  # 训练
              score = rfc.score(X_test, Y_test)  # 评估
              err = 1 - score  # 错误率
              es_list.append(err)
              print ("%d决策树数目，%d最大深度，正确率:%.2f%%" % (es, d, score * 100))
       err_list.append(es_list)


# 6.画图显示
plt.figure(facecolor='w')  # 画板
i = 0
colors = ['r','b','g','y']  # 颜色
lw = [1,2,4,3]  # 线粗
max_err = 0  # 最小错误率
min_err = 100  # 最大错误率
for es,l in zip(estimators, err_list):  # 遍历决策树个数和错误率，返回决策树个数es和对应的错误率
    plt.plot(depth, l, c=colors[i], lw=lw[i], label=u'树数目:%d' % es)
    max_err = max((max(l),max_err))
    min_err = min((min(l),min_err))
    i += 1
plt.xlabel(u'树深度', fontsize=16)
plt.ylabel(u'错误率', fontsize=16)
plt.legend(loc='upper left', fancybox=True, framealpha=0.8, fontsize=12)
plt.grid(True)
plt.xlim(min(depth), max(depth))
plt.ylim(min_err * 0.99, max_err * 1.01)
plt.title(u'随机森林中树数目、深度和错误率的关系图', fontsize=18)
plt.show()