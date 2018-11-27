# -*- coding: utf-8 -*-

"""
数据格式如下：(由于数据太长，这里只显示了一行数据，数据包括"?"所以要进行处理)
Age 	18
Number of sexual partners 	4
First sexual intercourse 	15
Num of pregnancies 	1
Smokes 	0
Smokes (years)	 0
Smokes (packs/year)	 0
Hormonal Contraceptives 	0
Hormonal Contraceptives (years) 	0
IUD 	0
IUD (years) 	0
STDs 	0
STDs (number) 	0
STDs:condylomatosis 	0
STDs:cervical condylomatosis 	0
STDs:vaginal condylomatosis 	0
STDs:vulvo-perineal condylomatosis 	0
STDs:syphilis 	0
STDs:pelvic inflammatory disease 	0	
STDs:genital herpes 	0
STDs:molluscum contagiosum 	0
STDs:AIDS 	0
STDs:HIV 	0
STDs:Hepatitis B 	0
STDs:HPV 	0
STDs: Number of diagnosis	 0
STDs: Time since first diagnosis 	?
STDs: Time since last diagnosis 	?
Dx:Cancer 	0
Dx:CIN 	0
Dx:HPV 	0
Dx 	0
Hinselmann 	0
Schiller 	0
Citology 	0
Biopsy 	0												
"""

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


# 8.模型评估
score = forest.score(X_test, Y_test)  # 准确率
print('准确率：%.2f%%' % (score * 100))

# 9.模型预测
forest_y_score = forest.predict_proba(X_test)  # prodict_proba输出概率
"""
forest_y_score输出为一个n*2的数组，每个[0.95877997, 0.04122003]表示属于0和1的概率
[[0.95877997, 0.04122003],...,[0.94747819, 0.05252181]]
"""


# 10.计算：ROC/AUC
"""
由于forest_y_score输出的是该样本数据0和1的概率，所以我们需要将测试标签数据二值化;
可以使用label_binarize，结果比如 0->[1,0]     1->[0,1]

label_binarize(['a','a','b','b'],classes=('a','b'))  结果为[[0],[0],[1],[1]]不满足
label_binarize(['a','a','b','b'],classes=('a','b','c')) 结果为[[1, 0, 0], [1, 0, 0],[0, 1, 0],[0, 1, 0]]也不满足
但是我们可以将第二个n*2的数组，去掉最后一列就可以了，先进行转置，然后去掉最后一行即可

ravel()是将数组变为扁平的数组，从而可以计算ROC值

## 正确的数据
y_true = label_binarize(y_test[names[-4]],classes=(0,1,2)).T[0:-1].T.ravel()
## 预测的数据 => 获取第一个目标属性的预测值，并将其转换为一维的数组
y_predict = forest_y_score[0].ravel()
其他标签也类似
"""
# 计算ROC值

forest_fpr1, forest_tpr1, _ = metrics.roc_curve(label_binarize(Y_test[names[-4]], classes=(0, 1, 2)).T[0:-1].T.ravel(), forest_y_score[0].ravel())
forest_fpr2, forest_tpr2, _ = metrics.roc_curve(label_binarize(Y_test[names[-3]], classes=(0, 1, 2)).T[0:-1].T.ravel(), forest_y_score[1].ravel())
forest_fpr3, forest_tpr3, _ = metrics.roc_curve(label_binarize(Y_test[names[-2]], classes=(0, 1, 2)).T[0:-1].T.ravel(), forest_y_score[2].ravel())
forest_fpr4, forest_tpr4, _ = metrics.roc_curve(label_binarize(Y_test[names[-1]], classes=(0, 1, 2)).T[0:-1].T.ravel(), forest_y_score[3].ravel())
# 计算AUC值
auc1 = metrics.auc(forest_fpr1, forest_tpr1)
auc2 = metrics.auc(forest_fpr2, forest_tpr2)
auc3 = metrics.auc(forest_fpr3, forest_tpr3)
auc4 = metrics.auc(forest_fpr4, forest_tpr4)
print ("Hinselmann目标属性AUC值：", auc1)
print ("Schiller目标属性AUC值：", auc2)
print ("Citology目标属性AUC值：", auc3)
print ("Biopsy目标属性AUC值：", auc4)


# 11.画图：ROC/AUC图
plt.figure(figsize=(8, 6), facecolor='w')  # 画板
plt.plot(forest_fpr1,forest_tpr1,c='r',lw=2,label=u'Hinselmann目标属性,AUC=%.3f' % auc1)
plt.plot(forest_fpr2,forest_tpr2,c='b',lw=2,label=u'Schiller目标属性,AUC=%.3f' % auc2)
plt.plot(forest_fpr3,forest_tpr3,c='g',lw=2,label=u'Citology目标属性,AUC=%.3f' % auc3)
plt.plot(forest_fpr4,forest_tpr4,c='y',lw=2,label=u'Biopsy目标属性,AUC=%.3f' % auc4)
plt.plot((0,1),(0,1),c='#a0a0a0',lw=2,ls='--')
plt.xlim(-0.001, 1.001)  # x轴范围
plt.ylim(-0.001, 1.001)  # y轴范围
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate(FPR)', fontsize=16)
plt.ylabel('True Positive Rate(TPR)', fontsize=16)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'随机森林多目标属性分类ROC曲线', fontsize=18)
plt.show()