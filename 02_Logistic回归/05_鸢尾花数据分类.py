# -*- coding: utf-8 -*-

"""
数据格式如下：
5.1, 3.5, 1.4, 0.2, Iris-setosa
7.0, 3.2, 4.7, 1.4, Iris-versicolor
7.3, 2.9, 6.3, 1.8, Iris-virginica
数据有三类
可以通过df['cla'].value_counts()查看
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LogisticRegressionCV  # Logistic回归模型
from sklearn.linear_model.coordinate_descent import ConvergenceWarning  # 警告处理
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.preprocessing import label_binarize  # 数据转换为矩阵形式
from sklearn import metrics

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 异常拦截
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)


# 1.加载数据
path = '../datas/iris.data'
names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'cla']
df = pd.read_csv(path, header=None, names=names)


# 2.数据预处理
# 需要将最后一个特征进行哑编码
# a.自定义哑编码
def parseRecord(record):
    result=[]
    r = zip(names,record)
    for name,v in r:
        if name == 'cla':
            if v == 'Iris-setosa':
                result.append(1)
            elif v == 'Iris-versicolor':
                result.append(2)
            elif v == 'Iris-virginica':
                result.append(3)
            else:
                result.append(np.nan)
        else:
            result.append(float(v))
    return result

# b.数据转换
datas = df.apply(lambda r: pd.Series(parseRecord(r),index=names), axis=1)

# c.异常数据处理
datas = datas.dropna(how='any')

# d.数据获取
X = datas[names[0:-1]]
Y = datas[names[-1]]


# 3.数据划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)


# 4.数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# 5.模型构建与训练
# print(help(LogisticRegressionCV))查看帮助信息
lr = LogisticRegressionCV(Cs=np.logspace(-4,1,50), cv=3, fit_intercept=True, 
							penalty='l2', solver='lbfgs', tol=0.01, multi_class='multinomial')
lr.fit(X_train, Y_train)

# 6.模型评估
# 将正确的数据转换为矩阵形式(每个类别使用向量的形式来表述)
y_test_hot = label_binarize(Y_test, classes=(1,2,3))
# 得到预测的损失值
lr_y_score = lr.decision_function(X_test)
# 计算roc的值
lr_fpr, lr_tpr, lr_threasholds = metrics.roc_curve(y_test_hot.ravel(),lr_y_score.ravel())
# threasholds阈值
# 计算auc的值
lr_auc = metrics.auc(lr_fpr, lr_tpr)
print ("Logistic算法R值：", lr.score(X_train, Y_train))
print ("Logistic算法AUC值：", lr_auc)


# 7. 模型预测
lr_y_predict = lr.predict(X_test)

# 画图1：ROC曲线画图
plt.figure(figsize=(8, 6), facecolor='w')
plt.plot(lr_fpr,lr_tpr,c='r',lw=2,label=u'Logistic算法,AUC=%.3f' % lr_auc)
plt.plot((0,1),(0,1),c='#a0a0a0',lw=2,ls='--')
plt.xlim(-0.01, 1.02)  # 设置X轴的最大和最小值
plt.ylim(-0.01, 1.02)  # 设置y轴的最大和最小值
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate(FPR)', fontsize=16)
plt.ylabel('True Positive Rate(TPR)', fontsize=16)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'鸢尾花数据Logistic回归分类的ROC/AUC', fontsize=18)
plt.show()


## 画图2：预测结果画图
x_test_len = range(len(X_test))
plt.figure(figsize=(12, 9), facecolor='w')
plt.ylim(0.5,3.5)
plt.plot(x_test_len, Y_test, 'ro',markersize = 6, zorder=3, label=u'真实值')
plt.plot(x_test_len, lr_y_predict, 'go', markersize = 10, zorder=2, label=u'Logis算法预测值,$R^2$=%.3f' % lr.score(X_test, Y_test))
plt.legend(loc = 'lower right')
plt.xlabel(u'数据编号', fontsize=18)
plt.ylabel(u'种类', fontsize=18)
plt.title(u'Logistic算法对鸢尾花数据分类', fontsize=20)
plt.show()