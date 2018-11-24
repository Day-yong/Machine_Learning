# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings  # 警告处理

from sklearn.linear_model import LogisticRegressionCV  # 逻辑回归模型
from sklearn.linear_model.coordinate_descent import ConvergenceWarning  # 警告处理
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import StandardScaler  # 数据标准化


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 拦截异常
warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)


# 1.读取数据
# 1000025, 5, 1, 1, 1, 2, 1, 3, 1, 1, 2
path = '../datas/breast-cancer-wisconsin.data'
names = ['id','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
         'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei',
         'Bland Chromatin','Normal Nucleoli','Mitoses','Class']
df = pd.read_csv(path, header=None, names=names)


# 2.异常数据处理
datas = df.replace('?', np.nan)  # 将非法字符"?"替换为np.nan
datas = datas.dropna(how = 'any')  # 将行中有空值的行删除


# 3.数据提取以及数据分割
# 提取
# 第一列数据为id，对分类决策没有帮助，所以无需作为特征数据
X = datas[names[1:10]]
Y = datas[names[10]]


# 4.数据分割
# test_size：测试集所占比例
# random_state：保证每次分割所产生的数据集是完全相同的
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# 5.数据归一化
ss = StandardScaler()  # 创建归一化
X_train = ss.fit_transform(X_train)  # 训练模型及归一化数据


# 6.模型训练
# 构建模型
# Logistic回归是一种分类算法，不能应用于回归中(也即是说对于传入模型的y值来讲，不能是float类型，必须是int类型)
lr = LogisticRegressionCV(multi_class='ovr',fit_intercept=True, Cs=np.logspace(-2, 2, 20), 
						  cv=2, penalty='l2', solver='lbfgs', tol=0.01)
"""
multi_class: 分类方式参数
	参数可选: ovr(默认)、multinomial
		这两种方式在二元分类问题中，效果是一样的；在多元分类问题中，效果不一样
	ovr: one-vs-rest， 对于多元分类的问题，先将其看做二元分类，分类完成后，再迭代对其中一类继续进行二元分类
	multinomial: many-vs-many（MVM）,即Softmax分类效果

penalty: 过拟合解决参数,l1或者l2
solver: 参数优化方式
	当penalty为l1的时候，参数只能是：liblinear(坐标轴下降法)；
nlbfgs和cg都是关于目标函数的二阶泰勒展开
	当penalty为l2的时候，参数可以是：lbfgs(拟牛顿法)、newton-cg(牛顿法变种)，seg(minibatch)
维度<10000时，lbfgs法比较好；维度>10000时，cg法比较好；显卡计算的时候，lbfgs和cg都比seg快
"""

# 模型训练
re=lr.fit(X_train, Y_train)


# 7.模型评估
r = re.score(X_train, Y_train)
print ("R值（准确率）：", r)
print ("稀疏化特征比率：%.2f%%" % (np.mean(lr.coef_.ravel() == 0) * 100))
print ("参数：",re.coef_)
print ("截距：",re.intercept_)
print(re.predict_proba(X_test))  # 获取sigmoid函数返回的概率值


# 模型持久化与加载模型可按需求操作
# 8.模型保存
# 引入包
from sklearn.externals import joblib
# 要求文件夹必须存在
joblib.dump(ss, "result/ss.model") # 将标准化模型保存
joblib.dump(lr, "result/lr.model") # 将模型保存


# # 9.数据预测
# # a. 预测数据格式化(归一化)
# X_test = ss.transform(X_test) # 使用模型进行归一化操作
# # b. 结果数据预测
# Y_predict = re.predict(X_test)


# # 10.画图
# x_len = range(len(X_test))
# plt.figure(figsize=(14,7), facecolor='w')
# plt.ylim(0,6)
# plt.plot(x_len, Y_test, 'ro',markersize = 8, zorder=3, label=u'真实值')
# plt.plot(x_len, Y_predict, 'go', markersize = 14, zorder=2, label=u'预测值,$R^2$=%.3f' % re.score(X_test, Y_test))
# plt.legend(loc = 'upper left')
# plt.xlabel(u'数据编号', fontsize=18)
# plt.ylabel(u'乳腺癌类型', fontsize=18)
# plt.title(u'Logistic回归算法对数据进行分类', fontsize=20)
# plt.show()