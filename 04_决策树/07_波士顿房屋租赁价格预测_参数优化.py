# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings  # 警告处理

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor  # 回归决策树模型
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import MinMaxScaler  # 数据归一化
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.pipeline import Pipeline  # 管道
from sklearn.model_selection import GridSearchCV  # 网格搜索交叉验证，用于选择最优的参数


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)

# 用于预处理数据
def notEmpty(s):
    return s != ''

# 1.读取数据
names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
path = "../datas/boston_housing.data"
# 由于每条数据的格式不统一，所以可以先按一行一条记录的方式来读取，然后再进行数据预处理
fd = pd.read_csv(path, header = None)  # header = None表示没有数据对应的名称，可以给数据加上


# 2.数据处理
data = np.empty((len(fd), 14))  # 生成形状为[len(fd), 14]的空数组
# 对每条记录依次处理
for i, d in enumerate(fd.values):  # enumerate生成一列索引i(表示fd中的每一条记录), d为其元素(此处d就是fd的一条记录内容)
    d = map(float, filter(notEmpty, d[0].split(' '))) # filter一个函数，一个list
    data[i] = list(d)
    # 遍历完所有数据，数据也就处理好了


# 3.划分数据
X, Y = np.split(data, (13,), axis=1)  # 前13个数据划为X，最后一个划为Y
Y = Y.ravel()


# 4.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=14)
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (X_train.shape[0], X_test.shape[0]))


# 5.参数优化
pipes = [
	Pipeline([
			('mms', MinMaxScaler()),  # 归一化处理
			('pca', PCA()),  # 降维
			('decision', DecisionTreeRegressor(criterion='mse'))  # 回归决策树模型，使用均方误差
		]),
	Pipeline([
			('mms', MinMaxScaler()),  # 归一化处理
			('decision', DecisionTreeRegressor(criterion='mse'))  # 回归决策树模型
		]),
	Pipeline([
			('decision', DecisionTreeRegressor(criterion='mse'))  # 回归决策树模型
		])
]
# 模型可用参数
parameters = [
	{
	"pca__n_components": [0.25, 0.5, 0.75, 1],
	"decision__max_depth": np.linspace(1, 20, 20).astype(np.int8)
	},
	{
	"decision__max_depth": np.linspace(1, 20, 20).astype(np.int8)
	},
	{
	"decision__max_depth": np.linspace(1, 20, 20).astype(np.int8)
	}
]


# 6.模型训练
for t in range(3):  # 遍历管道
    pipe = pipes[t]  # 选择管道
    gscv = GridSearchCV(pipe, param_grid=parameters[t])  # 构建模型
    gscv.fit(X_train, Y_train)  # 训练模型
    print (t,"score值:",gscv.best_score_,"最优参数列表:", gscv.best_params_)


"""
运行结果为：
0 score值: 0.4001529052721232 最优参数列表: {'decision__max_depth': 7, 'pca__n_components': 0.75}
1 score值: 0.7569661898236847 最优参数列表: {'decision__max_depth': 4}
2 score值: 0.7565404744169743 最优参数列表: {'decision__max_depth': 4}
"""

# 7.使用最优模型参数查看正确率
mms_best = MinMaxScaler()
dtr = DecisionTreeRegressor(criterion='mse', max_depth=4)
X_train = mms_best.fit_transform(X_train, Y_train)
X_test = mms_best.transform(X_test)
dtr.fit(X_train, Y_train)
print ("正确率:", dtr.score(X_test, Y_test))