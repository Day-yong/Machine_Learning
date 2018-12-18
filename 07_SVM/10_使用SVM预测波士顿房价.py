import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import sklearn
from sklearn.svm import SVR  # 对比SVC，是svm的回归形式
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

# 用于预处理数据
def notEmpty(s):
    return s != ''

# 1.加载数据
path = "../datas/boston_housing.data"
# 由于每条数据的格式不统一，所以可以先按一行一条记录的方式来读取，然后再进行数据预处理
fd = pd.read_csv(path, header = None)  # header = None表示没有数据对应的名称，可以给数据加上

"""
部分数据：
0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30 396.90   4.98  24.00
0.02731   0.00   7.070  0  0.4690  6.4210  78.90  4.9671   2  242.0  17.80 396.90   9.14  21.60
0.02729   0.00   7.070  0  0.4690  7.1850  61.10  4.9671   2  242.0  17.80 392.83   4.03  34.70
0.03237   0.00   2.180  0  0.4580  6.9980  45.80  6.0622   3  222.0  18.70 394.63   2.94  33.40

有14列数据
"""

# 2.数据处理
data = np.empty((len(fd), 14))  # 生成形状为[len(fd), 14]的空数组
# 对每条记录依次处理
for i, d in enumerate(fd.values):  # enumerate生成一列索引i(表示fd中的每一条记录), d为其元素(此处d就是fd的一条记录内容)
    d = map(float, filter(notEmpty, d[0].split(' '))) # filter一个函数，一个list
    """
	d[0].split(' ')：将每条记录按空格切分，生成list，可迭代
	notEmpty:调用前面的自定义的函数，将空格表示为False，非空格表示为True
	filter(function,iterable)：将迭代器传入函数中
	map(function,iterable)：对迭代器进行function操作，这里表示根据filter结果是否为真，来过滤list中的空格项
    """
    # map操作后的类型为map类型，转为list类型，并将该条记录存在之前定义的空数组中
    data[i] = list(d)
    # 遍历完所有数据，数据也就处理好了


# 3.划分数据
X, Y = np.split(data, (13,), axis=1)  # 前13个数据划为X，最后一个划为Y
# 将Y拉直为一个扁平的数组
Y = Y.ravel()
# 查看下数据
# print(y.shape)
# print ("样本数据量:%d, 特征个数：%d" % x.shape)
# print ("target样本数据量:%d" % y.shape[0])


# 4.数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# 5.模型构建（参数类型和SVC基本一样）
parameters = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 0.5,0.9,1,5],
    'gamma': [0.001,0.01,0.1,1]
}
model = GridSearchCV(SVR(), param_grid=parameters, cv=3)
model.fit(x_train, y_train)


# 6.获取最优参数
print ("最优参数列表:", model.best_params_)
print ("最优模型:", model.best_estimator_)
print ("最优准确率:", model.best_score_)
# 模型效果输出
print ("训练集准确率:%.2f%%" % (model.score(x_train, y_train) * 100))
print ("测试集准确率:%.2f%%" % (model.score(x_test, y_test) * 100))


# 7.画图
colors = ['g-', 'b-']
ln_x_test = range(len(x_test))
y_predict = model.predict(x_test)

plt.figure(figsize=(16,8), facecolor='w')
plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'真实值')
plt.plot(ln_x_test, y_predict, 'g-', lw = 3, label=u'SVR算法估计值,$R^2$=%.3f' % (model.best_score_))

# 图形显示
plt.legend(loc = 'upper left')
plt.grid(True)
plt.title(u"波士顿房屋价格预测(SVM)")
plt.xlim(0, 101)
plt.show()
