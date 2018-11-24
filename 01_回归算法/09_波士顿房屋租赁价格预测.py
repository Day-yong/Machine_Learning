# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings  # 警告处理

from sklearn.linear_model import LassoCV, RidgeCV  # 回归模型
from sklearn.model_selection import train_test_split  # 划分数据集的类
from sklearn.preprocessing import StandardScaler  # 数据标准化的类
from sklearn.preprocessing import PolynomialFeatures  # 模型特征的构造
from sklearn.pipeline import Pipeline  # 管道
from sklearn.model_selection import GridSearchCV  # 模型最优参数选择
from sklearn.linear_model.coordinate_descent import ConvergenceWarning  # 警告处理

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
# 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)
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


# 4.搭建管道
models = [
    Pipeline([
            ('ss', StandardScaler()),  # 数据标准化
            ('poly', PolynomialFeatures()),  # 模型特征的构造
            ('linear', RidgeCV(alphas=np.logspace(-3,1,20)))  # Ridge模型，带交叉验证
        ]),
    Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures()),
            ('linear', LassoCV(alphas=np.logspace(-3,1,20)))  # LASSO模型，带交叉验证
        ])
] 
# 参数字典，字典中的key是属性的名称，value是可选的参数列表
parameters = {
    "poly__degree": [3,2,1], 
    "poly__interaction_only": [True, False], # 不产生交互项，如X1*X2就是交叉项， X1*X1为非交叉项
    "poly__include_bias": [True, False], # 多项式幂为零的特征作为线性模型中的截距;true表示包含
    "linear__fit_intercept": [True, False]
}


# 5.数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# 6.模型训练，画图
# Lasso和Ridge模型比较运行图表展示
titles = ['Ridge', 'Lasso']
colors = ['g-', 'b-']
plt.figure(figsize=(16,8), facecolor='w')  # 画板，大小(16,8)，颜色白色
ln_x_test = range(len(x_test))

plt.plot(ln_x_test, y_test, 'r-', lw=2, label=u'真实值')  # 画真实值，红色

for t in range(2):
    # 获取模型并设置参数
    # GridSearchCV: 进行交叉验证，选择出最优的参数值出来
    # 第一个输入参数：进行参数选择的模型
    # param_grid： 用于进行模型选择的参数字段，要求是字典类型
    # cv: 进行几折交叉验证
    model = GridSearchCV(models[t], param_grid = parameters,cv=5, n_jobs=1) # 五折交叉验证
    # 模型训练-网格搜索
    model.fit(x_train, y_train)
    # 模型效果值获取（最优参数）
    print ("%s算法:最优参数:" % titles[t],model.best_params_)
    print ("%s算法:R值=%.3f" % (titles[t], model.best_score_))
    # 模型预测
    y_predict = model.predict(x_test)
    # 画图
    plt.plot(ln_x_test, y_predict, colors[t], lw = t + 3, label=u'%s算法估计值,$R^2$=%.3f' % (titles[t],model.best_score_))
# 图形显示
plt.legend(loc = 'upper left')
plt.grid(True)
plt.title(u"波士顿房屋价格预测")
plt.show()


# 补充(查看每个特征的对应的模型参数)
# 模型训练 我们选择上面Lasso回归得到的最优参数
names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
model = Pipeline([
            ('ss', StandardScaler()),
            ('poly', PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)),
            ('linear', LassoCV(alphas=np.logspace(-3,1,20), fit_intercept=False))
        ])
# 模型训练
model.fit(x_train, y_train)

# 数据输出
print ("参数:", list(zip(names, model.get_params('linear')['linear'].coef_)))
print ("截距:", model.get_params('linear')['linear'].intercept_)
"""
补充程序会输出线性模型的参数
比如：
参数: [('CRIM', 21.135499741068376), ('ZN', -0.0), ('INDUS', -0.0), ('CHAS', -0.0), 
		('NOX', 0.19539929236955278), ('RM', -0.0), ('AGE', 1.566235617592053), 
		('DIS', -0.38131114313786807), ('RAD', -0.6960425166192609), ('TAX', 0.0), 
		('PTRATIO', -0.0), ('B', -1.5063986238529539), ('LSTAT', 0.0)]
截距: 0.0

其中有些参数是接近0的，那么我们可以认为当前参数对应的特征属性在模型的判别中没有太大的决策信息，
所以，对于这样的属性可以删除；一般情况下，手动删除选择0附近1e-4范围内的特征属性
"""