# -*- coding: utf-8 -*-

"""
时间与功率的线性关系
"""

## 引入所需要的全部包
from sklearn.model_selection import train_test_split  # 数据划分的类
from sklearn.linear_model import LinearRegression  # 线性回归的类
from sklearn.preprocessing import StandardScaler  # 数据标准化的类

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time


## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False


## 加载数据
# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣机用电功率、热水器用电功率
path1 = '../datas/household_power_consumption_1000.txt'  # 数据相对路径
df = pd.read_csv(path1, sep=';', low_memory=False)

# 没有混合类型的时候可以通过low_memory=False调用更多内存，加速效率
# 可以通过print(df.info())查看格式信息
# print(df.head()) 获取前五行数据查看

# 异常数据处理(异常数据过滤)
new_df = df.replace('?', np.nan)  # 替换非法字符为np.nan
datas = new_df.dropna(axis=0, how = 'any')  # 只要有一个数据为空，就进行行删除操作
# datas.describe().T  # 观察数据的多种统计指标(只能看数值型的)


## 创建一个时间函数格式化字符串
def date_format(dt):
    # dt显示是一个series/tuple；dt[0]是date，dt[1]是time
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


## 需求：构建时间和功率之间的映射关系，可以认为：特征属性为时间；目标属性为功率值。
# 获取x和y变量, 并将时间转换为数值型连续变量
X = datas.iloc[:,0:2]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)  # 按行将时间转换为数值型
Y = datas['Global_active_power']


## 对数据集进行测试集合训练集划分
# X：特征矩阵(类型一般是DataFrame)
# Y：特征对应的Label标签(类型一般是Series)
# test_size: 对X/Y进行划分的时候，测试集合的数据占比, 是一个(0,1)之间的float类型的值
# random_state: 数据分割是基于随机器进行分割的，该参数给定随机数种子；给一个值(int类型)的作用就是保证每次分割所产生的数数据集是完全相同的
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 查看数据形状
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)

## 数据标准化
# StandardScaler：将数据转换为标准差为1的数据集(有一个数据的映射)
ss = StandardScaler()  # 模型对象创建
X_train = ss.fit_transform(X_train)  # 训练模型并转换训练集
X_test = ss.transform(X_test)  # 直接使用在模型构建数据上进行一个数据标准化操作 (测试集)

"""
scikit-learn中：如果一个API名字有fit，那么就有模型训练的含义，没法返回值
scikit-learn中：如果一个API名字中有transform， 那么就表示对数据具有转换的含义操作
scikit-learn中：如果一个API名字中有predict，那么就表示进行数据预测，会有一个预测结果输出
scikit-learn中：如果一个API名字中既有fit又有transform的情况下，那就是两者的结合(先做fit，再做transform)
"""

## 模型训练
lr = LinearRegression(fit_intercept=True) # 模型对象构建
lr.fit(X_train, Y_train) ## 训练模型
## 模型校验
y_predict = lr.predict(X_test) ## 预测结果

print("训练集上R2:",lr.score(X_train, Y_train))
print("测试集上R2:",lr.score(X_test, Y_test))
mse = np.average((y_predict-Y_test)**2)
rmse = np.sqrt(mse)
print("rmse:",rmse)



## 预测值和实际值画图比较
t=np.arange(len(X_test))
plt.figure(facecolor='w')#建一个画布，facecolor是背景色
plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label='预测值')
plt.legend(loc = 'upper left')#显示图例，设置图例的位置
plt.title("线性回归预测时间和功率之间的关系", fontsize=20)
plt.grid(b=True) # 加网格
plt.show()

