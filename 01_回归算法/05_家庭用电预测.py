# -*- coding: utf-8 -*-

"""
时间与电压的多项式关系
"""
from sklearn.model_selection import train_test_split  # 划分数据的类
from sklearn.linear_model import LinearRegression  # 线性回归的类
from sklearn.preprocessing import StandardScaler  # 数据标准化的类
from sklearn.preprocessing import PolynomialFeatures  # 多项式生成的类
from sklearn.pipeline import Pipeline  # 管道的类

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time

# 创建一个时间字符串格式化
def date_format(dt):
    t = time.strptime(' '.join(dt), '%d/%m/%Y %H:%M:%S')
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1.加载数据
# path = 'datas/household_power_consumption_200.txt' # 200行数据
path = '../datas/household_power_consumption_1000.txt' # 1000行数据
df = pd.read_csv(path, sep=';', low_memory=False)

# 日期、时间、有功功率、无功功率、电压、电流、厨房用电功率、洗衣服用电功率、热水器用电功率
# print(df.columns)  # 查看数据的列名，结果为下面的names

# 定义columns
names=['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']


# 2.异常数据处理(异常数据过滤)
new_df = df.replace('?', np.nan)  # 替换所有非法字符"?"为np.nan
datas = new_df.dropna(axis=0,how = 'any') # 只要有数据为空，就进行 行删除操作


# 3.管道构建
# Pipeline：管道的意思，将多个操作合并成为一个操作
# Pipleline总可以给定多个不同的操作，给定每个不同操作的名称即可，执行的时候，按照从前到后的顺序执行
# Pipleline对象在执行的过程中，当调用某个方法的时候，会调用对应过程的对应对象的对应方法
models = [
    Pipeline([
            ('Poly', PolynomialFeatures()), # 给定进行多项式扩展操作， 第一个操作：多项式扩展
            ('Linear', LinearRegression(fit_intercept=False)) # 第二个操作，线性回归
        ])
]
model = models[0]


# 4.获取数据
# 获取x和y变量, 并将时间转换为数值型连续变量
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]]

# 5.数据划分
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 6.数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # 训练并转换
X_test = scaler.transform(X_test) ## 直接使用在模型构建数据上进行一个数据标准化操作


# 7.模型训练，画图
# 横坐标
t = np.arange(len(X_test)) 

# 1,2,3,4阶
N = 5
d_pool = np.arange(1,N,1)  # 1,2,3,4

# 颜色数组
m = d_pool.size
clrs = [] 
for c in np.linspace(16711680, 255, m): # 4种颜色
    clrs.append('#%06x' % int(c))

# 线宽
line_width = 3 

plt.figure(figsize=(12,6), facecolor='w') # 创建一个绘图窗口，设置大小，设置颜色
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据索引和数据
for i,d in enumerate(d_pool):  # 0,1 1,2 2,3 3,4
    plt.subplot(N-1,1,i+1)  # 在当前图中添加子图，N-1行，1列，i+1顺序从上到下(1阶，2阶，3阶，4阶)
    plt.plot(t, Y_test, 'r-', label = u'真实值', ms = 10, zorder = 5)  # 每个子图上都先画出真实值，zorder = N表示覆盖，大的在上
    # 设置管道对象中的参数值，Poly是在管道对象中定义的操作名称， 后面跟参数名称；中间是两个下划线
    model.set_params(Poly__degree = d)  # 设置多项式的阶乘
    model.fit(X_train, Y_train) # 模型训练
    # Linear是管道中定义的操作名称
    # 获取线性回归算法模型对象
    lin = model.get_params()['Linear']
    output = u'%d阶，系数为：' % d
    # 判断lin对象中是否有对应的属性
    if hasattr(lin, 'alpha_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'alpha=%.6f, ' % lin.alpha_) + output[idx:]
    if hasattr(lin, 'l1_ratio_'):
        idx = output.find(u'系数')
        output = output[:idx] + (u'l1_ratio=%.6f, ' % lin.l1_ratio_) + output[idx:]
    print (output, lin.coef_.ravel())
    
    # 模型结果预测
    y_hat = model.predict(X_test)

    # 计算评估值
    s = model.score(X_test, Y_test)
    
    # 画图
    label = u'%d阶, 准确率=%.3f' % (d,s)
    plt.plot(t, y_hat, color=clrs[i], lw=line_width, alpha = 0.75, label = label, zorder = 0)
    plt.legend(loc = 'upper left')
    plt.grid(True)
    plt.ylabel(u'%d阶结果' % d, fontsize=12)

## 预测值和实际值画图比较
plt.suptitle(u"线性回归预测时间和功率之间的多项式关系", fontsize=20)
plt.grid(b=True)
plt.show()