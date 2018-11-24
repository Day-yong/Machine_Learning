# -*- coding: utf-8 -*-

"""
时间与电压的线性关系
"""

from sklearn.model_selection import train_test_split  # 划分数据的类
from sklearn.linear_model import LinearRegression  # 线性回归的类
from sklearn.preprocessing import StandardScaler  # 数据标准化的类

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

# 3.时间和电压之间的关系(Linear)
# 获取x和y变量, 并将时间转换为数值型连续变量
X = datas[names[0:2]]
X = X.apply(lambda x: pd.Series(date_format(x)), axis=1)
Y = datas[names[4]]

# 4.对数据集进行测试集合训练集划分 
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 5.数据标准化
ss = StandardScaler()
X_train = ss.fit_transform(X_train) # 训练并转换
X_test = ss.transform(X_test) ## 直接使用在模型构建数据上进行一个数据标准化操作 

# 6.模型训练
lr = LinearRegression()
lr.fit(X_train, Y_train) ## 训练模型

# 7.模型校验
y_predict = lr.predict(X_test) ## 预测结果

# 8.模型评估
print("准确率:",lr.score(X_test, Y_test))

# 9.预测值和实际值画图比较
t=np.arange(len(X_test))
plt.figure(facecolor='w')  # 设置背景色
plt.plot(t, Y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_predict, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc = 'lower right')
plt.title(u"线性回归预测时间和电压之间的关系", fontsize=20)
plt.grid(b=True) # 加网格
plt.show()