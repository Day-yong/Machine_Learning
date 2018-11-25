# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings  # 警告处理

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier  # 分类树
from sklearn.model_selection import train_test_split  # 数据划分
from sklearn.preprocessing import MinMaxScaler  # 数据归一化
from sklearn.feature_selection import SelectKBest  # 特征选择
from sklearn.feature_selection import chi2  # 卡方统计量
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.pipeline import Pipeline  # 管道
from sklearn.model_selection import GridSearchCV  # 网格搜索交叉验证，用于选择最优的参数


# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)


# 1.读取数据
path = '../datas/iris.data'
df = pd.read_csv(path, header=None)


# 2.划分数据
X = df[list(range(4))]  # 获取X变量
Y = pd.Categorical(df[4]).codes # 获取Y，并将其转换为1,2,3类型


# 3.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=14) # random_state随机数生成种子
# DecisionTreeClassifier是分类算法，要求Y必须是int类型
Y_train = Y_train.astype(np.int)
Y_test = Y_test.astype(np.int)


# 4.参数优化
pipe = Pipeline([
            ('mms', MinMaxScaler()),  # 归一化
            ('skb', SelectKBest(chi2)),  # 特征选择
            ('pca', PCA()),  # 降维
            ('decision', DecisionTreeClassifier(random_state=0))  # 决策树模型
        ])
# 参数
parameters = {
    "skb__k": [1,2,3,4],  # 卡方统计量
    "pca__n_components": [0.5,0.99],  # 设置为浮点数代表主成分方差所占最小比例的阈值
    "decision__criterion": ["gini", "entropy"],  # 基尼指数或信息增益
    "decision__max_depth": [1,2,3,4,5,6,7,8,9,10]  # 树的深度
}

# 5.模型构建：通过网格交叉验证，寻找最优参数列表， param_grid可选参数列表，cv：进行几折交叉验证
gscv = GridSearchCV(pipe, param_grid=parameters, cv=3)


# 6.模型训练
gscv.fit(X_train, Y_train)


# 7.模型评估
print("最优参数列表:", gscv.best_params_)
print("score值：",gscv.best_score_)
print("最优模型:", end='')
print(gscv.best_estimator_)



# 应用最优参数看效果
mms_best = MinMaxScaler()
skb_best = SelectKBest(chi2, k=3)
pca_best = PCA(n_components=0.99)
moedl = DecisionTreeClassifier(criterion='gini', max_depth=4)

X_train = pca_best.fit_transform(skb_best.fit_transform(mms_best.fit_transform(X_train), Y_train))
X_test = pca_best.transform(skb_best.transform(mms_best.transform(X_test)))
moedl.fit(X_train, Y_train)

print("正确率:", moedl.score(X_test, Y_test))