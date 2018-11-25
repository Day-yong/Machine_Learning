import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection  import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection  import GridSearchCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning


def notEmpty(s):
    return s != ''


mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False
## 拦截异常
warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)


names = ['CRIM','ZN', 'INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
path = "../datas/boston_housing.data"
## 由于数据文件格式不统一，所以读取的时候，先按照一行一个字段属性读取数据，然后再按照每行数据进行处理
fd = pd.read_csv(path, header=None)
data = np.empty((len(fd), 14))
for i, d in enumerate(fd.values):
    d = map(float, filter(notEmpty, d[0].split(' ')))
    data[i] = list(d)

x, y = np.split(data, (13,), axis=1)
y = y.ravel()

print ("样本数据量:%d, 特征个数：%d" % x.shape)
print ("target样本数据量:%d" % y.shape[0])

#数据的分割，
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=14)
x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))


#参数优化
pipes = [
    Pipeline([
            ('mms', MinMaxScaler()), ## 归一化操作
            ('pca', PCA()), ## 降纬
            ('decision', DecisionTreeRegressor(criterion='mse'))
        ]),
    Pipeline([
            ('mms', MinMaxScaler()),
            ('decision', DecisionTreeRegressor(criterion='mse'))
        ]),
    Pipeline([
            ('decision', DecisionTreeRegressor(criterion='mse'))
        ])
]

# 参数
parameters = [
    {
    "pca__n_components": [0.25,0.5,0.75,1],
    "decision__max_depth":  np.linspace(1,20,20).astype(np.int8)
    },
    {
    "decision__max_depth":  np.linspace(1,20,20).astype(np.int8)
    },
    {
    "decision__max_depth":  np.linspace(1,20,20).astype(np.int8)
    }
]
#获取数据
x_train2, x_test2, y_train2, y_test2 = x_train1, x_test1, y_train1, y_test1

for t in range(3):
    pipe = pipes[t]

    gscv = GridSearchCV(pipe, param_grid=parameters[t])

    gscv.fit(x_train2, y_train2)
    
    print (t,"score值:",gscv.best_score_,"最优参数列表:", gscv.best_params_)




