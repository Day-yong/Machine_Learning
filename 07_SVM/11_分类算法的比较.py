import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

from sklearn import  svm  # svm模型
from sklearn.model_selection import train_test_split  # 数据分割
from sklearn.preprocessing import StandardScaler  # 数据标准化
from sklearn.datasets import make_moons, make_circles, make_classification  # 数据集
from sklearn.neighbors import KNeighborsClassifier  # KNN模型
from sklearn.tree import DecisionTreeClassifier  # 决策树模型
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier  # 集成模型
from sklearn.linear_model import LogisticRegressionCV  # 逻辑回归


# 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1.构造数据集
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.4, random_state=1),
            linearly_separable
            ]

# 2.建模环节，用list把所有算法装起来
names = ["Nearest Neighbors", "Logistic","Decision Tree", "Random Forest", "AdaBoost", "GBDT","svm"]
classifiers = [
    KNeighborsClassifier(3),
    LogisticRegressionCV(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(n_estimators=10,learning_rate=1.5),
    GradientBoostingClassifier(n_estimators=10, learning_rate=1.5),
    svm.SVC(C=1, kernel='rbf')
    ]


# 3.模型训练及画图
figure = plt.figure(figsize=(27, 9), facecolor='w')
i = 1
h = .02  # 步长

for ds in datasets:
    X, y = ds
    X = StandardScaler().fit_transform(X)  # 数据标准化
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)  # 数据分割

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['r', 'b', 'y'])
    
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # 画每个算法的图
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        # hasattr是判定某个模型中，有没有哪个参数，
        # 判断clf模型中，有没有decision_function
        # np.c_让内部数据按列合并
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=25, horizontalalignment='right')
        i += 1

# 展示图
figure.subplots_adjust(left=.02, right=.98)
plt.show()
