## 聚类算法

### 一、`K-Means` 算法

#### 1、案例1: 手写实现 `K-means` 算法

- 本案例按照算法流程，使用欧氏距离实现 `K-means` 算法，并通过文本数据进行训练，具体内容可见代码注释
- 01_K-means算法实现.py


#### 2、案例2： `K-means` 算法应用1

- 本案例使用 `sklearn` 库中自带的 `KMeans` 模型对模拟产生的数据进行聚类
- 02_K-means算法应用1.py


#### 3、案例3：`K-means` 算法应用2

- 本案例实现不同数据分布对 `KMeans` 聚类的影响，主要有旋转后的数据、簇具有不同方差的数据、簇具有不同数目的数据
- 03_K-means聚类应用2.py


### 二、`Mini Batch KMeans` 算法

#### 4、K-Means算法和Mini Batch K-Means算法比较

- 本案例基于`scikit` 包中的创建模拟数据的 `API` 创建聚类数据，使用 `K-means` 算法和 `Mini Batch K-Means` 算法对数据进行分类操作，比较这两种算法的聚类效果以及聚类的消耗时间长度
- 04_K-means与Mini Batch K-means算法比较.py


#### 5、K-means与Mini Batch K-means算法效果评估

- 05_K-means与Mini Batch K-means算法效果评估.py