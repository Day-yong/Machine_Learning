## 二分类——Logistic回归

### 1、乳腺癌分类模型训练

- 01_乳腺癌分类.py
- 训练模型，并保存模型

### 2、乳腺癌分类加载模型预测

- 02_乳腺癌分类.py
- 加载模型，并进行预测

### 3、信贷审批

- 03_信贷审.py
- 0表示未通过，1表示通过

## 多分类——Softmax

### 1、葡萄酒质量预测

- 04_葡萄酒质量预测.py
- 回归算法目录下：[10_葡萄酒质量预测.py](https://github.com/Daycym/Machine-Learning/tree/master/01_%E5%9B%9E%E5%BD%92%E7%AE%97%E6%B3%95)
- 分别使用了线性回归模型、LASSO模型、Ridge模型、Elasitc Net模型对数据进行了训练预测
- 结果并不是很好，因为葡萄酒质量不是连续的值，只有3、4、5、6、7、8、9七种
- 本例使用Softmax实现多分类，来预测，效果稍有提升


### 2、鸢尾花数据分类

- 05_鸢尾花数据分类.py
- 数据有三类，LogisticRegressionCV中参数multi_class='multinomial'，
- 画出ROC/AUC
- 画出预测图