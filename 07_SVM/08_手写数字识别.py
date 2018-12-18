import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics


# 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1.加载数字图片数据
digits = datasets.load_digits()

# 2.获取样本数量，并将图片数据格式化（要求所有图片的大小、像素点都是一致的 => 转换成为的向量大小是一致的）
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# 3.模型构建
classifier = svm.SVC(gamma=0.001)  # 默认是rbf
# 使用二分之一的数据进行模型训练
# 取前一半数据训练，后一半数据测试
classifier.fit(data[:int(n_samples / 2)], digits.target[:int(n_samples / 2)])

# 4.测试数据部分实际值和预测值获取
# 后一半数据作为测试集
expected = digits.target[int(n_samples/2):]  # y_test
predicted = classifier.predict(data[int(n_samples / 2):])  # y_predicted
# 计算准确率
print("分类器%s的分类效果:\n%s\n" % (classifier, metrics.classification_report(expected, predicted)))
# 生成一个分类报告classification_report
print("混淆矩阵为:\n%s" % metrics.confusion_matrix(expected, predicted))
# 生成混淆矩阵
print("score_svm:\n%f" %classifier.score(data[int(n_samples / 2):], digits.target[int(n_samples / 2):]))

# 5.进行图片展示
plt.figure(facecolor='gray', figsize=(12,5))
# 先画出5个预测失败的
# 把预测错的值的 x值 y值 和y的预测值取出
images_and_predictions = list(zip(digits.images[int(n_samples / 2):][expected != predicted], expected[expected != predicted], predicted[expected != predicted]))
# 通过enumerate，分别拿出x值 y值 和y的预测值的前五个，并画图
for index,(image,expection, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')                          
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')  # 把cmap中的灰度值与image矩阵对应，并填充
    plt.title(u'预测值/实际值:%i/%i' % (prediction, expection))
# 再画出5个预测成功的
images_and_predictions = list(zip(digits.images[int(n_samples / 2):][expected == predicted], expected[expected == predicted], predicted[expected == predicted]))
for index, (image,expection, prediction) in enumerate(images_and_predictions[:5]):
    plt.subplot(2, 5, index + 6)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(u'预测值/实际值:%i/%i' % (prediction, expection))

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()