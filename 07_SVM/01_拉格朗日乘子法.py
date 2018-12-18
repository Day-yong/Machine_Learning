import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D


# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 拉格朗日乘子法理解
def f(x, y):
    return x**2 / 2.0 + y**2 / 3.0 - 1

# 构建数据
X1 = np.arange(-8, 8, 0.2)
X2 = np.arange(-8, 8, 0.2)
X1, X2 = np.meshgrid(X1, X2)
Y = np.array(list(map(lambda t: f(t[0], t[1]), zip(X1.flatten(), X2.flatten()))))
Y.shape = X1.shape

# 限制条件
X3 = np.arange(-4, 4, 0.2)
X3.shape = 1,-1
X4 = np.array(list(map(lambda t: t ** 2 -  t +1, X3)))

# 画图
fig = plt.figure(facecolor='w')
ax = Axes3D(fig)
ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap=plt.cm.jet)
ax.plot(X3, X4 , 'ro--', linewidth=2)


ax.set_title(u'函数$y=0.6 * (\\theta_1 + \\theta_2)^2 - \\theta_1 * \\theta_2$')
ax.set_xlabel('x')
ax.set_zlabel('z')
ax.set_ylabel('y')
plt.show()