from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers = []
colors = []
fr = open('../datas/SVMtestSet.txt')
for line in fr.readlines():
    lineSplit = line.strip().split('\t')
    xPt = float(lineSplit[0])
    yPt = float(lineSplit[1])
    label = int(lineSplit[2])
    if (label == -1):
        xcord0.append(xPt)
        ycord0.append(yPt)
    else:
        xcord1.append(xPt)
        ycord1.append(yPt)

fr.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0, ycord0, marker='s', s=90)
ax.scatter(xcord1, ycord1, marker='o', s=50, c='red')
plt.title('Support Vectors Circled')
circle = Circle((4.658191, 3.507396), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
circle = Circle((3.457096, -0.082216), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8),linewidth=3, alpha=0.5)
ax.add_patch(circle)
circle = Circle((5.286862, -2.358286), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3,alpha=0.5)
ax.add_patch(circle)
circle = Circle((6.080573, 0.418886), 0.5, facecolor='none', edgecolor=(0, 0.8, 0.8), linewidth=3,alpha=0.5)
ax.add_patch(circle)
# plt.plot([2.3,8.5], [-6,6]) #seperating hyperplane
b = -3.82407793
w0 = 0.81085367
w1 = -0.25395222
x = arange(-2.0, 12.0, 0.1)
y = (-w0 * x - b) / w1
ax.plot(x, y)
ax.axis([-2, 12, -8, 6])
plt.show()