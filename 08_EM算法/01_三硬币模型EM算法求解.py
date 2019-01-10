# -*- coding:utf-8 -*-

import numpy as np
import math

class EM:
	# 初始化
    def __init__(self, prob):
        self.pro_A, self.pro_B, self.pro_C = prob
        
    # E步，即博客E步中红色公式
    def pmf(self, i):
        pro_1 = self.pro_A * math.pow(self.pro_B, data[i]) * math.pow((1-self.pro_B), 1-data[i])
        pro_2 = (1 - self.pro_A) * math.pow(self.pro_C, data[i]) * math.pow((1-self.pro_C), 1-data[i])
        return pro_1 / (pro_1 + pro_2)
    
    # M步
    def fit(self, data):
        count = len(data)
        print('init prob:{}, {}, {}'.format(self.pro_A, self.pro_B, self.pro_C))
        for d in range(count):
            _ = yield
            _pmf = [self.pmf(k) for k in range(count)]
            # M步中三个蓝色公式
            pro_A = 1/ count * sum(_pmf)
            pro_B = sum([_pmf[k]*data[k] for k in range(count)]) / sum([_pmf[k] for k in range(count)])
            pro_C = sum([(1-_pmf[k])*data[k] for k in range(count)]) / sum([(1-_pmf[k]) for k in range(count)])
            print('{}/{}  pro_a:{:.3f}, pro_b:{:.3f}, pro_c:{:.3f}'.format(d+1, count, pro_A, pro_B, pro_C))
            self.pro_A = pro_A
            self.pro_B = pro_B
            self.pro_C = pro_C


# 测试
data=[1,1,0,1,0,0,1,0,1,1]

em = EM(prob=[0.5, 0.5, 0.5])
f = em.fit(data)
next(f)

# 第一次迭代
f.send(1)

# 第二次
f.send(2)

# 改变初始化值
em = EM(prob=[0.4, 0.6, 0.7])
f2 = em.fit(data)
next(f2)

f2.send(1)

f2.send(2)