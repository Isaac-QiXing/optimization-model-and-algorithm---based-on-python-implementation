#!/usr/bin/env python
# coding: utf-8

# 应用案例5.10.1   差分进化算法求解无约束二次优化问题

# In[1]:


import numpy as np
import random
class Population(object):
    # 定义种群类
    def __init__(self, min_range, max_range,                  dim, factor, rounds, size, object_func, CR=0.75):
        self.min_range = min_range
        self.max_range = max_range
        self.dimension = dim
        self.factor = factor
        self.rounds = rounds
        self.size = size
        self.cur_round = 1
        self.CR = CR
        self.get_object_function_value = object_func
        # 初始化种群
        self.individuality = [np.array([random.uniform(self.min_range,                               self.max_range) for s in range(self.dimension)])
                              for tmp in range(size)]
        self.object_function_values = [self.get_object_function_value(v)                                       for v in self.individuality]
        self.mutant = None
    def mutate(self):
    # 变异算子 
        self.mutant = []
        for i in range(self.size):
            select_range = [x for x in range(self.size)]
            select_range.remove(i)
            r0, r1, r2 = np.random.choice(select_range, 3, replace=False)
            tmp = self.individuality[r0]             + (self.individuality[r1] - self.individuality[r2]) * self.factor
            for t in range(self.dimension):
                if tmp[t] > self.max_range or tmp[t] < self.min_range:
                    tmp[t] = random.uniform(self.min_range, self.max_range)
            self.mutant.append(tmp)
    def crossover_and_select(self):
    # 交叉和选择算子
        for i in range(self.size):
            Jrand = random.randint(0, self.dimension)
            for j in range(self.dimension):
                if random.random() > self.CR and j != Jrand:
                    self.mutant[i][j] = self.individuality[i][j]
                tmp = self.get_object_function_value(self.mutant[i])
                if tmp < self.object_function_values[i]:
                    self.individuality[i] = self.mutant[i]
                    self.object_function_values[i] = tmp
    def print_best(self):
        m = min(self.object_function_values)
        i = self.object_function_values.index(m)
        print("轮数：" + str(self.cur_round))
        print("最佳个体：" + str(self.individuality[i]))
        print("目标函数值：" + str(m))
    def evolution(self):
        while self.cur_round < self.rounds:
            self.mutate()
            self.crossover_and_select()
            self.print_best()
            self.cur_round = self.cur_round + 1
if __name__ == "__main__":
    def f(v):
        return  v[0]**2+v[1]**2
    p = Population(min_range=-513, max_range=513, dim=2,                    factor=0.8, rounds=100, size=100, object_func=f)
    p.evolution()


# In[ ]:




