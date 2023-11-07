#!/usr/bin/env python
# coding: utf-8

# 实例6.3.1   迫近梯度法求解Lasso 问题

# In[1]:


import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
np.set_printoptions(precision=4,suppress=True)


# In[2]:


m = 500
n = 10

np.random.seed(1)
A = np.random.randn(m,n)
np.random.seed(2)
x_index = np.random.choice(n,4,replace=False)
x = np.zeros((n,1))
x[x_index] = 1
y = A @ x
np.random.seed(3)
y += np.random.normal(0,0.1,y.shape)


# In[3]:


def soft_thresold(z, t):
    # 软阈值算子
    assert t>0
    x = np.zeros(z.shape)
    x[z>t] = z[z>t]-t
    x[z<-t] = z[z<-t]+t
    return x


# In[4]:


x_old = np.random.randn(n,1)
eig, _ = np.linalg.eig(A.T @ A)
max_eig = np.linalg.norm(eig[0])
t = 1/max_eig
lambdas = [0.001,0.005,0.01,0.05,0.1]
record = {}
for lam in lambdas:
    count = 0
    while count<1000:
        z = x_old - t *(A.T @ A @ x_old - A.T @ y)
        x_new = soft_thresold(z, t*lam)
        err = np.linalg.norm(x_old - x_new)
        x_old= x_new
        if err<1e-4:
            break
    record[lam] = x_old


# 第1-16行代码展示了ISTA算法求解Lasso问题的迭代过程，其中，函数$F(x) = \frac{1}{2}\|Ax-y\|_2^2$, $\nabla F(x) = A^{\top}(Ax-y)$，正则化参数$\lambda$ 取了一系列不同的值.

# In[5]:


print('真值为:',x.reshape(n),'\n')
for key,value in record.items():
    print('lambda:',key,'\n','求解结果:',value.reshape(n))

