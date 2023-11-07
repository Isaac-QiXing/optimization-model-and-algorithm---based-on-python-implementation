#!/usr/bin/env python
# coding: utf-8

# 实例6.2.1 坐标下降法求解线性方程组

# ## 坐标下降法求解线性方程组
# $$
#    Ax=b
# $$
# 等价于求解最小二乘问题 
# $$
#     \min_x  f(x) = \frac{1}{2}\|Ax-b\|^2
# $$

# In[1]:


import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
np.set_printoptions(precision=4,suppress=True)


# In[2]:


def coordinate_descent(A,b,maxiter=1000,tol=1e-4):
    assert (type(A),type(b)) == (np.ndarray,np.ndarray),'请输入numpy数组类型'    
    m,n = A.shape
    if len(b)!=n:
        raise ValueError('维度不匹配')
    if b.ndim == 1:
        b = b.reshape(-1,1)        
    x = np.random.randn(n,1)
    count = 0 
    while count < maxiter:
        x_old = x.copy()
        for i in range(n):
            c = A[:,i].reshape(-1,1)
            sol = c.T@(b-A@x+x[i]*c)/(c.T.dot(c))
            x[i,0] = sol[0,0]
        count += 1
        if np.linalg.norm(x_old-x)<tol:
            break
    return x,count


# 代码第13-15行更新解$x$的第$i$个分量. 记当前迭代点为$\bar{x}$, 记一元函数 
# $$
#     \phi(x_i) = f(\bar{x}_1,\cdots,\bar{x}_{i-1}, x_i, \bar{x}_{i+1},\cdots,\bar{x}_n),
# $$
# 则 $\phi'(x_i)=a_i^{\top}(A\bar{x}-b-a_i\bar{x}_i + a_i x_i)$, 其中$a_i$ 为矩阵$A$的第$i$列. 令$\phi'(x_i) =0$, 可得凸优化问题 $\min_x \phi(x_i)$ 的最优解 
# $x_i^* = \frac{a_i^{\top}(b-A\bar{x}+a_i \bar{x}_i)}{\|a_i\|^2}
# $.
# 

# In[3]:


A = np.array([[1,1,1],[0,1,1],[0,0,1]])
b = np.array([3,2,1])
solution,_ = coordinate_descent(A,b)


# In[ ]:




