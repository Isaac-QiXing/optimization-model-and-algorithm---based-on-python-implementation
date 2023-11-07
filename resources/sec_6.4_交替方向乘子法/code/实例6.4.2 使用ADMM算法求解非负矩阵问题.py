#!/usr/bin/env python
# coding: utf-8

# 实例6.4.2 使用ADMM算法求解非负矩阵问题

# In[1]:


import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
np.set_printoptions(precision=4,suppress=True)


# In[2]:


p = 8
q = 6
r = 3
np.random.seed(1)
C = np.random.randint(3,size=(p,q))


# In[3]:


loss_list = []
X = np.zeros((p,q),np.float32)
V = np.zeros((p,r),np.float32)
W = np.ones((r,q),np.float32)
U = np.zeros((p,q),np.float32)


# In[4]:


maxiter = 30
tol = 1e-3
multiplier = 2
count = 0


# In[5]:


while count < maxiter:
    X1 = cp.Variable((p,q+r), nonneg=True) 
    obj1 = cp.square(cp.norm(X1[:,:q]-C,p='fro'))+(multiplier/2)*cp.square(cp.norm(X1[:,:q]-X1[:,q:]@W+U,p='fro'))
    prob1 = cp.Problem(cp.Minimize(obj1))
    prob1.solve(solver='CVXOPT',verbose=True)
    X = X1.value[:,:q]
    V = X1.value[:,q:]    
    X2 = cp.Variable((r,q), nonneg=True)
    obj2 = cp.square(cp.norm(X-V@X2+U,p='fro'))
    prob2 = cp.Problem(cp.Minimize(obj2))
    prob2.solve(solver='CVXOPT',verbose=True)
    W = X2.value    
    loss = np.linalg.norm(V@W-C)
    loss_list.append(loss)    
    if loss<= tol:
        break        
    U = U+X-V@W
    count += 1


# 代码第2行将变量X和V合并为一个矩阵变量，第3行定义目标函数， 第4-5行调用CVXOPT包求解子问题，更新变量X和V，第8-11行求解子问题，更新变量W，第17行更新乘子变量U。

# In[6]:


plt.plot(loss_list)
plt.ylabel('loss',fontsize=14)
plt.xlabel('steps',fontsize=14)


# In[7]:


print('矩阵C的非负矩阵分解结果为:C≈V@W')
print('C =','\n',C,)
print('V = ','\n',V)
print('W = ','\n',W)


# In[ ]:




