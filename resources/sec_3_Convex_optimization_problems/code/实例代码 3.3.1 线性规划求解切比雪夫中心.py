#!/usr/bin/env python
# coding: utf-8

# 实例代码 3.3.1
# 线性规划求解切比雪夫中心
# 
# 多面体$P$:
# $$ \left[\begin{array} {cccc}
# 1&1\\
# 1 & -1\\
# -1&0
# \end{array} \right] \cdot 
# \left[\begin{array} {cccc}
# y\\
# x\\
# \end{array} \right] \preceq
# \left[\begin{array} {cccc}
# 1\\
# 1\\
# 0
# \end{array} \right] $$
# 

# In[3]:


from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt

c = np.array([1,0,0])
A = np.array([[np.sqrt(2),1,1],[np.sqrt(2),1,1],[1,-1,0]])
b = np.array([1,1,0])

res = optimize.linprog(-c,A,b)
res.x


# In[4]:


plt.plot([-1,0],[0,1],'b')
plt.plot([0,1],[1,0],'b')
plt.plot([-1,1],[0,0],'b')
r,x,y = res.x[0],res.x[2],res.x[1]
theta = np.arange(0,2*np.pi,0.01)
plt.plot(x+r*np.cos(theta),y+r*np.sin(theta),'r')
plt.plot(x,y,'k.')
plt.axis('equal')


# In[ ]:




