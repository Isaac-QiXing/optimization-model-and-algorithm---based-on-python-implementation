#!/usr/bin/env python
# coding: utf-8

# 实例代码3.4.1 二次规划求多面体间的距离
# 
# 多面体$P_1$:
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
# 多面体$P_2$:
# $$ \left[\begin{array} {cccc}
# 0&-1\\
# -1 & 1\\
# 1&1
# \end{array} \right] \cdot 
# \left[\begin{array} {cccc}
# y\\
# x\\
# \end{array} \right] \preceq
# \left[\begin{array} {cccc}
# -2\\
# 2\\
# 4
# \end{array} \right] $$
# 
# 

# In[3]:


from cvxopt import solvers, matrix
import numpy as np
from matplotlib import pyplot as plt

P = matrix(2*np.array([[1.0,0.0,-1.0,0.0],[0.0,1.0,0.0,-1.0]]).T@np.array([[1.0,0.0,-1.0,0.0],[0.0,1.0,0.0,-1.0]]))
q = matrix(np.zeros(4,float))
constraint1 = np.array([[1.0,1.0],[1.0,-1.0],[-1.0,0.0]])
constraint2 = np.array([[0.0,-1.0],[-1.0,1.0],[1.0,1.0]])
G1 = np.hstack((constraint1,np.zeros((3,2),float)))
G2 = np.hstack((np.zeros((3,2),float),constraint2))
G = matrix(np.vstack((G1,G2)))
h = matrix(np.array([1.0,1.0,0.0,-2.0,2.0,4.0]))
sol = solvers.qp(P,q,G,h)
print(sol['x'])


# In[4]:


plt.plot([-1,0],[0,1],'b')
plt.plot([0,1],[1,0],'b')
plt.plot([-1,1],[0,0],'b')
plt.plot([2,2],[0,2],'y')
plt.plot([2,3],[0,1],'y')
plt.plot([2,3],[2,1],'y')
plt.plot([sol['x'][1],sol['x'][3]],[sol['x'][0],sol['x'][2]],'r--')
plt.axis('equal')


# In[ ]:




