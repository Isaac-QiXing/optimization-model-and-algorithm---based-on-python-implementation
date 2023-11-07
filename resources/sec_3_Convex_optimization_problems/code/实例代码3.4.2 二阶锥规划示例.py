#!/usr/bin/env python
# coding: utf-8

# 实例代码3.4.2 二阶锥规划示例
# 
# $$\begin{array}{ll}
# \operatorname{minimize} & -2 x_{1}+x_{2}+5 x_{3} \\
# \text { subject to } & \left\|\left[\begin{array}{c}
# -13 x_{1}+3 x_{2}+5 x_{3}-3 \\
# -12 x_{1}+12 x_{2}-6 x_{3}-2
# \end{array}\right]\right\|_{2} \leq-12 x_{1}-6 x_{2}+5 x_{3}-12 \\
# & \left\|\left[\begin{array}{c}
# -3 x_{1}+6 x_{2}+2 x_{3} \\
# x_{1}+9 x_{2}+2 x_{3}+3 \\
# -x_{1}-19 x_{2}+3 x_{3}-42
# \end{array}\right]\right\|_{2} \leq-3 x_{1}+6 x_{2}-10 x_{3}+27 .
# \end{array}$$

# In[1]:


from cvxopt import matrix, solvers
c = matrix([-2., 1., 5.])
G = [ matrix( [[12., 13., 12.], [6., -3., -12.], [-5., -5., 6.]] ) ] 
G += [ matrix( [[3., 3., -1., 1.], [-6., -6., -9., 19.], [10., -2., -2., -3.]] ) ]
h = [ matrix( [-12., -3., -2.] ),  matrix( [27., 0., 3., -42.] ) ]
sol = solvers.socp(c, Gq = G, hq = h)
print(sol['x'])


# In[ ]:




