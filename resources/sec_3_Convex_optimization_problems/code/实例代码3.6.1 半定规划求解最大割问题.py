#!/usr/bin/env python
# coding: utf-8

# 实例代码3.6.1 半定规划求解最大割问题

# In[1]:


import networkx as nx
import numpy as np
# 邻接矩阵
w = np.array([[0,3,2,4,1],
             [3,0,3,4,1],
             [2,3,0,1,1],
             [4,4,1,0,2],
             [1,1,1,2,0]])
G=nx.Graph(w)
position = nx.circular_layout(G)
nx.draw_networkx_nodes(G,position, node_color="r")
nx.draw_networkx_edges(G,position)
nx.draw_networkx_labels(G,position) 


# In[2]:


import cvxpy as cp

C = w
X = cp.Variable((5,5), symmetric=True)
# 算子>> 表示矩阵不等式.
constraints = [X >> 0]
constraints += [cp.diag(X)==np.ones(5)]
prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),constraints)
prob.solve(solver=cp.CVXOPT)
print("最优值是：", prob.value)
print("最优解X 是：")
print(X.value)


# 从结果可以看出节点0,1,4应该分为一类，节点2,3分到另一类

# 
