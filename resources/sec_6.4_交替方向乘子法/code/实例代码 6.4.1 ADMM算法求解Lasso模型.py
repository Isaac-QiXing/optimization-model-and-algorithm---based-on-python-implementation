#!/usr/bin/env python
# coding: utf-8

# 实例代码 6.4.1 ADMM算法求解Lasso模型

# In[1]:


#生成数据
import numpy as np
import random
ASize = (50, 100)
XSize = 100
A = np.random.normal(0, 1, ASize)
X = np.zeros(XSize)
e = np.random.normal(0, 0.1, 50)
XIndex = random.sample(list(range(XSize)), 5)  # 5 稀疏度
for xi in XIndex:
    X[xi] = np.random.randn()
b = np.dot(A, X) + e
#ADMM算法
import matplotlib.pyplot as plt
import numpy as np
XSize = 100
P_half = 0.01
c = 0.005
Xk = np.zeros(XSize)
Zk = np.zeros(XSize)
Vk = np.zeros(XSize)
X_opt_dst_steps = []
X_dst_steps = []
while True:
    Xk_new = np.dot(
        np.linalg.inv(np.dot(A.T, A) + c * np.eye(XSize, XSize)),
        c*Zk + Vk + np.dot(A.T, b)
    )
    # 软阈值算子
    Zk_new = np.zeros(XSize)
    for i in range(XSize):
        if Xk_new[i] - Vk[i] / c < - P_half / c:
            Zk_new[i] = Xk_new[i] - Vk[i] / c + P_half / c
        elif Xk_new[i] - Vk[i] / c > P_half / c:
            Zk_new[i] = Xk_new[i] - Vk[i] / c - P_half / c
    Vk_new = Vk + c * (Zk_new - Xk_new)
    X_dst_steps.append(np.linalg.norm(Xk_new - X, ord=2))
    X_opt_dst_steps.append(Xk_new)
    if np.linalg.norm(Xk_new - Xk, ord=2) < 1e-5:
        break
    else:
        Xk = Xk_new.copy()
        Zk = Zk_new.copy()
        Vk = Vk_new.copy()
X_opt = X_opt_dst_steps[-1]
for i, data in enumerate(X_opt_dst_steps):
    X_opt_dst_steps[i] = np.linalg.norm(data - X_opt, ord=2)
plt.title("Distance")
plt.plot(X_opt_dst_steps, label='X-opt-distance')
plt.plot(X_dst_steps, label='X-real-distance')
plt.legend()
plt.show()


# In[ ]:




