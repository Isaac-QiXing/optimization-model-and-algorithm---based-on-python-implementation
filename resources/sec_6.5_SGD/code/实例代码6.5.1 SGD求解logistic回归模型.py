#!/usr/bin/env python
# coding: utf-8

# 实例代码6.5.1 SGD求解logistic回归模型

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import math
import sklearn
import time


# In[2]:


# 生成分类数据集
from sklearn.datasets._samples_generator import make_classification

n_sample = 120
dim = 2 
X,label_v = make_classification(n_samples=n_sample,
            n_features = dim,n_redundant = 0,random_state=1,n_clusters_per_class=2)
# 在特征中加入噪声
rng = np.random.RandomState(0)
X += 0.5* rng.uniform(size = X.shape) 

print(X)
print(label_v)


# In[3]:


# 标准化处理 
x_mean = np.mean(X, 0)
x_std = np.std(X, 0)
X_arr_scale = (X - x_mean) / x_std
# 截距
X_arr_scale = np.append(np.ones((n_sample, 1)), X_arr_scale, 1)
X_arr = np.append(np.ones((n_sample, 1)), X, 1)

#print(X_arr_scale)
#print(X_arr)

# 绘制数据
plt.figure(1, figsize=(8, 6))
for k in range(n_sample):
    if label_v[k] > 0:
        plt.scatter(X_arr[k][1], X_arr[k][2],c='b')
    else:
        plt.scatter(X_arr[k][1], X_arr[k][2], c='m',marker='x')
plt.xlabel('x1', fontsize='medium'); 
plt.ylabel('x2', fontsize='medium')
plt.show()


# In[4]:


def sigmoid(z):   
    return 1.0 / (1.0 + np.exp(-z))


def cost(xMat,weights,yMat):
    """
    计算损失函数
    xMat: 特征数据-矩阵
    weights: 参数
    yMat: 标签数据-矩阵
    return: 损失函数
    """
    m, n = xMat.shape
    hypothesis = sigmoid(np.dot(xMat, weights))  # 预测值
    cost = (-1.0 / m) * np.sum(yMat * np.log(hypothesis)             + (1 - yMat)* np.log(1 - hypothesis))  # 损失函数
    return cost
    

def obj_logistic(X, y, beta_v):
    # 目标函数: 对数似然函数 的负数
    # 输入:
    # X: 特征值矩阵, n_sample * dim
    # y: 标签向量
    # beta_v: 系数向量
    # 输出: 
    #   对数似然函数值    
    p_v = sigmoid(X.dot(beta_v))
    log_likelihood =   np.dot(y,np.log(p_v) )                       + np.dot(1-y,np.log(1 - p_v) )
    return -1.0* log_likelihood   
    
def grad_logistic(X, y, beta_v):
    # 计算对数似然函数的梯度
    n_sample = y.size
    p_v = sigmoid(X.dot(beta_v))
    return   -(1/n_sample) * (y-p_v ) @ X 

def SGD(X,y, alpha=0.1, maxepochs=10000,epsilon=1e-4):
    starttime = time.time()
    xMat = np.mat(X)
    yMat = np.mat(label_v)
    m, n = xMat.shape
    weights = np.ones((n, 1))  # 模型参数
    epochs_count = 0
    loss_list = []
    epochs_list = []
    while epochs_count < maxepochs:
        rand_i = np.random.randint(m)  # 随机取一个样本
        loss = cost(xMat,weights,yMat) #前一次迭代的损失值
        hypothesis = sigmoid(np.dot(xMat[rand_i,:],weights)) #预测值
        error = hypothesis -yMat.T[rand_i,:] #预测值与实际值误差
        grad = np.dot(xMat[rand_i,:].T,error) #损失函数的梯度
        weights = weights - alpha*grad #参数更新
        loss_new = cost(xMat,weights,yMat)#当前迭代的损失值
        if abs(loss_new-loss)<epsilon:
            break
        loss_list.append(loss_new)
        epochs_list.append(epochs_count)
        epochs_count += 1
    print(loss_new)
    print("随机梯度下降算法耗时：", time.time() - starttime)
    print('迭代到第{}次，结束迭代！'.format(epochs_count))
    plt.plot(epochs_list,loss_list)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    return weights
    


# In[5]:


weights= SGD(X,label_v, alpha=0.1, maxepochs=10000, epsilon=1e-4)


# In[ ]:




