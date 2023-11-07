#!/usr/bin/env python
# coding: utf-8

# 实例代码 5.3.1 梯度下降法求解logistic 回归模型

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import math


# In[2]:


# 生成分类数据集
from sklearn.datasets._samples_generator import make_classification

n_sample = 120
dim = 2 
X,label_v = make_classification(n_samples=n_sample,            n_features = dim,n_redundant = 0,random_state=1,n_clusters_per_class=2)
# 在特征中加入噪声
rng = np.random.RandomState(0)
X += 0.5* rng.uniform(size = X.shape) 


# In[3]:


# 标准化处理 
x_mean = np.mean(X, 0)
x_std = np.std(X, 0)
X_arr_scale = (X - x_mean) / x_std
# 截距
X_arr_scale = np.append(np.ones((n_sample, 1)), X_arr_scale, 1)
X_arr = np.append(np.ones((n_sample, 1)), X, 1)


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


# In[6]:


def sigmoid(z):   
    return 1.0 / (1.0 + np.exp(-z))

def dev_sigmoid(z):
    # Sigmoid 函数的导函数
    return np.exp(-z) / (1.0 + np.exp(-z))^2
def sigmoid(z):   
    return 1.0 / (1.0 + np.exp(-z))


def grad_logistic(X, y, beta_v):
    # 计算对数似然函数的梯度
    n_sample = y.size
    p_v = sigmoid(X.dot(beta_v))
    return   -(1/n_sample) * (y-p_v ) @ X

def obj_logistic(X, y, beta_v):
    # 目标函数: 对数似然函数 的负数
    # 输入:
    # X: 特征值矩阵, n_sample * dim
    # y: 标签向量
    # beta_v: 系数向量
    # 输出: 
    #   对数似然函数值    
    p_v = sigmoid(X.dot(beta_v))
    log_likelihood =   np.dot(y,np.log(p_v) ) + np.dot(1-y,np.log(1 - p_v) )
    return -1.0* log_likelihood 
    
def line_search_armijo(xk,fun,grad_k,dk,alpha = 0.1):
# 按Armijo 规则进行线搜索
# 输入：
# xk: 当前迭代点
# fun: 目标函数
# grad_k: 当前迭代点处的梯度
# dk: 搜索方向 (下降方向)
# alpha : Armijo 搜索参数：f(xk) - f(x_(k+1)) >= alpha <dk,xk-x_(k+1)>
# 输出：alpha_k: 步长 
    maxIter = 50
    stepsize = 1.0 # 初始步长
    omega = 0.8 
    fxk = fun(xk)
    direc_dir = np.dot(-grad_k,dk) 
    
    if direc_dir <=0:
        error('搜索方向应为下降方向')
        
    for ii in range(maxIter):
        x2 = xk + stepsize * dk
        fx_2 = fun(x2)
        if fxk - fx_2 >= alpha * direc_dir * stepsize:
            break
        stepsize *=omega 
        
    return  stepsize
    

def gradient_descent(X, y, coef,stepsize=0.1,maxIter = 500):
    """ 梯度下降函数
    输入：
    X: 各样本的特征值, 每行表示一个样本，ndarray类型 
    y: 各样本的标签,ndarray类型 
    coef: 初始模型参数,ndarray类型 
    stepsize: 学习率
    maxIter：最大迭代次数
    
    输出1: 更新后的模型参数
    输出2： 1: 成功找到近似解, 前后两次迭代函数值小于容许下降量
            2： 成功找到近似解,目标函数迭代梯度值小于容许误差
            0：达到最大迭代次数
    输出3： 迭代次数
    输出4：各次迭代的目标函数值
    """
    
    tolFun=1E-6 #　 前后两次迭代函数值最小下降量
    tolGrad = 1E-5 #   目标函数迭代梯度值容许误差
    coef_new = coef.copy()
    sum_err = 0.0
    fobj_0 = 0 
    flag = 0
    fobj_v = np.zeros((maxIter,1))
    
    # 检查 stepsize 的变量类型  
    if type(stepsize)==int or type(stepsize)==float:
        type_stepsize = 0 
    elif type(stepsize)==str:
        if stepsize.lower()=='armijo':
            type_stepsize = 1
        else: 
            error('仅支持 Armijo 搜索准则和常数步长')
    
    obj_fun = lambda coef: obj_logistic(X, y, coef)
    
    for ii in range(maxIter):
        grad  = grad_logistic(X, y, coef_new)
        d = -1.0 * grad 
        
        if type_stepsize ==0: # 常数步长
            alpha = stepsize 
        else:
            alpha  =  line_search_armijo(coef_new,obj_fun,grad,d)
        
        coef_new  = coef_new + alpha * d 
            
        fobj = obj_logistic(X, y, coef_new)  
        fobj_v[ii]=fobj
        if ii > 0 and np.abs(fobj  - fobj_0)<= tolFun:
            flag = 1 
            break
        if np.linalg.norm(d,1) <=tolGrad:
            flag = 2
            break
        fobj_0 = fobj
        
      
    ite = ii+1
    return  coef_new,flag,ite,fobj_v

    


# In[7]:


#stepsize = 0.2
stepsize = 'armijo'
coef = np.array([0.01, 0.01, 0.01])  # 初始参数    
maxIter = 1000

# 梯度下降  
coef_new,flag,ite,fobj_v  = gradient_descent(X_arr,                            label_v,coef, stepsize,maxIter )        

# 绘制当前学习率下，每次迭代后的目标函数值
plt.figure(1, figsize=(8, 6))
plt.plot(fobj_v )    
plt.xlabel('iteration', fontsize='medium'); 
plt.ylabel('obj values', fontsize='medium')
plt.show()


# 绘制分类结果

# In[8]:


# 绘制数据
plt.figure(1, figsize=(8, 6))
for k in range(n_sample):
    if label_v[k] > 0:
        plt.scatter(X_arr[k][1], X_arr[k][2],c='b')
    else:
        plt.scatter(X_arr[k][1], X_arr[k][2], c='m',marker='x')

# 绘制分类函数：beta1*x1 + beta2*x2 + beta0 * 1  = 0.5\
#<=> x2 = - beta1/beta2 *x1  + (0.5- beta0)/beta2   
print(coef_new)
f = lambda t: -1.0*coef_new[1]/coef_new[2] * t + (0.5- coef_new[0])/coef_new[2] 

x = np.linspace(-2,2,100)
y = f(x)
plt.plot(x,y)

plt.xlabel('x1', fontsize='medium'); 
plt.ylabel('x2', fontsize='medium')
plt.show()


# In[ ]:




