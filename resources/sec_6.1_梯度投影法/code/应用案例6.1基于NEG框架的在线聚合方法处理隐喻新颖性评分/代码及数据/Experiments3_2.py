# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:09:38 2021

@author: lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from Classifiers_v2 import *
import Classifiers3

#隐喻新颖性评分数据与基准算法对比及权重变化曲线

MAE_predictor_base_with_weights = []
MAE_predictor_base_with_average = []
MAE_predictor_base_with_sampling = []
learning_rate = 0.01
num_experiments = 10
strategy='power'
lr_collection=[]
for t in range(num_experiments):
    dataset1 = pd.read_csv("test.csv")
    dataset2 = pd.read_csv("train.csv")
    dataset3 = pd.read_csv("validation.csv")
    data=dataset1.append(dataset2)
    data=data.append(dataset3)
    data=data.loc[:,['True_Label','A1','A2','A3','A4','A5']].sample(frac=1)
    true_labels = np.array(data["True_Label"])
    X = np.array(data.drop(['True_Label'], axis = 1))
    w = Classifiers3.Base_with_weights()
    s = Classifiers3.Base_with_sampling()
    a = Classifiers3.Base_with_average()
    w.training(X, true_labels, learning_rate, strategy)
    s.training(X, true_labels, learning_rate, strategy)
    a.training(X, true_labels, learning_rate, strategy)
    lr_collection.append(np.array(w.lr_collection))
    MAE_predictor_base_with_weights.append(w.MAE_collection)
    MAE_predictor_base_with_average.append(a.MAE_collection)
    MAE_predictor_base_with_sampling.append(s.MAE_collection)
    
MAE_predictor_base_with_average = np.array(MAE_predictor_base_with_average)
MAE_predictor_base_with_weights = np.array(MAE_predictor_base_with_weights)
MAE_predictor_base_with_sampling = np.array(MAE_predictor_base_with_sampling)

iterations_displayed = MAE_predictor_base_with_average.shape[-1]
x = range(iterations_displayed)
plt.grid()

#plt.ylim(0,8)
y = MAE_predictor_base_with_weights.mean(axis=0)[:iterations_displayed]
std = MAE_predictor_base_with_weights.std(axis=0)[:iterations_displayed]
plt.plot(x,y, label = "Proposed", marker='x', markevery=500, markersize=8)
#plt.plot(x,y, label = "Proposed")
plt.fill_between(x, y - std, y + std,alpha=0.2)

y = MAE_predictor_base_with_average.mean(axis=0)[:iterations_displayed]
std = MAE_predictor_base_with_average.std(axis=0)[:iterations_displayed]
plt.plot(x,y, label = "Average", marker='o', markevery=500, markersize=8)
#plt.plot(x,y, label = "Average")
plt.fill_between(x, y - std, y + std, alpha=0.2)

y = MAE_predictor_base_with_sampling.mean(axis=0)[:iterations_displayed]
std = MAE_predictor_base_with_sampling.std(axis=0)[:iterations_displayed]
plt.plot(x,y, label = u"Sampling", marker='^', markevery=500, markersize=8)
#plt.plot(x,y, label = u"Sampling")
plt.fill_between(x, y - std, y + std, alpha=0.2)
plt.xlabel('Number of samples')
plt.ylabel('MAE')
plt.legend()
plt.show()

# In[ ]:

marker=['x','o','>','P','^']
weight_collection = np.array(w.weight_collection)
plt.grid()
for i in range(weight_collection.shape[-1]):
    plt.plot(weight_collection[:, i], label = "Worker{}".format(i + 1), marker=marker[i], markevery=500,markersize=8)
plt.xlabel('Number of samples')
plt.ylabel('Weights')
plt.legend()
plt.show()

