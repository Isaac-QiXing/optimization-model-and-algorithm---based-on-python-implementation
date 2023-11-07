# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:09:38 2021

@author: lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from Classifiers_v2 import *
import Classifiers2

#图片年龄数据与基准算法对比（学习率为power）
MAE_predictor_base_with_weights = []
MAE_predictor_base_with_average = []
MAE_predictor_base_with_sampling = []
learning_rate = 0.01
num_experiments = 10
strategy='power'
lr_collection=[]
for t in range(num_experiments):

    dataset = pd.read_excel("age data.xls")
    dataset = dataset.drop(['File_name'], axis = 1).sample(frac = 1)
    true_labels = np.array(dataset["truth lable"])
    X = np.array(dataset.drop(['truth lable'], axis = 1))
    num = 1
    X_extend = X.repeat(num, axis = 0)
    y_extend = true_labels.repeat(num)
    w = Classifiers2.Base_with_weights()
    s = Classifiers2.Base_with_sampling()
    a = Classifiers2.Base_with_average()
    w.training(X_extend, y_extend, learning_rate, strategy)
    s.training(X_extend, y_extend, learning_rate, strategy)
    a.training(X_extend, y_extend, learning_rate, strategy)
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

plt.ylim(0,8)
y = MAE_predictor_base_with_weights.mean(axis=0)[:iterations_displayed]
std = MAE_predictor_base_with_weights.std(axis=0)[:iterations_displayed]
#plt.plot(x,y, label = "Proposed", marker='x', markevery=200, markersize=8)
plt.plot(x,y, label = "Proposed")
plt.fill_between(x, y - std, y + std,alpha=0.2)

y = MAE_predictor_base_with_average.mean(axis=0)[:iterations_displayed]
std = MAE_predictor_base_with_average.std(axis=0)[:iterations_displayed]
#plt.plot(x,y, label = "Average", marker='o', markevery=200, markersize=8)
plt.plot(x,y, label = "Average")
plt.fill_between(x, y - std, y + std, alpha=0.2)

y = MAE_predictor_base_with_sampling.mean(axis=0)[:iterations_displayed]
std = MAE_predictor_base_with_sampling.std(axis=0)[:iterations_displayed]
#plt.plot(x,y, label = u"Sampling", marker='^', markevery=200, markersize=8)
plt.plot(x,y, label = u"Sampling")
plt.fill_between(x, y - std, y + std, alpha=0.2)
plt.legend()


# In[ ]:
'''

weight_collection = np.array(w.weight_collection)
for i in range(weight_collection.shape[-1]):
    plt.plot(weight_collection[:, i], label = "Worker{}".format(i + 1))
plt.legend()

'''
