# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:04:44 2021

@author: lenovo
"""
import pandas as pd
import numpy as np
import Classifiers2
import math
import matplotlib.pyplot as plt

#图片年龄数据在线聚合算法参数敏感性分析-MAE及STD
#学习率变化趋势曲线
o = 1/1000*math.sqrt(2*math.log(10)/10000)
d = 1/1000*math.sqrt(2*math.log(10)*2)
t=0.2
strategy = 'constant'
i=0
lr_collection=[]
loss=[]
num_experiments=10

for learning_rate in [0.0001, 0.001, 0.01, 0.1, o, d]:
    MAE_predictor_base_with_weights = []
    if learning_rate == d:
        strategy = 'doubling'
    elif learning_rate == t:
        strategy = 'power'
    else:
        strategy = 'constant'
    for t in range(num_experiments):
        dataset = pd.read_excel("age data.xls")
        dataset = dataset.drop(['File_name'], axis = 1).sample(frac = 1)
        true_labels = np.array(dataset["truth lable"])
        X = np.array(dataset.drop(['truth lable'], axis = 1))
        num = 10
        X_extend = X.repeat(num, axis = 0)
        y_extend = true_labels.repeat(num)
        w = Classifiers2.Base_with_weights()
        w.training(X_extend, y_extend, learning_rate, strategy)
        MAE_predictor_base_with_weights.append(w.MAE_collection)
        
    lr_collection.append(np.array(w.lr_collection))    
    lr=lr_collection[0]
    for i in range(len(lr_collection)-1):
        lr = np.vstack((lr, lr_collection[i+1]))
             
    result_predictor_base_with_weights = MAE_predictor_base_with_weights[0]
    for i in range(len(MAE_predictor_base_with_weights)-1):
        result_predictor_base_with_weights = np.vstack((result_predictor_base_with_weights, MAE_predictor_base_with_weights[i+1]))
    loss.append(result_predictor_base_with_weights)

strategy = 'power'
MAE_predictor_base_with_weights = []
for t in range(num_experiments):
    dataset = pd.read_excel("age data.xls")
    dataset = dataset.drop(['File_name'], axis = 1).sample(frac = 1)
    true_labels = np.array(dataset["truth lable"])
    X = np.array(dataset.drop(['truth lable'], axis = 1))
    num = 10
    X_extend = X.repeat(num, axis = 0)
    y_extend = true_labels.repeat(num)
    w = Classifiers2.Base_with_weights()
    w.training(X_extend, y_extend, learning_rate, strategy)
    MAE_predictor_base_with_weights.append(w.MAE_collection)
        
lr_collection.append(np.array(w.lr_collection))    
lr=lr_collection[0]
for i in range(len(lr_collection)-1):
    lr = np.vstack((lr, lr_collection[i+1]))
             
result_predictor_base_with_weights = MAE_predictor_base_with_weights[0]
for i in range(len(MAE_predictor_base_with_weights)-1):
    result_predictor_base_with_weights = np.vstack((result_predictor_base_with_weights, MAE_predictor_base_with_weights[i+1]))
loss.append(result_predictor_base_with_weights) 
print(list(map(lambda x: x.mean(axis = 0)[-1], loss)))
print(list(map(lambda x: x.std(axis = 0)[-1], loss)))

iterations_displayed = 1500 
x = range(iterations_displayed)

label=["0.0001","0.001","0.01","0.1","oracle","doubling trick","power"]
marker=['x','o','>','D','p','^','>']
for i in range(len(label)):
    result=lr[i]
    y=result[:iterations_displayed]
    plt.plot(x,y, label = label[i], marker=marker[i], markevery=200, markersize=8)
plt.ylim(0,0.015)
plt.xlabel("Number of samples")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid()
