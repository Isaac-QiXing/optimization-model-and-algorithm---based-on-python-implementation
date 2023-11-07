# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:04:44 2021

@author: lenovo
"""
import pandas as pd
import numpy as np
import Classifiers3
import math
import matplotlib.pyplot as plt

#隐喻新颖性评分数据在线聚合算法参数敏感性分析(含power)-MAE及STD
o = 1/9*math.sqrt(2*math.log(10)/3112)
d = 1/9*math.sqrt(2*math.log(10)*2)
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
        dataset1 = pd.read_csv("test.csv")
        dataset2 = pd.read_csv("train.csv")
        dataset3 = pd.read_csv("validation.csv")
        data=dataset1.append(dataset2)
        data=data.append(dataset3)
        data=data.loc[:,['True_Label','A1','A2','A3','A4','A5']].sample(frac=1)
        true_labels = np.array(data["True_Label"])
        X = np.array(data.drop(['True_Label'], axis = 1))
        w = Classifiers3.Base_with_weights()
        w.training(X, true_labels, learning_rate, strategy)
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
    dataset1 = pd.read_csv("test.csv")
    dataset2 = pd.read_csv("train.csv")
    dataset3 = pd.read_csv("validation.csv")
    data=dataset1.append(dataset2)
    data=data.append(dataset3)
    data=data.loc[:,['True_Label','A1','A2','A3','A4','A5']].sample(frac=1)
    true_labels = np.array(data["True_Label"])
    X = np.array(data.drop(['True_Label'], axis = 1))
    w = Classifiers3.Base_with_weights()
    w.training(X, true_labels, learning_rate, strategy)
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
marker=['x','o','>','D','p','^',"<"]
for i in range(len(label)):
    result=lr[i]
    y=result[:iterations_displayed]
    plt.plot(x,y, label = label[i], marker=marker[i], markevery=200, markersize=8)
plt.ylim(0,0.015)
plt.xlabel("Number of samples")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid()
