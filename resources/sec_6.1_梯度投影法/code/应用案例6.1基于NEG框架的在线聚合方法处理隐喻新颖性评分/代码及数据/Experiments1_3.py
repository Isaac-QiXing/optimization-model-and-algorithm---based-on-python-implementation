# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:04:44 2021

@author: lenovo
"""
import numpy as np
import Classifiers1
import math

#人工合成数据在线聚合算法参数敏感性分析（含power）-MAE及STD
base_learners_collection = [2,3,3,2]
true_labels = np.random.randint(0,255,10000)
variance= {'0': [0.5, 0.7, 0.9], '1': [3, 4, 5], '2': [7, 8, 9], '3': [10, 11, 12]}
o = 1/100*math.sqrt(2*math.log(10)/10000)
d = 1/100*math.sqrt(2*math.log(10)*2)
t=0.01
strategy = 'constant'
i=0
lr_collction=[]
loss=[]
num_experiments=10
for learning_rate in [0.0001, 0.001, 0.01, 0.1, o, d, t]:
    MAE_predictor_base_with_weights = []
    if learning_rate == d:
        strategy = 'doubling'
    elif learning_rate == t:
        strategy = 'power'
    else:
        strategy = 'constant' 
    for t in range(num_experiments):
        predictor_base_with_stepsize = Classifiers1.Base_with_stepsize(base_learners_collection = base_learners_collection, variance_dict = variance)
        predictor_base_with_stepsize.training(true_labels, learning_rate, strategy)
        MAE_predictor_base_with_weights.append(np.array(predictor_base_with_stepsize.MAE_collection))
    
    lr_collction.append(np.array(predictor_base_with_stepsize.lr_collection))
    lr=lr_collction[0]
    for i in range(len(lr_collction)-1):
        lr = np.vstack((lr, lr_collction[i+1]))
             
    result_predictor_base_with_weights = MAE_predictor_base_with_weights[0]
    for i in range(len(MAE_predictor_base_with_weights)-1):
        result_predictor_base_with_weights = np.vstack((result_predictor_base_with_weights, MAE_predictor_base_with_weights[i+1]))
    loss.append(result_predictor_base_with_weights)
  
print(list(map(lambda x: x.mean(axis = 0)[-1], loss)))
print(list(map(lambda x: x.std(axis = 0)[-1], loss)))