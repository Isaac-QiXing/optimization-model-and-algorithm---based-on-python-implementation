# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 08:03:58 2021

@author: lenovo
"""
import numpy as np
import matplotlib.pyplot as plt
import Classifiers1
#import math

#人工合成数据与基准算法的对比(学习率为power)
base_learners_collection = [2,3,3,2]
true_labels = np.random.randint(0,255,10000)
variance= {'0': [0.5, 0.7, 0.9], '1': [3, 4, 5], '2': [7, 8, 9], '3': [10, 11, 12]}
MAE_predictor_base_with_weights = []
MAE_predictor_base_with_average = []
MAE_predictor_base_with_sampling = []

num_experiments=10
learning_rate=0.2
strategy = 'power'
lr_collection=[]
for t in range(num_experiments):
    predictor_base_with_weights =  Classifiers1.Base_with_weights(base_learners_collection = base_learners_collection,
                                                                  variance_dict = variance)
    predictor_base_with_weights.training(true_labels, learning_rate, strategy)
    MAE_predictor_base_with_weights.append(np.array(predictor_base_with_weights.MAE_collection))
    lr_collection.append(np.array(predictor_base_with_weights.lr_collection))
    
    predictor_base_with_average =  Classifiers1.Base_with_average(base_learners_collection = base_learners_collection, variance_dict = variance)
    predictor_base_with_average.training(true_labels, learning_rate, strategy)
    MAE_predictor_base_with_average.append(np.array(predictor_base_with_average.MAE_collection))
    #lr_collection 
    predictor_base_with_sampling =  Classifiers1.Base_with_sampling(base_learners_collection = base_learners_collection, variance_dict = variance)
    predictor_base_with_sampling.training(true_labels, learning_rate, strategy)
    MAE_predictor_base_with_sampling.append(np.array(predictor_base_with_sampling.MAE_collection))
    lr_collection.append(np.array(predictor_base_with_sampling.lr_collection))
  
result_predictor_base_with_weights = MAE_predictor_base_with_weights[0]
result_predictor_base_with_average = MAE_predictor_base_with_average[0]
result_predictor_base_with_sampling = MAE_predictor_base_with_sampling[0]

for i in range(len(MAE_predictor_base_with_weights)-1):
    result_predictor_base_with_weights = np.vstack((result_predictor_base_with_weights, MAE_predictor_base_with_weights[i+1]))
    result_predictor_base_with_average = np.vstack((result_predictor_base_with_average, MAE_predictor_base_with_average[i+1]))
    result_predictor_base_with_sampling = np.vstack((result_predictor_base_with_sampling, MAE_predictor_base_with_sampling[i+1]))
  
iterations_displayed = 1500
x = range(iterations_displayed)

y = result_predictor_base_with_weights.mean(axis=0)[:iterations_displayed]
std = result_predictor_base_with_weights.std(axis=0)[:iterations_displayed]
plt.plot(x,y, label = "Proposed", marker='x', markevery=200, markersize=8)
plt.fill_between(x, y - std, y + std,alpha=0.2)

y = result_predictor_base_with_average.mean(axis=0)[:iterations_displayed]
std = result_predictor_base_with_average.std(axis=0)[:iterations_displayed]
plt.plot(x,y, label = "Average", marker='o', markevery=200, markersize=8)
plt.fill_between(x, y - std, y + std, alpha=0.2)

y = result_predictor_base_with_sampling.mean(axis=0)[:iterations_displayed]
std = result_predictor_base_with_sampling.std(axis=0)[:iterations_displayed]
plt.plot(x,y, label = u"Sampling", marker='^', markevery=200, markersize=8)
plt.fill_between(x, y - std, y + std, alpha=0.2)

plt.xlabel("Number of samples")
plt.ylabel("MAE")
plt.legend()
plt.grid()

