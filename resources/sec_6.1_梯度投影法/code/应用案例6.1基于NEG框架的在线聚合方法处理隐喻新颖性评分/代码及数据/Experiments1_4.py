import numpy as np
import matplotlib.pyplot as plt
import Classifiers1
import math 


#人工合成数据不同学习率下各等级工作者权重及MAE变化曲线（在线聚合算法参数敏感性分析-有power）
base_learners_collection = [2,3,3,2]
true_labels = np.random.randint(0,255,10000)
variance= {'0': [0.5, 0.7, 0.9], '1': [3, 4, 5], '2': [7, 8, 9], '3': [10, 11, 12]}
o = 1/100*math.sqrt(2*math.log(10)/10000)
d = 1/100*math.sqrt(2*math.log(10)*2)
t=1
strategy = 'constant'
i=0
lr_collction=[]
title=['0.0001','0.001','0.01', '0.1','oracle','doubling trick','power']
for learning_rate in [0.0001,0.001, 0.01,0.1, o, d, t]:
    if learning_rate == d:
        strategy = 'doubling'
    elif learning_rate == t:
        strategy = 'power'
    else:
        strategy = 'constant'
    predictor_base_with_weights = Classifiers1.Base_with_weights(base_learners_collection = base_learners_collection, variance_dict = variance)
    predictor_base_with_weights.training(true_labels, learning_rate, strategy)
    plt.title(title[i])
    plt.grid()
    predictor_base_with_weights.plot_weights_curve()
    plt.title(title[i])
    predictor_base_with_weights.plot_MAE_curve()
    i+=1
    
    lr_collction.append(np.array(predictor_base_with_weights.lr_collection))
    lr=lr_collction[0]
    for j in range(len(lr_collction)-1):
        lr = np.vstack((lr, lr_collction[j+1]))

#学习率变化曲线       
iterations_displayed = 1500 
x = range(iterations_displayed)

label=['0.0001','0.001','0.01', '0.1','oracle','doubling trick','power']
marker=['x','o','>','D','p','^','<']
plt.ylim(0,0.015)
for i in [0,1,4,5,6]:
    result=lr[i]
    y=result[:iterations_displayed]
    plt.plot(x,y, label = label[i], marker=marker[i], markevery=200, markersize=8)

plt.xlabel("Number of samples")
plt.ylabel("Learning Rate")
plt.legend()
plt.grid()