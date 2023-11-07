import numpy as np
import matplotlib.pyplot as plt
import Classifiers2
import math 
import pandas as pd

#图片年龄数据不同学习率下各等级工作者权重及MAE变化曲线（在线聚合算法参数敏感性分析-有power）
dataset = pd.read_excel("age data.xls")
dataset = dataset.drop(['File_name'], axis = 1).sample(frac = 1)
true_labels = np.array(dataset["truth lable"])
X = np.array(dataset.drop(['truth lable'], axis = 1))
num = 10
X_extend = X.repeat(num, axis = 0)
y_extend = true_labels.repeat(num)
o = 1/1000*math.sqrt(2*math.log(10)/1000)
d = 1/1000*math.sqrt(2*math.log(10)*2)
t=1
strategy = 'constant'
i=0
lr_collction=[]
title=['0.0001','0.001','oracle','doubling trick','power']
for learning_rate in [0.0001,0.001, o, d, t]:
    if learning_rate == d:
        strategy = 'doubling'
    elif learning_rate == t:
        strategy = 'power'
    else:
        strategy = 'constant'
    predictor_base_with_weights = Classifiers2.Base_with_weights()
    predictor_base_with_weights.training(X_extend, y_extend, learning_rate, strategy)
    plt.title(title[i])
    plt.grid()
    predictor_base_with_weights.plot_weights_curve_v2()
    plt.title(title[i])
    predictor_base_with_weights.plot_MAE_curve()
    i+=1
    
    lr_collction.append(np.array(predictor_base_with_weights.lr_collection))
    lr=lr_collction[0]
    for j in range(len(lr_collction)-1):
        lr = np.vstack((lr, lr_collction[j+1]))
        
iterations_displayed = 1500 
x = range(iterations_displayed)

label=["0.0001","0.001","oracle","doubling trick","power"]
marker=['x','o','>','D','p','^']
plt.ylim(0,0.015)
for i in [0,1,2,3,4]:
    result=lr[i]
    y=result[:iterations_displayed]
    plt.plot(x,y, label = label[i], marker=marker[i], markevery=200, markersize=8)

plt.xlabel("Number of samples")
plt.ylabel("Learning Rate")
plt.legend(loc=3)
plt.grid()