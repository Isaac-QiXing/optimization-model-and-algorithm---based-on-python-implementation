# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:04:44 2021

@author: lenovo
"""
import pandas as pd
import numpy as np
import Classifiers2
import matplotlib.pyplot as plt

#图片年龄数据分析
dataset = pd.read_excel("age data.xls")
dataset = dataset.drop(['File_name'], axis = 1).sample(frac = 1)
true_labels = np.array(dataset["truth lable"])
X = np.array(dataset.drop(['truth lable'], axis = 1))
learning_rate=1
strategy='power'
error_=[]
w = Classifiers2.Base_with_weights()
w.training(X, true_labels, learning_rate, strategy)
error_.append(w.error)
error=[]
loss=[]
z=[0,0,0,0,0,0,0,0,0,0]
f=[0,0,0,0,0,0,0,0,0,0]
l=[0,0,0,0,0,0,0,0,0,0]
for i in range(len(true_labels)):
    loss_=X[i]-true_labels[i]
    loss.append(abs(X[i]-true_labels[i]))
    error.append(X[i]-true_labels[i])
    for j in range(len(loss_)):
        if loss_[j]>0:
            z[j]+=1
        elif loss_[j]==0:
            l[j]+=1
        else:
            f[j]+=1
error_c=np.array(error_[0])
#print(error_c)
'''
for i in range(len(error_c)):
   if error_c[i]>0:
       z[5]+=1
   elif error_c[i]==0:
       l[5]+=1
   else:
       f[5]+=1
'''     
result=loss[0]
for j in range(len(loss)-1):
    result = np.vstack((result, loss[j+1]))
print(result.mean(axis=0))
print(result.std(axis=0))
    
result=error[0]
for j in range(len(error)-1):
    result = np.vstack((result, error[j+1]))

    #loss=abs(predict-true_labels)
marker=['x','o','>','P','^','>','D','*','p','o']
#label=['Worker1','Worker2','Worker3','Worker4','Worker5','Proposed']
#weight_collection = np.array(self.weight_collection)
x=np.linspace(0,1001,1002)
plt.scatter(x, error_, label = "Proposed", marker='<',s=1)   
for i in range(0,10):
    plt.scatter(x, result[:,i], label = "Worker{}".format(i + 1), marker=marker[i],s=1)   
plt.xlabel('Number of samples')
plt.ylabel('Error')
plt.legend()
plt.show()

x=[1,2,3,4,5,6,7,8,9,10]
#x=[0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
a=[]
p=[]
for i in range(len(f)):
    a.append(f[i]+l[i])
    p.append(str(z[i])+':'+str(f[i]))
    
plt.figure(figsize=(10,5))
#plt.xlim(0.5,5.7) 
plt.ylim(0,1100)    
plt.bar(x, f, align="center",color='gray', tick_label=["Worker1", "Worker2", "Worker3", "Worker4", "Worker5","Worker6", "Worker7", "Worker8", "Worker9", "Worker10"], label="Error<0")
plt.bar(x, l, align="center",color='#B088FF', bottom=f,label="Error=0")
plt.bar(x, z, align="center",color='#FF3333', bottom=a, label="Error>0")
for i in range(1,11):
    plt.text(i,1030,p[i-1],horizontalalignment="center", fontsize=8)
#plt.xlabel('Workers')
plt.ylabel('Number of samples')
plt.legend(loc=5)
plt.show() 
