# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:04:44 2021

@author: lenovo
"""
import pandas as pd
import numpy as np
import Classifiers3
import matplotlib.pyplot as plt

dataset1 = pd.read_csv("test.csv")
dataset2 = pd.read_csv("train.csv")
dataset3 = pd.read_csv("validation.csv")
data=dataset1.append(dataset2)
data=data.append(dataset3)
data=data.loc[:,['True_Label','A1','A2','A3','A4','A5']].sample(frac=1)
true_labels = np.array(data["True_Label"])
learning_rate=1
strategy='power'
error_=[]
X = np.array(data.drop(['True_Label'], axis = 1))
w = Classifiers3.Base_with_weights()
w.training(X, true_labels, learning_rate, strategy)
error_.append(w.error)
error=[]
loss=[]
z=[0,0,0,0,0,0]
f=[0,0,0,0,0,0]
l=[0,0,0,0,0,0]
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

for i in range(len(error_c)):
   if error_c[i]>0:
       z[5]+=1
   elif error_c[i]==0:
       l[5]+=1
   else:
       f[5]+=1
      
result=loss[0]
for j in range(len(loss)-1):
    result = np.vstack((result, loss[j+1]))
print(result.mean(axis=0))
print(result.std(axis=0))
    
result=error[0]
for j in range(len(error)-1):
    result = np.vstack((result, error[j+1]))

    #loss=abs(predict-true_labels)
marker=['x','o','>','P','^']
#label=['Worker1','Worker2','Worker3','Worker4','Worker5','Proposed']
#weight_collection = np.array(self.weight_collection)
x=np.linspace(0,3111,3112)
plt.scatter(x, error_, label = "Proposed", marker='<',s=1)   
for i in range(0,5):
    plt.scatter(x, result[:,i], label = "Worker{}".format(i + 1), marker=marker[i],s=1)   
plt.xlabel('Number of samples')
plt.ylabel('Error')
plt.legend()
plt.show()

x=[1,2,3,4,5,6]
a=[]
p=[]
for i in range(len(f)):
    a.append(f[i]+l[i])
    p.append(str(z[i])+':'+str(f[i]))

#plt.xlim(0.5,5.7) 
plt.ylim(0,3400)    
plt.bar(x, f, align="center",color='gray', tick_label=["Worker1", "Worker2", "Worker3", "Worker4", "Worker5",'Proposed'], label="Error<0")
plt.bar(x, l, align="center",color='#B088FF', bottom=f,label="Error=0")
plt.bar(x, z, align="center",color='#FF3333', bottom=a, label="Error>0")
for i in range(1,7):
    plt.text(i,3200,p[i-1],horizontalalignment="center", fontsize=10)
#plt.xlabel('Workers')
plt.ylabel('Number of samples')
plt.legend(loc=4)
plt.show() 
