import pandas as pd 
from cccp_reg import *
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_1 = pd.read_csv(r'elevators.csv')
print("该数据共有%d行，%d列。" % (data_1.shape[0],data_1.shape[1]))
np.random.seed(2)
data = data_1.sample(n=2000,axis=0,random_state=111)

X = data.drop('Y', axis=1)
X = np.array(X)
X = standarize(X)
print(X.shape)
y = data.iloc[:,data.columns == 'Y']
y =np.array(y)

#划分训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=0)
print('X_train.shape:',X_train.shape,'y_train.shape:',y_train.shape)
np.random.seed(2)

#设置模型参数 
arg_kernel = {'name':'rbf','par':3} # 核参数
t = 1 # 热核参数
arg_model = {'gamma_A':2E-4, 'gamma_I':0,'arg_kernel':arg_kernel,'t':t}
arg_alg = {'maxIte':50}

model,iteInf = train_ramp(X_train,y_train,arg_model,arg_alg)

#测试集上准确率
classifier = model['f']
alpha = model['alpha']
y_pred =  classifier(X_test,alpha)  # 预测标签
y_pred = np.array(y_pred)
y_pred = y_pred.reshape((len(y_test),1))
TP = np.sum( (y_pred ==1) & (y_test==1))
TN = np.sum( (y_pred ==-1) & (y_test==-1))
FP = np.sum( (y_pred ==1) & (y_test==-1))
FN = np.sum( (y_pred ==-1) & (y_test==1))
accuracy = (TP + TN)/(TP + TN + FP + FN)
P = TP/(TP + FP)
R = TP/(TP + FN)
F1 = 2*P*R/(P+R)
print('TP:',TP,'TN:',TN,'FP:',FP,'FN:',FN)
print('测试集准确率:',accuracy)
print('精确率：',P,'召回率：',R,'F1: ',F1)

#训练集上准确率 
y_pred1 =   classifier(X_train,alpha)
y_pred1 = np.array(y_pred1)
y_pred1 = y_pred1.reshape((len(y_train),1))
TP1 = np.sum( (y_pred1 ==1) & (y_train==1))
TN1 = np.sum( (y_pred1 ==-1) & (y_train==-1))
FP1 = np.sum( (y_pred1 ==1) & (y_train==-1))
FN1 = np.sum( (y_pred1 ==-1) & (y_train==1))
accuracy1 = (TP1 + TN1)/(TP1 + TN1 + FP1 + FN1)
print('训练集准确率:',accuracy1)
