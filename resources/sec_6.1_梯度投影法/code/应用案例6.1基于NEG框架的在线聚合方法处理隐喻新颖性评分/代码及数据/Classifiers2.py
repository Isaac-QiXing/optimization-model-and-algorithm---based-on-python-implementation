import numpy as np
import matplotlib.pyplot as plt
#import math

#图片年龄数据调用
class Base_with_weights:
    
    def __init__(self):
        self.num_base_learners = 10
        self.weight = 1.0 / self.num_base_learners * np.ones(self.num_base_learners)
        self.cumulative_loss = 0 
        self.MAE_collection = [] 
        self.weight_collection = [] 
        self.step_counter = 0 
        self.MAE_ = []
        self.lr_collection=[]
        self.error=[]
        self.z=[]
          
    def training(self, X, y,learning_rate, strategy):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        num_observations = len(self.X)
        for t in range(num_observations):
            self.t=t
            self.learning_rate=self.learning_rate_adjusting(strategy)
            predict = self.predict(self.X[t])
            self.cumulative_loss += np.abs(predict - self.y[t])
            self.MAE_.append(abs(predict - self.y[t]))
            self.error.append(predict - self.y[t])
            gradient = (predict - self.y[t]) * self.X[t]
            self.z.append(abs(gradient))
            numerator = self.weight * np.exp(-self.learning_rate * gradient)
            self.weight = numerator * 1.0 / np.sum(numerator)
            self.weight_collection.append(self.weight)
            self.step_counter += 1
            self.MAE_collection.append(self._get_MAE())
            self.lr_collection.append(self.learning_rate)
            
    def is_two_power(self, x):
        if (x & (x-1)) == 0:
            return True
        else:
            return False
    
    def learning_rate_adjusting(self, strategy):
        if strategy == "constant":
            pass
        elif strategy == "oracle":
            pass
        elif strategy == "doubling":
            if self.is_two_power(self.step_counter + 1):
                self.learning_rate = self.learning_rate / np.sqrt(2)
        elif strategy == "power":
            self.learning_rate = 0.2*((self.t+1)**(-2/3))
        else:
            raise NameError#, "No such an option."
        return self.learning_rate
            
    def predict(self, x):
        return np.dot(self.weight, x)
    
    def _get_MAE(self): 
        return self.cumulative_loss * 1.0 / self.step_counter
    
    def plot_MAE_curve(self, x_label = "Number of samples", y_label = "MAE"):
        plt.plot(self.MAE_collection)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    def plot_weights_curve(self, x_label = "Number of samples", y_label = "Weights"):
        weight_collection = np.array(self.weight_collection)
        for i in range(weight_collection.shape[-1]):
            #plt.plot(weight_collection[:, i], label = "Worker{}".format(i + 1))
            if i == 0:
                plt.plot(weight_collection[:, i], c = "r", label = "Class 1")
            if 0 < i < 2:
                plt.plot(weight_collection[:, i], c = "r")
            if i == 2:
                plt.plot(weight_collection[:, i], c = "g", label = "Class 2")
            if 2 < i < 5:
                plt.plot(weight_collection[:, i], c = "g")
            if i == 5:
                plt.plot(weight_collection[:, i], c = "b", label = "Class 3")
            if 5 < i < 8:
                plt.plot(weight_collection[:, i], c = "b")
            if i == 8:
                plt.plot(weight_collection[:, i], c = "k", label = "Class 4")
            if i > 8:
                plt.plot(weight_collection[:, i], c = "k")
            
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()
        
    def plot_weights_curve_v2(self, x_label =  "Number of samples", y_label = "Weights"):
        weight_collection = np.array(self.weight_collection)
        for i in range(weight_collection.shape[-1]):
            plt.plot(weight_collection[:, i], label = "Worker {}".format(i + 1))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc=3)
        plt.show()


class Base_with_average:

    def __init__(self):
        self.num_base_learners = 10
        self.weight = 1.0 / self.num_base_learners * np.ones(self.num_base_learners)
        self.cumulative_loss = 0
        self.MAE_collection = []
        self.step_counter = 0
        self.lr_collection=[]
        self.MAE_ = []
        self.weight_collection = []
    
    def training(self, X, y,learning_rate, strategy):
        self.X = X
        self.y=y
        self.learning_rate = learning_rate
        num_observations = len(self.X)
        for t in range(num_observations):
            self.t=t
            self.learning_rate=self.learning_rate_adjusting(strategy)
            predict = self.X[t].mean()
            self.cumulative_loss += np.abs(predict - self.y[t])
            self.MAE_.append(abs(predict - self.y[t]))
            gradient = (predict - self.y[t]) * self.X[t]
            numerator = self.weight * np.exp(-self.learning_rate * gradient)
            self.weight = numerator * 1.0 / np.sum(numerator)
            self.weight_collection.append(self.weight)
            self.step_counter += 1
            self.MAE_collection.append(self._get_MAE())
            self.lr_collection.append(self.learning_rate)
    
    def is_two_power(self, x):
        if (x & (x-1)) == 0:
            return True
        else:
            return False
    
    def learning_rate_adjusting(self, strategy):
        if strategy == "constant":
            pass
        elif strategy == "oracle":
            pass
        elif strategy == "doubling":
            if self.is_two_power(self.step_counter + 1):
                self.learning_rate = self.learning_rate / np.sqrt(2)
        elif strategy == "power":
            self.learning_rate = 0.2*((self.t+1)**(-2/3))
        else:
            raise NameError#, "No such an option."
        return self.learning_rate
   
            
    def predict(self, x):
        return np.dot(self.weight, x)
    
    def _get_MAE(self): 
        return self.cumulative_loss * 1.0 / self.step_counter
    
    def plot_MAE_curve(self, x_label = "Number of samples", y_label = "MAE"):
        plt.plot(self.MAE_collection)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


class Base_with_sampling:
 
    def __init__(self):
        self.num_base_learners = 10
        self.weight = 1.0 / self.num_base_learners * np.ones(self.num_base_learners)
        self.cumulative_loss = 0
        self.MAE_collection = []
        self.weight_collection = []
        self.step_counter = 0
        self.lr_collection=[]
        self.MAE_ = []        
    
    def training(self, X, y,learning_rate, strategy):
        self.X = X
        self.y=y
        self.learning_rate = learning_rate
        num_observations = len(self.X)
        for t in range(num_observations):
            self.t=t
            self.learning_rate=self.learning_rate_adjusting(strategy)
            predict = self.X[t][self.sampling()]
            self.cumulative_loss += np.abs(predict - self.y[t])
            gradient = (predict - self.y[t]) * self.X[t]
            numerator = self.weight * np.exp(-self.learning_rate * gradient)
            self.weight = numerator * 1.0 / np.sum(numerator)
            self.weight_collection.append(self.weight)
            self.step_counter += 1
            self.MAE_collection.append(self._get_MAE())
            self.lr_collection.append(self.learning_rate)
            
    def sampling(self):
        outcome = np.argmax(np.random.multinomial(1, self.weight, size = 1))
        return outcome            
            
    def is_two_power(self, x):
        if (x & (x-1)) == 0:
            return True
        else:
            return False
    
    def learning_rate_adjusting(self, strategy):
        if strategy == "constant":
            pass
        elif strategy == "oracle":
            pass
        elif strategy == "doubling":
            if self.is_two_power(self.step_counter + 1):
                self.learning_rate = self.learning_rate / np.sqrt(2)
        elif strategy == "power":
            self.learning_rate = 0.2*((self.t+1)**(-2/3))
        else:
            raise NameError#, "No such an option."
        return self.learning_rate
   
    def predict(self, x):
        return x[self.sampling()]
    
    def _get_MAE(self): 
        return self.cumulative_loss * 1.0 / self.step_counter
    
    def plot_MAE_curve(self, x_label = "Number of samples", y_label = "MAE"):
        plt.plot(self.MAE_collection)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    def plot_weights_curve(self, x_label = "Number of samples", y_label = "Weights"):
        weight_collection = np.array(self.weight_collection)
        for i in range(weight_collection.shape[-1]):
            plt.plot(weight_collection[:, i], label = "Worker{}".format(i + 1))
            
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


class Base_with_stepsize:

    def __init__(self):
        self.num_base_learners = 10
        self.weight = 1.0 / self.num_base_learners * np.ones(self.num_base_learners)
        self.cumulative_loss = 0 
        self.MAE_collection = [] 
        self.weight_collection = []      
        self.step_counter = 0 
        self.lr_collection=[]
        self.MAE_ = []
        
    def training(self, X, y,learning_rate, strategy):
        self.X = X
        self.y=y
        self.learning_rate = learning_rate
        num_observations = len(self.X)
        for t in range(num_observations):
            self.t=t
            self.learning_rate=self.learning_rate_adjusting(strategy)
            predict = self.predict(self.X[t])
            self.cumulative_loss += np.abs(predict - self.y[t])
            gradient = (predict - self.y[t]) * self.X[t]
            numerator = self.weight * np.exp(-self.learning_rate * gradient)
            self.weight = numerator * 1.0 / np.sum(numerator)
            self.weight_collection.append(self.weight)
            self.step_counter += 1
            self.MAE_collection.append(self._get_MAE())
            self.lr_collection.append(self.learning_rate)
            
            
    def is_two_power(self, x):
        if (x & (x-1)) == 0:
            return True
        else:
            return False
    
    def learning_rate_adjusting(self, strategy):
        if strategy == "constant":
            pass
        elif strategy == "oracle":
            pass
        elif strategy == "doubling":
            if self.is_two_power(self.step_counter + 1):
                self.learning_rate = self.learning_rate / np.sqrt(2)
        elif strategy == "power":
            self.learning_rate = 0.2*((self.t+1)**(-2/3))
        else:
            raise NameError#, "No such an option."
        return self.learning_rate
        
    def predict(self, x):
        return np.dot(self.weight, x)
    
    def _get_MAE(self): 
        return self.cumulative_loss * 1.0 / self.step_counter
    
    def plot_MAE_curve(self, x_label = "Number of samples", y_label = "MAE"):
        plt.plot(self.MAE_collection)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
    
    def plot_weights_curve(self, x_label =  "Number of samples", y_label = "Weights"):
        weight_collection = np.array(self.weight_collection)
        for i in range(weight_collection.shape[-1]):
            plt.plot(weight_collection[:, i], label = "Worker {}".format(i + 1))
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

def shuffle(X, labels):

    randomize = np.arange(len(labels))
    np.random.shuffle(randomize)
    X_shuffle = X[randomize]
    labels_shuffle = labels[randomize]
    return X_shuffle, labels_shuffle
