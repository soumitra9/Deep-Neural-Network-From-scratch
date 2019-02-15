import os
import numpy as np
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import train_test as dataset
from math import log, exp
import random 
import pandas as pd

#implementing a class for layer
class Perceptron_layer():
    
    def __init__(self,inp,neurons, weight, bias):
        self.input = inp
        self.neurons = neurons
        #self.weight = np.random.rand(self.input.shape[0],neurons)
        #self.bias = np.random.rand(neurons,1)
        self.weight = weight
        self.bias = bias
        #self.target = tar
        #print(self.weight,self.bias)
        self.output = 0
    
    
    def relu(self,x):
        return np.maximum(x, 0)
    
    def sigmoid(self,x):
        deno = 1 + np.exp(-x)
        return 1/deno
    
    def differenciate_sigmod(self , x):
        return self.sigmoid(x) * (1- self.sigmoid(x))
    
    def softmax(self,x):
        exp_out = np.exp(x)
        total = np.sum(exp_out)
        return exp_out/total
           
    def calc_output(self):
        #print np.transpose(self.weight), np.transpose(self.weight).shape , type(np.transpose(self.weight))
        #print self.input, self.input.shape, type(self.input)
        return self.softmax(self.sigmoid(np.matmul(np.transpose(self.weight),self.input)))
        #print np.transpose(self.input)
        #self.output = self.sigmoid(np.matmul(np.transpose(self.weight),self.input))# + self.bias)
        #print self.output
        #return self.output
        
    def local_gradient(self):
        return self.differenciate_sigmod(np.matmul(np.transpose(self.weight),self.input))
        
    
    def local_grad_on_softmax(self):
        a = self.sigmoid(np.matmul(np.transpose(self.weight),self.input))
        part_1 = -1 * np.exp(a) * np.sum(a)
        part_2 = self.calc_output()
        return part_1 + part_2
    

    
class architecture(Perceptron_layer):
    def __init__(self,inp,target,layers,learning_rate):
        self.input = inp
        self.target = target
        self.layers = layers
        self.learning_rate = learning_rate
        self.loss = 0
        # initialize all layer outputs
        
        #structure
        
        #initialize weight and bias of layer 1(not the input) - very careful with the dimensions
        self.l1_w = np.random.rand(self.input.shape[0],3)# -----> 3 is no. of neuron
        self.l1_b = np.random.rand(3,1)
        self.layer1_out = 0
        self.l1_local_gradient = 0
        self.del_w = 0
        self.del_b = 0
        #common
        self.softmax_relu_local_grad = 0
        
        
    
    def forward_pass(self):#just one forward pass --- returns 
        #p = Perceptron_layer(self.input,2)
        
        #LAYER 1
        l1 = Perceptron_layer(self.input,self.l1_b.shape[1], self.l1_w, self.l1_b)
        
        self.layer1_out = l1.calc_output()
        self.l1_local_gradient = l1.local_gradient()
        self.softmax_relu_local_grad = l1.local_grad_on_softmax()
                              
                              
    def backward_pass(self):
        #start with loss at the last layer
        self.loss = self.loss_softmax(self.layer1_out,self.target)
                              
        #Then comes dL/dypred --- pred === softmax here
        loss_der = A.loss_derivative_softmax(self.layer1_out,self.target)
                              
        #Then comes dypred / drelu or dsoftmax / dsigmoid
        #self.soft_local_grad 
                              
        #Then comes dsigmoid / dw i.e === diff(sigmoid) matmul X(input)
        #self.l1_local_gradient
                              
        #Then u multiply ALL
        self.del_w = loss_der * np.matmul(self.input , (np.transpose((self.l1_local_gradient * (self.softmax_relu_local_grad)))))
        self.del_b = loss_der * self.l1_local_gradient * self.softmax_relu_local_grad
                              
        new_weight = self.l1_w - (self.learning_rate * self.del_w)
                              
        new_bias = self.l1_b - (self.learning_rate * self.del_b)
        
        self.l1_w = new_weight
        self.l1_b = new_bias
                              
   
    def loss_softmax(self,y_pred,y_tar):
        loss = np.sum(-1*(y_tar*np.log(y_pred)))
        return loss
    
    def loss_derivative_softmax(self,y_pred,y_tar): #dL/dypred
        
        return np.sum(-1*(y_tar/y_pred))

#data

X_train = dataset.X_train
X_test = dataset.X_test
Y_train = dataset.Y_train

x = np.transpose(X_train[:1,:]).astype(float)# Double bracket is important
y = np.transpose(Y_train[:1,:])
A = architecture(x,y,1,0.003)

for i in range(1,100):
    
    #print input_train[i],input_target[i][0]
    print('------------------epoch{0}-----------'.format(i))
    rand = random.randrange(0, len(X_train)-1, 1)
    
    A.input = np.transpose(X_train[rand:rand+1,:]).astype(float)# Double bracket is important
    A.target = np.transpose(Y_train[rand:rand+1,:])
    #print('weight ', A.l1_w)
    #print('bias ', A.l1_b)
    
    A.forward_pass()
    A.backward_pass()
    print(A.loss)
