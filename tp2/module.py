# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:33:39 2017
@author: kalifou rene TRAORE - M2 DAC
"""
import numpy as np

class Module(object) :
    """Abstract class to model a learning Module"""
    
    def __init__(self, N_C,N_X):
        """Initializing gradient & paramateres matrices"""
        self.grad = np.zeros((N_C,N_X)) 
        variance = 0.01
        self.weights = np.random.uniform(-variance,variance,size=(N_C,N_X))

    def backward_update_gradient(self,x,delta):
        """Updating the gradient"""
        pass

    def update_parameters(self,eps):
        """ Updating the module's parameters according to the equation : 
            W = W - Eps * G  """
        self.weights =  self.weights- eps * self.grad # 

    def reset_grad(self,N_C,N_X):
        """ Resetting the gradient to zero"""
        self.grad = np.zeros((N_C,N_X))

    def forward(self,x):
        """ Inference on X"""
        pass
    
    def backward_delta(self,x,delta):
        """ backward pass"""
        pass

class Linear_Module(Module):
    """Class for a Perceptron with a Linear activation function"""
    def forward(self,x):  
        """ Inference on X"""
        return x.dot(self.weights)
 
    def backward_update_gradient(self,x,delta):
        """Updating the gradient using delta coming from the following module and input X"""
        t = np.matrix(x).getT()
        r = t.dot(np.matrix(delta))
        self.grad += r   

    def backward_delta(self,x,delta):
        """Computing the delta to send backwards :delta * (derivative with respect to x of current forward)"""
        return self.weights.dot(delta)
        
class Tan_H_Module(Module):
    """Class for a Perceptron with a Linear activation function"""
    def forward(self,x):
        """ Inference on X"""
        return np.tanh(x)
        
    def backward_delta(self,x,delta):
        """Delta to send backwards """
        inter = np.array(1 - np.tanh(x)**2) ## derivative with respect to x of tan_h
        return inter * delta 