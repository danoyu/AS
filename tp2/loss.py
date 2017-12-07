# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:35:06 2017
@author: kalifou rene TRAORE - M2 DAC
"""
import numpy as np

class Loss(object):
    """Class to model a General Loss function"""
    def __init__(self,Y_dim):
        self.dim=Y_dim
    def getLossValue(self,y_predict,y_reel):
        pass
    
    def backward(self,y_predict,y_reel):
        pass
    
class MSE_Loss(Loss):
    """Class for the Mean Square Error Loss"""
    
    def getLossValue(self,y_predict,y_reel):
        loss_val=0
        if type(y_predict[0]) == np.ndarray:
            Y_l = len(y_predict)
            for i in range(Y_l):
                loss_val += (y_predict[i]-y_reel[i])**2
        else:
            Y_l = 1
            loss_val +=(y_predict-y_reel)**2
        
        return sum(loss_val)/ self.dim
        
    def backward(self,y_predict,y_reel):
        """derivative along y_reel of the Loss"""
        if type(y_predict) == np.ndarray:
            Y = len(y_predict)
        else:
            Y=1        
        grad_val = 2*( y_predict- y_reel )
        
        return grad_val / float(Y)