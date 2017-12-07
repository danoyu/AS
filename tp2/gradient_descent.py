# -*- coding: utf-8 -*-

import numpy as np
import random
from module import Linear_Module
from module import Tan_H_Module
from loss import MSE_Loss

def SGD(X,X_test,Y,Y_test,K_iterations,eps,measurement_mode=True):
    """ Stochastic Gradient Descent"""        
    
    N = len(Y)    
    print 'Y,Y[0]', N,len(Y[0])
    N_test = len(Y_test)     
    N_X =len(X[0])
    N_C = len(Y[0]) if type(Y[0]) == np.ndarray else 1
    print 'Ns dim',N,N_C,N_test,N_X
    loss = MSE_Loss(Y.shape)
    
    # Modules to process the data
    module_1 = Linear_Module(256,500)
    module_2 = Tan_H_Module(500,500)
    module_3 = Linear_Module(500,10)
    module_4 = Tan_H_Module(10,10)    
    
    cpt = 0
    losses =[]
    acc_training = []
    acc_test = []
    
    # Init. vectors for Inference
    Y_predicted_1 = np.zeros((N,500))
    Y_predicted_2 = np.zeros((N,500))
    Y_predicted_3 = np.zeros((N,10))
    Y_predicted_4 = np.zeros((N,10))
    
    while cpt < K_iterations:
        
        # iterating over random Xs
        i = random.randint(0,N-1)        
        
        # Resetting gradients
        module_1.reset_grad(256,500)
        module_2.reset_grad(500,500)
        module_3.reset_grad(500,10)
        module_4.reset_grad(10,10)
        
        # Inference
        Y_predicted_1[i]= module_1.forward(X[i])
        Y_predicted_2[i]= module_2.forward(Y_predicted_1[i])
        Y_predicted_3[i]= module_3.forward(Y_predicted_2[i])
        Y_predicted_4[i]= module_4.forward(Y_predicted_3[i])
        

        # Loss & Loss derivative
        err = loss.getLossValue(Y_predicted_4[i],Y[i])
        delta_loss = loss.backward(Y_predicted_4[i],Y[i])
        
        # Updating the model 2's values : G = G + Delta | Computing the delta to send backwards
        module_4.backward_update_gradient(Y_predicted_3[i],delta_loss)
        delta_4 = module_4.backward_delta(Y_predicted_3[i],delta_loss)

        # Updating the model 3's values : G = G + Delta | Computing the delta to send backwards
        module_3.backward_update_gradient(Y_predicted_2[i],delta_4)
        delta_3 = module_3.backward_delta(Y_predicted_2[i],delta_4)        
        
        # Updating the model 2's values : G = G + Delta | Computing the delta to send backwards
        module_2.backward_update_gradient(Y_predicted_1[i],delta_3)
        delta_2 = module_2.backward_delta(Y_predicted_1[i],delta_3)
        
        # Updating the model 1's values : G = G + Delta
        module_1.backward_update_gradient(X[i],delta_2)
        
        module_1.update_parameters(eps)
        module_2.update_parameters(eps)
        module_3.update_parameters(eps) 
        module_4.update_parameters(eps)                 
        
        # Accuracy on test & training
        y_pred_on_test_1 = np.zeros((N,500))
        y_pred_on_test_2 = np.zeros((N,500))
        y_pred_on_test_3 = np.zeros((N,10))
        y_pred_on_test_4 = np.zeros((N,10))
        
        y_pred_on_training_1 = np.zeros((N,500))
        y_pred_on_training_2 = np.zeros((N,500))       
        y_pred_on_training_3 = np.zeros((N,10))
        y_pred_on_training_4 = np.zeros((N,10))        
        
        
        ####### Inference for accuracy measurement ##################
        cmp_test = 0
        cmp_train = 0
        if measurement_mode:
                
                y_pred_on_training_1 = module_1.forward(X)
                y_pred_on_training_2 = module_2.forward(y_pred_on_training_1)            
                y_pred_on_training_3 = module_3.forward(y_pred_on_training_2)            
                y_pred_on_training_4 = module_4.forward(y_pred_on_training_3)
        
                y_pred_on_test_1 = module_1.forward(X_test)
                y_pred_on_test_2 = module_2.forward(y_pred_on_test_1)            
                y_pred_on_test_3 = module_3.forward(y_pred_on_test_2)            
                y_pred_on_test_4 = module_4.forward(y_pred_on_test_3)
                
                
                cmp_train = sum( [ 1 if  np.array_equal(np.sign(y_pred_on_training_4[k]),Y[k])\
                                     else 0 for k in range(N)])
                cmp_test = sum( [1 if np.array_equal(np.sign(y_pred_on_test_4[kk]),Y_test[kk])\
                                   else 0 for kk in range(N_test)])      
                                   
                err = loss.getLossValue(y_pred_on_training_4,Y)/len(Y)   
                
        acc_test.append( cmp_test/float(N_test))
        acc_training.append( cmp_train/float(N))        
        
        losses.append(err)
        cpt+=1
            
    print "Done with SGD"
    return losses,acc_training,acc_test
