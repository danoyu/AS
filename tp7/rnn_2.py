# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:49:42 2017

@author:  3200109
"""

from charDataset import *
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

train = torch.load('as-tp6/train_data.tx')
vocab = torch.load('as-tp6/vocab.tx')
n_words = len(vocab)


#c = CharDataset('train_data.tx','vocab.tx',50)

#retourne une sequence dans [0] et le reste dans [1]
def seq(text,debut,fin):
    return text[debut:fin], text[fin:]

# retourne une sequence codé dans [0] et le reste dans [1]
def code_seq(text,debut,fin,vocab):
    s = seq(text,debut,fin)
    return char2code(s[0],vocab), char2code(s[1],vocab)

text = 'je suis la'


def transform_one_hot(digit,n):
    y_onehot = torch.FloatTensor(n)
    y_onehot.zero_()
    y_onehot[digit] = 1
    return y_onehot


def transform_one_hot_sequence(sequence,n):
    seq_onehot = torch.FloatTensor(len(sequence),n)
    for i in range(len(sequence)):
        seq_onehot[i] = transform_one_hot(sequence[i],n)
    return seq_onehot

#transform_one_hot_sequence(char2code('n vreb br',vocab),n_words)


class Modele(nn.Module):
    def __init__(self,l):
        super(Modele,self).__init__()
        self.l = l
        
        #init h a 0 de dimension d a changer a chaque linear donc 2 param a apprendre w_h et w_x
        # Il faut 3 linear: 
        # - un pour passer de Rv -> Rd
        # le 1er converti une lettre (dimension v) dans l'espace latent
        # - un pour passer de Rd -> Rd
        # le 2e predit la suite dans l'espace latent
        # - un pour passer de Rd -> Rv
        # le 3e transforme dans v
        # on doit faire un logSoftmax pour avoir p(y1 | x_0,h_0) puis tirage puis one_hot
        self.encodage = nn.Linear(n_words,self.l, bias=False)
        self.decodage = nn.Linear(self.l,self.l, bias=False)
        self.d2 = nn.Linear(self.l, n_words, bias=False)
        self.tanh = nn.Tanh()
        
    def forward_train(self,x,y,maxlenght = 140 , stopWord = "~"):
        #mettre formule du forward 
        # ne marche pas au niveau du decodage je sais pas pourquoi
        out = []
        H = [Variable(torch.zeros(x.size(0),self.l))]
        logSoftmax = nn.LogSoftmax()
        _, argmax = x[:,0].max(dim=-1)
        preds = [argmax]
        print(preds)
        print(preds[-1])
        print(preds[-1].data[0])
        maxlenght = maxlenght if test else y.size()
        print("************")
        print(gfg)
        for i in range(len(x)):
            h = self.tanh(self.encodage(x[i]) + self.decodage(H[-1]))
            H.append(h)
            predProba = self.d2(h)
            #print(predProba)
            print(torch.sum(predProba))
            out.append(predProba)
        return out
    

    def loss(self,x,y):
        return 0


m = Modele(2)
text = 'je suis la'
test = code_seq(text,0,3,vocab)
x_train = Variable(transform_one_hot_sequence(test[0],n_words))
y_train = Variable(transform_one_hot_sequence(test[1],n_words))

a = m.forward_train(x_train,y_train)
#print(len(y_train))
#len(a)
#print(a[0])