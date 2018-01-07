# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:44:32 2017

@author: rportelas
"""
import torch as t
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import _pickle as pickle
import os.path


class Model(nn.Module):

    def __init__(self,dim_x,dim_y,h):
        nn.Module.__init__(self)
        
        self.f1 = nn.Linear(dim_x,dim_h)
        self.f1.weight.data.uniform_(-0.1,0.1)
        
        self.f2 = nn.Linear(dim_h,dim_y)
        self.f2.weight.data.uniform_(-0.1,0.1)
        
    def forward(self,x):
        h_out = F.relu(self.f1(x))
        out = self.f2(h_out)
        return F.log_softmax(out,dim=0)


#function taken from http://karpathy.github.io/2016/05/31/rl/
def pre_process(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

#function taken from http://karpathy.github.io/2016/05/31/rl/
def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * discount_factor + r[t]
    discounted_r[t] = running_add
  return discounted_r

'''Hyper parameters settings'''
dim_h = 200
dim_x = 80*80 #according to pre_process function
dim_y = 2 #only 2 possible actions: [UP,DOWN]
learning_rate = 0.0001
discount_factor = 0.99
save_step = 100
render = False
resume = False

if __name__ == '__main__':

    #init environment
    env = gym.make("Pong-v0")
    observation = env.reset()

    #store previous x to compute difference of image
    previous_x = None

    running_reward = None
    reward_sum = 0
    step_reward_sum = 0
    step_reward_list = []
    rewards = []
    logP_actions = []
    episode_number = 0

    #init model and optim
    if resume and os.path.isfile('pg_model_save.p') and os.path.isfile('step_reward_list.p'):
        print("starting from loaded model")
        model = pickle.load(open('pg_model_save.p', 'rb'))
        step_reward_list = pickle.load(open('step_reward_list.p', 'rb'))
        episode_number = len(step_reward_list) * save_step
    else:
        print("starting from scratch")
        model = Model(dim_x,dim_y,dim_h)
    Optim = t.optim.SGD(model.parameters(), lr=learning_rate)
    num = 0
    #train for 10K steps and saves model every save_step steps
    while True:
        if render: env.render()
        #create x input by preprocessing observation + computing difference
        current_x = pre_process(observation)
        x = current_x - previous_x if previous_x is not None else np.zeros(dim_x)
        previous_x = current_x

        x = Variable(t.FloatTensor(x))

        #forward pass and sample action
        logP = model(x)
        action = t.multinomial(logP.exp(),1)
        logP_actions.append(logP[action]) 

        #map action to game action then step environment
        game_action = 2 if action.data[0] == 0 else 3
        observation, reward, done, info = env.step(game_action)

        #record reward
        rewards.append(reward)
        reward_sum += reward
        num += 1
        if num==1000: # an episode is finished
            #print(rewards)
            episode_number += 1

            #compute the discounted reward backwards through time
            disc_rewards = discount_rewards(rewards)
            #standardize the rewards to be unit normal (helps control the gradient estimator variance)
            disc_rewards = (disc_rewards - np.mean(disc_rewards)) / np.std(disc_rewards)

            #compute loss and backward prop it
            loss = []
            for i in range(len(disc_rewards)):
                loss.append(-logP_actions[i] * disc_rewards[i])
            Optim.zero_grad()
            loss = t.cat(loss).sum()
            print(loss)
            
            loss.backward() 
            Optim.step()

            print('resetting env. episode reward total was %f.' % (reward_sum))

            step_reward_sum += reward_sum
            reward_sum = 0
            observation = env.reset() # reset env
            prev_x = None
            #reset accumulator for new episode
            logP_actions = []
            rewards = []

            if (episode_number % save_step) == 0: 
                #store model
                #pickle.dump(model, open('pg_model_save.p', 'wb'))

                step_reward_list.append(step_reward_sum / save_step)
                step_reward_sum = 0

                #store average reward_sum on last 100 episodes
                #pickle.dump(step_reward_list, open('step_reward_list.p', 'wb'))
                #plot evolution of average score
                #print(step_reward_list)
                plt.plot(step_reward_list)
                plt.xlabel('number of episodes (x'+str(save_step)+')')
                plt.ylabel('average reward')
                plt.show(block=False)
                plt.savefig('reward_evolution.png')
               

        if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
