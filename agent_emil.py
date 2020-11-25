#%% Imports

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#%% Agent

class ReplayMemory(object):
    """Experience Replay Memory"""
    
    def __init__(self, capacity):
        #self.size = size
        self.memory = deque(maxlen=capacity)
    
    def add(self, *args):
        """Add experience to memory."""
        self.memory.append([*args])
    
    def sample(self, batch_size):
        """Sample batch of experiences from memory with replacement."""
        return random.sample(self.memory, batch_size)
    
    def count(self):
        return len(self.memory)

class DQN(nn.Module):
    """Deep Q-network with target network"""
    
    def __init__(self, NN_PARAMS):
        super(DQN, self).__init__()
        self.NN_PARAMS = NN_PARAMS
        # network
        if NN_PARAMS['n_hidden_layers']==0:
            self.out = nn.Linear(NN_PARAMS['n_inputs'], NN_PARAMS['n_outputs'])
        else:
            self.out = nn.Linear(NN_PARAMS['n_hidden_units'][-1], NN_PARAMS['n_outputs'])
        
        # hidden layers
        if NN_PARAMS['n_hidden_layers']>0:
            self.hd_layers = []
            input_units = NN_PARAMS['n_inputs']
            for hidden_units in NN_PARAMS['n_hidden_units']:
                self.hd_layers.append(nn.Linear(input_units, hidden_units))
                input_units = hidden_units
        
        # training
        if NN_PARAMS['optimizer']=='SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=NN_PARAMS['learning_rate'])
        elif NN_PARAMS['optimizer']=='ADAM':
            self.optimizer = optim.Adam(self.parameters(),lr=NN_PARAMS['learning_rate'],weight_decay=NN_PARAMS['weight_decay'])
    
    def forward(self, x):
        if self.NN_PARAMS['n_hidden_layers']==0:
            return self.out(x)
        else:
            for f_hid_layer in self.hd_layers:
                if self.NN_PARAMS['relu']:
                    x = F.relu(f_hid_layer(x))
                else:
                    x = f_hid_layer(x)
            return self.out(x)        
    
    def loss(self, q_outputs, q_targets):
        return torch.sum(torch.pow(q_targets - q_outputs, 2))
    
    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)

# %%
