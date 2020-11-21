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
    
    def __init__(self, n_inputs, n_outputs, NN_PARAMS):
        super(DQN, self).__init__()
        # network
        self.out = nn.Linear(n_inputs, n_outputs)
        
        # training
        if NN_PARAMS['optimizer']=='SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=NN_PARAMS['learning_rate'])
        elif NN_PARAMS['optimizer']=='ADAM':
            self.optimizer = optim.ADAM(self.parameters(),lr=NN_PARAMS['learning_rate'],weight_decay=NN_PARAMS['weight_decay'])
    
    def forward(self, x):
        return self.out(x)
    
    def loss(self, q_outputs, q_targets):
        return torch.sum(torch.pow(q_targets - q_outputs, 2))
    
    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)

