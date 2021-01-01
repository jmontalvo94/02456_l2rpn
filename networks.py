import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    """Deep Q-network with target network"""
    
    def __init__(self, nn_params):
        super().__init__()

        # General parameters
        self.layers = nn_params['layers']
        self.depth = len(self.layers) - 1
        
        # Create architecture
        self.nn = nn.Sequential()
        for n in range(self.depth - 1):
            self.nn.add_module(f"layer_{n}", nn.Linear(self.layers[n], self.layers[n + 1]))
            self.nn.add_module(f"act_{n}", nn.ReLU())
        self.nn.add_module(f"layer_{n + 1}", nn.Linear(self.layers[n + 1], self.layers[n + 2]))
        
        # Optimizer
        if nn_params['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=nn_params['learning_rate'])
        elif nn_params['optimizer'] == 'ADAM':
            self.optimizer = optim.Adam(self.parameters(), lr=nn_params['learning_rate'], weight_decay=nn_params['weight_decay'])

    def forward(self, x):
        return self.nn(x)
    
    def loss(self, q_outputs, q_targets):
        return torch.sum(torch.pow(q_targets - q_outputs, 2))
    
    def update_params(self, new_params, tau):
        params = self.state_dict();
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)