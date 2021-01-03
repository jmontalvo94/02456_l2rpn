#%% Imports
import grid2op
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from agents import DQNAgent
from collections import deque
from grid2op.Action import TopologySetAction
from grid2op.Converter import IdToAct
from grid2op.Parameters import Parameters
from grid2op.Reward import L2RPNReward
from grid2op.Runner import Runner
from lightsim2grid import LightSimBackend
from networks import DQN
from rewards import FlowLimitAndBlackoutReward
from utils import set_seed_everywhere, cli_test


#%% General parameters

# args, general, params, nn_params = cli_test()
# NAME = args.name + '_last.pth'

NAME = 'policy_net_last.pth'

general = {
        "seed": 1,
        "lightsim": True,
        "checkpoint": 10,
        "chunk_size": 288, # a day 24*60 / 5
    }

test = {
        "runner": True,
        "n_episodes": 1,
        "n_core": 1,
        "chronic_id": 1
    }

nn_params = {
        "optimizer": "ADAM",
        "layers": [330, 330, 330, 200],
        "learning_rate": 0.001,
        "weight_decay": 0
    }

SEED = general['seed']
CHECKPOINT = general['checkpoint']
CHUNK_SIZE = general['chunk_size']
LIGHTSIM = general['lightsim']

if LIGHTSIM:
    backend = LightSimBackend() # faster ACOPF!

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#%% Initialization

# Set parameters for environment
p = Parameters()

# Hard-coded to avoid authomatic disconnection
p.HARD_OVERFLOW_THRESHOLD = 9999

# Initialize environment
env = grid2op.make('rte_case14_realistic', reward_class=FlowLimitAndBlackoutReward, backend=backend, action_class=TopologySetAction, param=p)

env.seed(SEED) # set seed
set_seed_everywhere(SEED) # set seed

env.deactivate_forecast() # no forecast or simulation, faster calculations

env.set_chunk_size(CHUNK_SIZE) # to avoid loading all the episode and fill memory


#%% Parameters

# Unpack parameters
RUNNER = test['runner']
N_EPISODES = test['n_episodes']
CHRONIC_ID = test['chronic_id']
N_CORE = test['n_core']
n_in = 330 # hard-coded observation vector size to remove unused variables
n_out = 151 # hard-coded action vector size for fixed action space

# Fix NN layers
layers = deque(nn_params['layers'])
layers.appendleft(n_in)
layers.append(n_out)
nn_params['layers'] = layers

# Initialize agent
agent = DQNAgent(env.action_space, IdToAct, nn_params, path='trained_models/' + NAME, change_bus_vect=False, set_line_status=False, change_line_status=False, redispatch=False)
agent.seed(SEED) # set seed

#%% Run the model

if RUNNER:

    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)
    runner.run(nb_episode=N_EPISODES, nb_process=N_CORE, path_save=NAME, pbar=True)

else:

    if N_EPISODES > 1:

        total_rewards = []

        for i in range(N_EPISODES):

            o = env.reset()

            # Initialize variables
            r = 0
            done = False
            total_reward = []
            length = 0

            # Play episode
            while True:
                a = agent.act(o, r, done)
                o, r, done, info = env.step(a)
                total_reward.append(r)
                length += 1
                if done:
                    total_rewards.append(total_reward)
                    break # if episode is over or game over
                
        env.close()

        print("The total reward was {:.2f}".format(sum(total_reward)))

    else:

        # Set environment to specific chronic
        env.set_id(CHRONIC_ID)

        o = env.reset()

        # Initialize variables
        r = 0
        done = False
        total_reward = []
        length = 0

        # Play episode
        while True:
            a = agent.act(o, r, done)
            o, r, done, info = env.step(a)
            total_reward.append(r)
            length += 1
            if done:
                break # if episode is over or game over
            
        env.close()

        print("The total reward was {:.2f}".format(sum(total_reward)))

# %% Visualization

if N_EPISODES > 1:
    plt.plot([item for sublist in total_rewards for item in sublist])
    plt.title(f'Episodes 1-{N_EPISODES}')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
else:
    plt.plot(total_reward)
    plt.title(f'Episode {CHRONIC_ID}')
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
