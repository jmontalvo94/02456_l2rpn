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
from utils import set_seed_everywhere, cli_test, obs_mask, get_max_values


#%% General parameters

args, general, params, nn_params, obs_params = cli_test()
NAME = args.name

# NAME = 'ddqn_500'
PATH_TRAINED = 'trained_models/'
PATH_SAVE = 'runner_agents/'
os.mkdir(PATH_SAVE+NAME)

# general = {
#         "seed": 1,
#         "lightsim": True,
#         "checkpoint": 10,
#         "agent_type": "DQN",
#         "chunk_size": 288, # a day 24*60 / 5
#     }

# params = {
#         "runner": True,
#         "n_episodes": 1,
#         "n_core": 1,
#         "chronic_id": 1
#     }

# nn_params = {
#         "optimizer": "ADAM",
#         "layers": [200, 200, 200, 200, 200],
#         "learning_rate": 0.001,
#         "weight_decay": 0
#     }

# obs_params = {
#         "year": False,
#         "month": False,
#         "day": False,
#         "hour": False,
#         "minute": False,
#         "day_of_week": False,
#         "prod_p": True,
#         "prod_q": True,
#         "prod_v": True,
#         "load_p": True,
#         "load_q": True,
#         "load_v": True,
#         "p_or": False,
#         "q_or": False,
#         "v_or": False,
#         "a_or": False,
#         "p_ex": False,
#         "q_ex": False,
#         "v_ex": False,
#         "a_ex": False,
#         "rho": True,
#         "line_status": True,
#         "timestep_overflow": True,
#         "topology_vector": True,
#         "line_cooldown": False,
#         "substation_cooldown": False,
#         "maintenance_time": False,
#         "maintenance_duration": False,
#         "target_dispatch": False,
#         "actual_dispatch": False
#     }

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
RUNNER = params['runner']
N_EPISODES = params['n_episodes']
CHRONIC_ID = params['chronic_id']
N_CORE = params['n_core']
n_out = 151 # action vector size from converter (includes do nothing action)
n_in = 164 # observation vector size from mask

# Fix NN layers
layers = deque(nn_params['layers'])
layers.appendleft(n_in)
layers.append(n_out)
nn_params['layers'] = layers

# Initialize agent
mask = obs_mask(env, obs_params)
max_values = get_max_values(env, mask)
agent = DQNAgent(env.action_space, mask, max_values, IdToAct, nn_params, path=PATH_TRAINED+NAME+'_policy_net_last.pth', change_bus_vect=False, set_line_status=False, change_line_status=False, redispatch=False)
agent.seed(SEED) # set seed

#%% Run the model

if RUNNER:

    runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)
    runner.run(nb_episode=N_EPISODES, nb_process=N_CORE, path_save=PATH_SAVE+NAME, pbar=True)

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
