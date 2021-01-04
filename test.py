#%% Imports
import grid2op
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from agents import DQNAgent
from collections import deque
from glob import glob
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
# PATH_TRAINED = 'trained_models/'
# PATH_SAVE = 'runner_agents/'

# general = {
#         "seed": 1,
#         "lightsim": True,
#         "checkpoint": 10,
#         "agent_type": "DDQN",
#         "chunk_size": 288, # a day 24*60 / 5
#     }

# params = {
#         "multiple": True,
#         "runner": False,
#         "n_episodes": 10,
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
AGENT_TYPE = general['agent_type']

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
MULTIPLE = params["multiple"]
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

# Build
if MULTIPLE:
    if AGENT_TYPE == "DQN":
        agents = glob(PATH_TRAINED+"dqn*.pth")
    elif AGENT_TYPE == "DDQN":
        agents = glob(PATH_TRAINED+"ddqn*.pth")

#%% Run the model

# TODO refactor code

if MULTIPLE:
    
    agents_rewards = []
    agents_lengths = []
    
    for a in agents:
        print(a)
        
        agent = DQNAgent(env.action_space, mask, max_values, IdToAct, nn_params, path=a, change_bus_vect=False, set_line_status=False, change_line_status=False, redispatch=False)
        agent.seed(SEED) # set seed
        
        if RUNNER:
            
            os.mkdir(PATH_SAVE+a.split('/')[1].split('.')[0])

            runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)
            runner.run(nb_episode=N_EPISODES, nb_process=N_CORE, path_save=PATH_SAVE+a.split('/')[1].split('.')[0], pbar=True)

        else:
            
            agent_rewards = []
            agent_lengths = []

            if N_EPISODES > 1:

                total_rewards = []
                lengths = []

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
                            lengths.append(length)
                            break # if episode is over or game over
                        
                agent_rewards.append(total_rewards)
                agent_lengths.append(lengths)

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
                
                agent_rewards.append(total_reward)
                agent_lengths.append(length)
            
        agents_rewards.append(agent_rewards)
        agents_lengths.append(agent_lengths)
    
else:
    
    agent = DQNAgent(env.action_space, mask, max_values, IdToAct, nn_params, path=PATH_TRAINED+NAME+'_policy_net_last.pth', change_bus_vect=False, set_line_status=False, change_line_status=False, redispatch=False)
    agent.seed(SEED) # set seed
    
    if RUNNER:
        
        os.mkdir(PATH_SAVE+NAME)

        runner = Runner(**env.get_params_for_runner(), agentClass=None, agentInstance=agent)
        runner.run(nb_episode=N_EPISODES, nb_process=N_CORE, path_save=PATH_SAVE+NAME, pbar=True)

    else:
        
        agent_rewards = []
        agent_lengths = []

        if N_EPISODES > 1:

            total_rewards = []
            lengths = []

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
                        lengths.append(length)
                        break # if episode is over or game over

            agent_rewards.append(total_rewards)
            agent_lengths.append(lengths)

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
            
            agent_rewards.append(total_reward)
            agent_lengths.append(length)


# %% Visualizations

agents_name = [a.split('/')[1].split('.')[0].split('_policy')[0] for a in agents]

if MULTIPLE:
    
    if N_EPISODES > 1:

        # Total rewards
        width = 0.25 # set width of bar
        t_rewards = [[np.sum(np.array(agents_rewards).squeeze()[i][j]) for i in range(len(agents_rewards))] for j in range(N_EPISODES)]
        rs = []
        rs.append(np.arange(len(t_rewards)))
        for i in range(len(agents_name)):
            rs.append([x + 0.25 for x in rs[i]])
        for i in range(len(agents_name)):
            plt.bar(rs[i], np.array(t_rewards)[:,i], width=width, edgecolor='white', label=agents_name[i])
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.xticks([r + width for r in range(len(t_rewards))], [s for s in range(len(t_rewards))])
        plt.legend()
        plt.show()
        
        # Lengths
        width = 0.25 # set width of bar
        t_lengths = [[np.sum(np.array(agents_lengths).squeeze()[i][j]) for i in range(len(agents_lengths))] for j in range(N_EPISODES)]
        rs = []
        rs.append(np.arange(len(t_lengths)))
        for i in range(len(agents_name)):
            rs.append([x + 0.25 for x in rs[i]])
        for i in range(len(agents_name)):
            plt.bar(rs[i], np.array(t_lengths)[:,i], width=width, edgecolor='white', label=agents_name[i])
        plt.xlabel('Episodes')
        plt.ylabel('Episode length')
        plt.xticks([r + width for r in range(len(t_lengths))], [s for s in range(len(t_lengths))])
        plt.legend()
        plt.show()

    else:
        
        # Total reward        
        for i, a in enumerate(agents_name):
            plt.bar(i,np.sum(agents_rewards[i][0]), label=a)
        plt.xlabel('Agents')
        plt.ylabel('Total Reward')
        plt.xticks(np.arange(len(agents)), agents_name)
        plt.legend()
        plt.show()
        
        # Running reward
        for i, a in enumerate(agents_name):
            plt.plot(agents_rewards[i][0], label=a)
        plt.xlabel('Agents')
        plt.ylabel('Reward')
        plt.legend()
        plt.show()
        
        # Lengths
        for i, a in enumerate(agents_name):
            plt.bar(i, agents_lengths[i][0], label=a)
        plt.xlabel('Agents')
        plt.ylabel('Episode length')
        plt.xticks(np.arange(len(agents)), agents_name)
        plt.legend()
        plt.show()

else:

    if N_EPISODES > 1:
        plt.plot([item for sublist in total_rewards for item in sublist])
        plt.title(f'Episodes 1-{N_EPISODES}')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
    else:
        
        # Rewards
        plt.plot(agent_rewards[0])
        plt.title(f'Episode {CHRONIC_ID}')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        
        # Lengths
        plt.bar(agent_lengths[0])
        plt.title(f'Episode {CHRONIC_ID}')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')

# %%
