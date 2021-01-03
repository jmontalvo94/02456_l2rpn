#%% Imports
import grid2op
import numpy as np
import os
import torch
from buffer import ReplayBuffer
from collections import deque
from converter import ActObsConverter
from grid2op.Action import TopologySetAction
from grid2op.Converter import IdToAct
from grid2op.Parameters import Parameters
from grid2op.Reward import L2RPNReward
from lightsim2grid import LightSimBackend
from networks import DQN
from rewards import FlowLimitAndBlackoutReward
from utils import set_seed_everywhere, cli_train, obs_mask, get_max_values


#%% General parameters

args, general, params, nn_params, obs_params = cli_train()
NAME = args.name

PATH_TRAINED = 'trained_models/'
PATH_STATS = 'statistics/'

# NAME = 'dqn_1'

# general = {
#         "seed": 1,
#         "lightsim": True,
#         "checkpoint": 10,
#         "agent_type": "DQN",
#         "chunk_size": 288, # a day 24*60 / 5
#     }

# params = {
#         "n_episodes": 1,
#         "chronic_id": 1,
#         "batch_size": 64,
#         "buffer_cap": 100,
#         "gamma": 0.99,
#         "tau": 0.01
#     }

# nn_params ={
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

#%%

SEED = general['seed']
CHECKPOINT = general['checkpoint']
CHUNK_SIZE = general['chunk_size']
LIGHTSIM = general['lightsim']
AGENT_TYPE = general['agent_type']

if LIGHTSIM:
    backend = LightSimBackend() # faster ACOPF!

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Initialize converter
mask = obs_mask(env, obs_params)
max_values = get_max_values(env, mask)
converter = ActObsConverter(env.action_space, mask, max_values, IdToAct, change_bus_vect=False, set_line_status=False, change_line_status=False, redispatch=False)
converter.seed(SEED) # set seed


#%% Parameters

# Unpack parameters
N_EPISODES = params['n_episodes']
CHRONIC_ID = params['chronic_id']
BATCH_SIZE = params['batch_size']
BUFFER_CAP = params['buffer_cap']
GAMMA = params['gamma']
TAU = params['tau']
n_in = sum(mask) # observation vector size from mask
n_out = converter.action_space.n # action vector size from converter (includes do nothing action)

# Fix NN layers
layers = deque(nn_params['layers'])
layers.appendleft(n_in)
layers.append(n_out)
nn_params['layers'] = layers

# Instantiate networks and replay buffer
policy_net = DQN(nn_params)
target_net = DQN(nn_params)
target_net.load_state_dict(policy_net.state_dict()) # same weights as policy_net
buffer = ReplayBuffer(BUFFER_CAP)

# Move NN parameters to GPU (if available)
policy_net.to(device)
target_net.to(device)

#%% Experience replay buffer

while len(buffer) < BUFFER_CAP:
    env.set_id(CHRONIC_ID) # set for one chronic/episode only
    o = env.reset()
    o = converter.convert_obs(o) # convert object to numpy array
    while True:
        a = converter.action_space.sample()
        o1, r, done, info = env.step(converter.convert_act(a))
        o1 = converter.convert_obs(o1)
        buffer.push(o, a, r, o1, done)
        if not done:
            o = np.copy(o1)
        else:
            break


#%% Training loop

# TODO refractor code, no need to repeat training loop (adapt agent class to handle this)
if AGENT_TYPE == "DQN":

    epsilon = 1.0
    rewards, lengths, losses, epsilons, dones = [], [], [], [], []
    
    for i in range(N_EPISODES):

        env.set_id(CHRONIC_ID)
        o = env.reset()
        o = converter.convert_obs(o)
        length = 0
        ep_reward, ep_loss = [], []

        while True:
            
            # Select action with epsilon-greedy strategy
            if np.random.rand() < epsilon:
                a = converter.action_space.sample()
            else:
                with torch.no_grad():
                    a = policy_net(torch.tensor(o, device=device).float()).argmax().item()

            o1, r, done, info = env.step(converter.convert_act(a)) # act
            o1 = converter.convert_obs(o1)
            buffer.push(o, a, r, o1, done) # store transition in experience replay buffer

            # Sample from buffer
            batch = np.array(buffer.sample(BATCH_SIZE)) 
            o_batch, a_batch, r_batch, o1_batch , done_mask = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:, 4]

            # Convert and to Tensor (if needed)
            o_batch = torch.tensor(list(o_batch), device=device).float()
            a_batch = torch.tensor(a_batch.astype(int, copy=False), device=device)
            r_batch = torch.tensor(r_batch.astype(float, copy=False), device=device)
            o1_batch = torch.tensor(list(o1_batch), device=device).float()
            done_mask = done_mask.astype(bool, copy=False)

            policy_net.optimizer.zero_grad() # clean gradients

            # Compute policy Q-values (s, a) from observation batch
            Q = policy_net(o_batch).gather(1, a_batch.unsqueeze(1))
            
            # Compute target max a Q(s', a) from next observation batch
            Q1 = torch.zeros(BATCH_SIZE, device=device)
            Q1[~done_mask] = target_net(o1_batch[~done_mask]).max(1)[0].detach()
            
            # Compute expected target values for each sampled experience
            Q_target = r_batch + (GAMMA * Q1)

            # Update network weights
            loss = policy_net.loss(Q, Q_target.unsqueeze(1))
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1) # clamping gradients
            policy_net.optimizer.step()

            # Update target network parameters from policy network parameters
            target_net.update_params(policy_net.state_dict(), TAU)
            
            # Bookkeeping
            o = np.copy(o1)
            length += 1
            ep_reward.append(r)
            ep_loss.append(loss.item())

            if done:
                break

        # Bookkeeping
        epsilon *= N_EPISODES/(i/(N_EPISODES/20) + N_EPISODES) # decrease epsilon
        lengths.append(length)

        if (i+1) % CHECKPOINT == 0:
            epsilons.append(epsilon); rewards.append(ep_reward); losses.append(ep_loss)
            torch.save(policy_net.state_dict(), PATH_TRAINED + NAME + f'_policy_net_{i+1}.pth')
            torch.save(target_net.state_dict(), PATH_TRAINED + NAME + f'_target_net_{i+1}.pth')
            np.savez(PATH_STATS + NAME + f'_stats_{i+1}', epsilons=epsilons, rewards=rewards, lengths=lengths, losses=losses)

elif AGENT_TYPE == "DDQN":
    
    epsilon = 1.0
    rewards, lengths, losses, epsilons, dones = [], [], [], [], []
    
    for i in range(N_EPISODES):

        env.set_id(CHRONIC_ID)
        o = env.reset()
        o = converter.convert_obs(o)
        length = 0
        ep_reward, ep_loss = [], []

        while True:
            
            # Select action with epsilon-greedy strategy
            if np.random.rand() < epsilon:
                a = converter.action_space.sample()
            else:
                with torch.no_grad():
                    a = policy_net(torch.tensor(o, device=device).float()).argmax().item()

            o1, r, done, info = env.step(converter.convert_act(a)) # act
            o1 = converter.convert_obs(o1)
            buffer.push(o, a, r, o1, done) # store transition in experience replay buffer

            # Sample from buffer
            batch = np.array(buffer.sample(BATCH_SIZE)) 
            o_batch, a_batch, r_batch, o1_batch , done_mask = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:, 4]

            # Convert and to Tensor (if needed)
            o_batch = torch.tensor(list(o_batch), device=device).float()
            a_batch = torch.tensor(a_batch.astype(int, copy=False), device=device)
            r_batch = torch.tensor(r_batch.astype(float, copy=False), device=device)
            o1_batch = torch.tensor(list(o1_batch), device=device).float()
            done_mask = done_mask.astype(bool, copy=False)

            policy_net.optimizer.zero_grad() # clean gradients

            # Compute Q-values (s, a) and a' from policy net
            Q = policy_net(o_batch).gather(1, a_batch.unsqueeze(1))
            a_next = policy_net(o1_batch[~done_mask]).argmax(dim=1, keepdim=True)
            
            # Compute Q-values (s', a') from target net
            Q1 = torch.zeros(BATCH_SIZE, device=device)
            Q1[~done_mask] = target_net(o1_batch[~done_mask]).gather(1, a_next).squeeze(1).detach()
            
            # Compute expected target values for each sampled experience
            Q_target = r_batch + (GAMMA * Q1)

            # Update network weights
            loss = policy_net.loss(Q, Q_target.unsqueeze(1))
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1) # clamping gradients
            policy_net.optimizer.step()

            # Update target network parameters from policy network parameters
            target_net.update_params(policy_net.state_dict(), TAU)
            
            # Bookkeeping
            o = np.copy(o1)
            length += 1
            ep_reward.append(r)
            ep_loss.append(loss.item())

            if done:
                break

        # Bookkeeping
        epsilon *= N_EPISODES/(i/(N_EPISODES/20) + N_EPISODES) # decrease epsilon
        lengths.append(length)

        if (i+1) % CHECKPOINT == 0:
            epsilons.append(epsilon); rewards.append(ep_reward); losses.append(ep_loss)
            torch.save(policy_net.state_dict(), PATH_TRAINED + NAME + f'_policy_net_{i+1}.pth')
            torch.save(target_net.state_dict(), PATH_TRAINED + NAME + f'_target_net_{i+1}.pth')
            np.savez(PATH_STATS + NAME + f'_stats_{i+1}', epsilons=epsilons, rewards=rewards, lengths=lengths, losses=losses)
