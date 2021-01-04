#%% Imports

import gym

#%%

env = wrappers.Monitor(env, "./gym-results", force=True) 
env.reset()
for _ in range(100):
    s, a, done, info = env.step(env.action_space.sample()) # take a random action
    if done: break
env.close()
show_replay()

#%% Imports
import gym
import numpy as np
import os
import torch
from buffer import ReplayBuffer
from collections import deque
from networks import DQN
from utils import set_seed_everywhere, cli_train_ll


#%% General parameters

args, general, params, nn_params = cli_train_ll()
NAME = args.name

PATH_TRAINED = 'trained_models/'
PATH_STATS = 'statistics/'

# NAME = 'dqn_1'

# general = {
#         "seed": 1,
#         "checkpoint": 10,
#         "agent_type": "DQN"
#     }

# params = {
#         "n_episodes": 1,
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


SEED = general['seed']
AGENT_TYPE = general['agent_type']
CHECKPOINT = general['checkpoint']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Initialization

env = gym.make('LunarLander-v2')


#%% Parameters

# Unpack parameters
N_EPISODES = params['n_episodes']
BATCH_SIZE = params['batch_size']
BUFFER_CAP = params['buffer_cap']
GAMMA = params['gamma']
TAU = params['tau']
n_in = env.observation_space # observation vector size from mask
n_out = env.action_space.n # action vector size from converter (includes do nothing action)

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
