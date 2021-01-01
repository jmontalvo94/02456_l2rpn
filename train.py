#%% Imports
import grid2op
import numpy as np
import os
import torch
from agents import DQNAgent
from buffer import ReplayBuffer
from collections import deque
from grid2op.Action import TopologySetAction
from grid2op.Converter import IdToAct
from grid2op.Parameters import Parameters
from grid2op.Reward import L2RPNReward
from lightsim2grid import LightSimBackend
from networks import DQN
from rewards import FlowLimitAndBlackoutReward
from utils import set_seed_everywhere, cli


#%% General parameters

# args, general, training, nn_params = cli()
# NAME = args.name

NAME = 'test'

os.mkdir(NAME)

general = {
        "seed": 1,
        "lightsim": True,
        "checkpoint": 10,
        "chunk_size": 288, # a day 24*60 / 5
    }

training = {
        "n_episodes": 10,
        "chronic_id": 1,
        "batch_size": 64,
        "buffer_cap": 100,
        "gamma": 0.99,
        "tau": 0.001
    }

nn_params ={
        "optimizer": "ADAM",
        "layers": [330, 330, 330],
        "learning_rate": 0.001,
        "weight_decay": 0
    }

SEED = general['seed']
CHECKPOINT = general['checkpoint']
CHUNK_SIZE = general['chunk_size']
LIGHTSIM = general['lightsim']

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

# Initialize empty agent
agent = DQNAgent(env.action_space, IdToAct, change_bus_vect=False, set_line_status=False, change_line_status=False, redispatch=False)
agent.seed(SEED) # set seed

#%% Training

# Unpack parameters
N_EPISODES = training['n_episodes']
CHRONIC_ID = training['chronic_id']
BATCH_SIZE = training['batch_size']
BUFFER_CAP = training['buffer_cap']
GAMMA = training['gamma']
TAU = training['tau']
n_in = 330 # hard-coded observation vector size to remove unused variables
n_out = 151 # hard-coded action vector size for fixed action space

# Fix NN layers
layers = deque(nn_params['layers'])
layers.appendleft(n_in)
layers.append(n_out)
nn_params['layers'] = layers

# Initialize networks and replay buffer
policy_net = DQN(nn_params)
target_net = DQN(nn_params)
target_net.load_state_dict(policy_net.state_dict()) # same weights as policy_net
buffer = ReplayBuffer(BUFFER_CAP)

# Move NN parameters to GPU
policy_net.to(device)
target_net.to(device)

#%%

# Fill experience replay buffer
while len(buffer) < BUFFER_CAP:
    env.set_id(CHRONIC_ID) # set for one chronic only
    o = env.reset()
    while True:
        a = agent.action_space.sample()
        o1, r, done, info = env.step(agent.convert_act(a))
        buffer.push(o, a, r, o1, done)
        if not done:
            o = o1
        else:
            break

#%%

# Training loop
epsilon = 1.0
rewards, lengths, losses, epsilons, dones = [], [], [], [], []
for i in range(N_EPISODES):

    env.set_id(CHRONIC_ID)
    o, ep_reward, ep_loss = env.reset(), 0, 0
    l = 0

    while True:
        
        # Select action with epsilon-greedy strategy
        if np.random.rand() < epsilon:
            a = agent.action_space.sample()
        else:
            with torch.no_grad():
                a = policy_net(torch.tensor(agent.convert_obs(o), device=device)).argmax().item()

        o1, r, done, info = env.step(agent.convert_act(a)) # act        
        buffer.push(o, a, r, o1, done) # store experience in experience replay buffer

        # Sample from buffer
        batch = np.array(buffer.sample(BATCH_SIZE)) 
        o_batch, a_batch, r_batch, o1_batch , done_mask = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:, 4]

        # Convert and to Tensor (if needed)
        o_batch = torch.tensor(list(map(agent.convert_obs, o_batch)), device=device)
        a_batch = torch.tensor(a_batch.astype(int, copy=False), device=device)
        r_batch = torch.tensor(r_batch.astype(float, copy=False), device=device)
        o1_batch = torch.tensor(list(map(agent.convert_obs, o1_batch)), device=device)
        done_mask = done_mask.astype(bool, copy=False)

        # Compute policy Q-values (s, a) from observation batch
        policy_net.optimizer.zero_grad() # clean gradients
        Q = policy_net(o_batch).gather(1, a_batch.unsqueeze(1))
        
        # Compute target V-values (s) from next observation batch
        V1 = torch.zeros(BATCH_SIZE, device=device)
        V1[~done_mask] = target_net(o1_batch[~done_mask]).max(1)[0].detach()
        
        # Compute expected target values for each sampled experience
        # Q_target = Q.clone()
        # for k in range(BATCH_SIZE):
        #     Q_target[k, a_batch[k]] = r_batch[k] + gamma * Q1[k].max().item() * (not done_mask[k])
        Q_target = r_batch + (V1 * GAMMA)

        # Update network weights
        loss = policy_net.loss(Q, Q_target.unsqueeze(1))
        loss.backward()
        policy_net.optimizer.step()

        # Update target network parameters from policy network parameters
        target_net.update_params(policy_net.state_dict(), TAU)
        
        # Bookkeeping
        o = o1
        ep_reward += r
        l += 1
        ep_loss += loss.item()

        if done:
            break

    # Bookkeeping
    epsilon *= N_EPISODES/(i/(N_EPISODES/20) + N_EPISODES) # decrease epsilon
    epsilons.append(epsilon); rewards.append(ep_reward); lengths.append(l); losses.append(ep_loss)

    if (i+1) % CHECKPOINT == 0:
        torch.save(policy_net.state_dict(), NAME+f'/policy_net_{i+1}.pth')
        torch.save(target_net.state_dict(), NAME+f'/target_net_{i+1}.pth')
        np.savez(NAME+f'/stats_{i+1}.pth', epsilons=np.copy(epsilons), rewards=np.copy(rewards), lengths=np.copy(lengths), losses=np.copy(losses))

# %%
