#%% Imports
import grid2op
from grid2op.Action import TopologySetAction
from grid2op.Converter import IdToAct
from grid2op.Parameters import Parameters
from lightsim2grid import LightSimBackend
from agent import dqn_agent
from utils import set_seed_everywhere, cli
from buffer import ReplayBuffer
from networks import DQN


#%% General parameters

args, general, training, nn_params = cli()

SEED = 123
CHUNK_SIZE = 100
N_EPISODES = 1
CHRONIC_ID = 1
BATCH_SIZE = 64
BUFFER_CAP = 10000
GAMMA = 0.99
TAU = 0.01

p = Parameters()

# No authomatic disconnection
p.HARD_OVERFLOW_THRESHOLD = 9999

# if LIGHTSIM:
backend = LightSimBackend() # faster ACOPF!

#%% Initialize environment

env = grid2op.make('rte_case14_realistic', backend=backend, action_class=TopologySetAction, param=p)

env.seed(SEED) # set seed
set_seed_everywhere(SEED) # set seed

env.deactivate_forecast() # no forecast or simulation, faster calculations

env.set_chunk_size(CHUNK_SIZE) # to avoid loading all the episode and fill memory

# Initialize agent with action converter
agent = dqn_agent(env.action_space, IdToAct, change_bus_vect=False, set_line_status=False, change_line_status=False, redispatch=False)
agent.seed(SEED) # set seed

#%% Training

# # Training loop
# for i in range(N_EPISODES):

#     # Set environment to specific chronic
#     env.set_id(CHRONIC_ID)

#     obs = env.reset()

#     # Initialize variables
#     reward = 0
#     done = False
#     total_reward = []

#     # Play episode
#     while True:
#        action = agent.act(obs, reward, done)
#        obs, reward, done, info = env.step(action)
#        total_reward.append(reward)
#        if done:
#            break # if episode is over or game over

# env.close()
# print("The total reward was {:.2f}".format(total_reward))