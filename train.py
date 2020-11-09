#%% Imports
import grid2op
import re
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Action import DontAct
from grid2op.PlotGrid import PlotMatplot

#%% Create environment
# no opponent no maintenance
env = grid2op.make("l2rpn_neurips_2020_track1_small", 
                                     data_feeding_kwargs= {"gridvalueClass": GridStateFromFileWithForecasts}, # disable maintenance
                                     opponent_init_budget=0, opponent_budget_per_ts=0, opponent_action_class=DontAct  # disable opponent
                                     )

# # no opponent, but maintenance
# env = grid2op.make("l2rpn_neurips_2020_track1_small", 
#                                           opponent_init_budget=0, opponent_budget_per_ts=0, opponent_action_class=DontAct  # disable opponent
#                                           )

# # regular env
# env = grid2op.make("l2rpn_neurips_2020_track1_small")

#%% Plotting
obs = env.reset()
plot_helper = PlotMatplot(env.observation_space, width=1920,height=1080, line_id=False)
#_ = plot_helper.plot_layout()
_ = plot_helper.plot_info(line_values=env._thermal_limit_a, gen_values=env.gen_pmax, load_values=[el for el in range(env.n_load)])

# %% Grid information

# Interconnections modelled as loads (positive or negative)
[el for el in env.name_load if re.match(".*interco.*", el) is not None]

# Loads (positive)
[el for el in env.name_load if re.match(".*load.*", el) is not None]

# Load ids
print("\nInjection information:")
load_to_subid = env.action_space.load_to_subid
print ('There are {} loads connected to substations with id: {}'.format(len(load_to_subid), load_to_subid))

# Generators ids
gen_to_subid = env.action_space.gen_to_subid
print ('There are {} generators, connected to substations with id: {}'.format(len(gen_to_subid), gen_to_subid))

# Line id sender
print("\nPowerline information:")
line_or_to_subid = env.action_space.line_or_to_subid
line_ex_to_subid = env.action_space.line_ex_to_subid
print ('There are {} transmissions lines on this grid. They connect:'.format(len(line_or_to_subid)))
for line_id, (ori, ext) in enumerate(zip(line_or_to_subid, line_ex_to_subid)):
    print("Line with id {} connects: substation origin id {} to substation extremity id {}".format(line_id, ori, ext))

# Num of elements per SE
print("\nSubstations information:")
for i, nb_el in enumerate(env.action_space.sub_info):
    print("On substation {} there are {} elements.".format(i, nb_el))


#%% Training

num_episodes = 1000
episode_limit = 100
batch_size = 64
learning_rate = 0.005
gamma = 0.99 # discount rate
tau = 0.01 # target network update rate
replay_memory_capacity = 10000
prefill_memory = True
val_freq = 100 # validation frequency

n_inputs = env.observation_space.n
n_outputs = env.action_space.n

# initialize DQN and replay memory
policy_dqn = DQN(n_inputs, n_outputs, learning_rate)
target_dqn = DQN(n_inputs, n_outputs, learning_rate)
target_dqn.load_state_dict(policy_dqn.state_dict())

replay_memory = ReplayMemory(replay_memory_capacity)

# prefill replay memory with random actions
if prefill_memory:
    print('prefill replay memory')
    s = env.reset()
    while replay_memory.count() < replay_memory_capacity:
        a = env.action_space.sample()
        s1, r, d, _ = env.step(a)
        replay_memory.add(s, a, r, s1, d)
        s = s1 if not d else env.reset()
        
# training loop
try:
    print('start training')
    epsilon = 1.0
    rewards, lengths, losses, epsilons = [], [], [], []
    for i in range(num_episodes):
        # init new episode
        s, ep_reward, ep_loss = env.reset(), 0, 0
        for j in range(episode_limit):
            # select action with epsilon-greedy strategy
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = policy_dqn(torch.from_numpy(one_hot([s], n_inputs)).float()).argmax().item()
            # perform action
            s1, r, d, _ = env.step(a)
            # store experience in replay memory
            replay_memory.add(s, a, r, s1, d)
            # batch update
            if replay_memory.count() >= batch_size:
                # sample batch from replay memory
                batch = np.array(replay_memory.sample(batch_size), dtype=int)
                ss, aa, rr, ss1, dd = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
                # do forward pass of batch
                policy_dqn.optimizer.zero_grad()
                Q = policy_dqn(torch.from_numpy(one_hot(ss, n_inputs)).float())
                # use target network to compute target Q-values
                with torch.no_grad():
                    # TODO: use target net
                    Q1 = target_dqn(torch.from_numpy(one_hot(ss1, n_inputs)).float())
                # compute target for each sampled experience
                q_targets = Q.clone()
                for k in range(batch_size):
                    q_targets[k, aa[k]] = rr[k] + gamma * Q1[k].max().item() * (not dd[k])
                # update network weights
                loss = policy_dqn.loss(Q, q_targets)
                loss.backward()
                policy_dqn.optimizer.step()
                # update target network parameters from policy network parameters
                target_dqn.update_params(policy_dqn.state_dict(), tau)
            else:
                loss = 0
            # bookkeeping
            s = s1
            ep_reward += r
            ep_loss += loss.item()
            if d: break
        # bookkeeping
        epsilon *= num_episodes/(i/(num_episodes/20)+num_episodes) # decrease epsilon
        epsilons.append(epsilon); rewards.append(ep_reward); lengths.append(j+1); losses.append(ep_loss)
        if (i+1) % val_freq == 0: print('%5d mean training reward: %5.2f' % (i+1, np.mean(rewards[-val_freq:])))
    print('done')
except KeyboardInterrupt:
    print('interrupt')