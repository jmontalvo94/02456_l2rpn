#%% Imports
import grid2op
import re
from grid2op.Chronics import GridStateFromFileWithForecasts
from grid2op.Action import DontAct
from grid2op.PlotGrid import PlotMatplot
from grid2op.Parameters import Parameters
from grid2op.Action import DontAct, PowerlineChangeAction, TopologySetAction, TopologyChangeAction


#%% Plotting
obs = env.reset()
plot_helper = PlotMatplot(env.observation_space, width=960,height=540, line_id=False)
_ = plot_helper.plot_layout()
#_ = plot_helper.plot_info(line_values=env._thermal_limit_a, gen_values=env.gen_pmax, load_values=[el for el in range(env.n_load)])

#_ = plot_helper.plot_info()


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
n_configs = 0
print("\nSubstations information:")
for i, nb_el in enumerate(env.action_space.sub_info):
    print("On substation {} there are {} elements.".format(i, nb_el))
    n_configs += 2**(nb_el-1)


# %% Action space information

action_space = env.action_space

line_action = action_space({"set_line_status": [(3,1), (5,-1)], # connect line 3 and disconnect line 5
                             "change_line_status": [0,1] # change line status of line 0 and 1
                            })

reconnecting_line_1 = action_space.reconnect_powerline(line_id=1,bus_or=1,bus_ex=1) # is this slow?
reconfigure_substation_id_1 = action_space({"set_bus": {"substations_id": [(1, (1,2,2,1,1,2))]}})

action_space.grid_objects_types[13] # element id, total = 56. gives np.array([sub_id, load, gen, origin, extremity])

action_space.get_obj_connect_to(substation_id=1)