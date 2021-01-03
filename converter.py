import torch
from grid2op.Agent import AgentWithConverter
from networks import DQN

class ActObsConverter(AgentWithConverter):
    """Class that converts and observation, extends converter IdToAct with convert_obs function.
    
        Args:
            action_space: action space from the environment
            converter: object of class Converter
            kwargs: arguments to initialize the converter
    """
    def __init__(self, action_space, mask, max_values, converter, **kwargs):
        AgentWithConverter.__init__(self, action_space, converter, **kwargs)
        self.mask = mask
        self.max_values = max_values

    def convert_obs(self, observation):
        transformed_observation = (observation.to_vect()[self.mask])/self.max_values
        return transformed_observation

    def my_act(self, transformed_observation, reward, done=False):
        return None