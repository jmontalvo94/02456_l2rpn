import torch
from grid2op.Agent import AgentWithConverter
from networks import DQN

class DQNAgent(AgentWithConverter):
    """Agent that converts observation, selects an encoded action with a NN, 
    and converts back the action into an interpretable action by the environment.
    
        Args:
            action_space: action space from the environment
            converter: object of class Converter
            kwargs: arguments to initialize the converter
    """
    def __init__(self, action_space, mask, max_values, converter, nn_params, path, **kwargs):
        AgentWithConverter.__init__(self, action_space, converter, **kwargs)
        self.mask = mask
        self.max_values = max_values
        self.neural_network = DQN(nn_params)
        self.neural_network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def convert_obs(self, observation):
        transformed_observation = (torch.tensor(observation.to_vect()[self.mask]))/self.max_values
        return transformed_observation

    def my_act(self, transformed_observation, reward, done=False):
        act_predicted = self.neural_network(transformed_observation).argmax().item()
        return act_predicted