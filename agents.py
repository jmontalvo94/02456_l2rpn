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
    def __init__(self, action_space, converter, **kwargs): # path=None
        AgentWithConverter.__init__(self, action_space, converter, **kwargs)
        # self.neural_network = DQN()
        # self.neural_network.load(path)

    def convert_obs(self, observation):
        transformed_observation = observation.to_vect()[:330] # hard-coded
        return transformed_observation

    def my_act(self, transformed_observation, reward, done=False):
        # act_predicted = self.neural_network(transformed_observation)
        # return act_predicted
        pass