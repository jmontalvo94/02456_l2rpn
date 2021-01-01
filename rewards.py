import numpy as np
from grid2op.dtypes import dt_float
from grid2op.Reward import BaseReward

class FlowLimitAndBlackoutReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)

    def initialize(self, env):
        self.reward_min = dt_float(-env.backend.n_line*0.95)
        self.reward_max = dt_float(env.backend.n_line*0.95)
        self.reward_none = dt_float(0.0)

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if not is_done and not has_error:
            relative_flow = self.__get_lines_capacity_usage(env) # (flow/max capacity)
            limit_diff = dt_float(0.95) - relative_flow
            reward = np.sum((limit_diff**2) * np.sign(limit_diff))
        elif is_done and not has_error:
            reward = self.reward_max * (24*60 / 5) # completed!
        elif has_error:
            reward = self.reward_min * (24*60 / 5) # blackout for a day
        else:
            reward = self.reward_none
        return reward

    @staticmethod
    def __get_lines_capacity_usage(env):
        ampere_flows = np.abs(env.backend.get_line_flow(), dtype=dt_float)
        thermal_limits = np.abs(env.get_thermal_limit(), dtype=dt_float)
        thermal_limits += 1e-5  # for numerical stability
        relative_flow = np.divide(ampere_flows, thermal_limits, dtype=dt_float)
        return relative_flow