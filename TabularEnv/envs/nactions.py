import gym
from gym import spaces
import numpy as np
from gym.envs import registration


class NActions(gym.Env):
    """
    Environment that consists of L grids, episode length 1 step and L number of possible actions. 
    This is a simple testbed for tabular Soft Actor Critic / Soft Q Learning
    """

    metadata= {
        'render.modes': ['rgb_array'],
    }

    def __init__(self, L=100):
        self.L = L      # Length of the array. L is always an 
                        # odd number, and the agent always starts at Array[L/2].
        self.action_space = gym.spaces.Discrete(L) # actions : {0, 1, 2, ..., L}
        self.observation_space = gym.spaces.Box(low=0, high=L, shape=(1, 1), dtype=np.int8)

    def step(self, action):
        self.agent_pos = action + 1
        state = action + 1
        # state = self.agent_pos
        reward = self.reward()
        done = True # Only enable 1 step
        info = {}
        return state, reward, done, info
        

    def reset(self):
        self.agent_pos = 0
        return self.agent_pos

    def reward(self):
        return self.agent_pos

    def render(self, mode='rgb_array'):
        return self.agent_pos
        # return self.agent_pos_onehot()

    def state_action_encoder(self, state, action):
        # state: 0, 1, 2. both 1 and 2 are end states.
        return state*self.L + action