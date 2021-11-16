import gym
from gym import spaces
import numpy as np
from gym.envs import registration


class NoisyBinTree(gym.Env):
    """
    Environment that consists of 3 layer binary (7 states) tree with Gaussian noise over the reward.
    """

    metadata= {
        'render.modes': ['rgb_array'],
    }

    def __init__(self):
        self.L = 7
        self.action_space = gym.spaces.Discrete(2) # we allow two actions: 0 (left), 1 (right), from parent to children nodes.
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8)
        self.agent_pos = 0
        self.step_cnt = 0
        self.states = [0, 1, 2, 3, 4, 5, 6]

    def step(self, action):
        self.step_cnt += 1
        _left = {0: 1, 1: 3, 2: 5}
        _right = {0: 2, 1: 4, 2: 6}

        if action == 0:
            self.agent_pos = _left[self.agent_pos]
        else:
            self.agent_pos = _right[self.agent_pos]
        state = self.agent_pos
        reward = self.reward()
        done = True if (self.agent_pos >= 3) else False
        info = None
        return state, reward, done, info
        

    def reset(self):
        # self.states = np.arange(self.L, dtype=np.int8)
        self.step_cnt == 0
        self.agent_pos = 0
        return self.agent_pos

    def reward(self):
        if self.agent_pos <= 2:
            return 0
        noise = np.random.normal(0, 1)
        if self.agent_pos == 3:
            return -1 + noise
        elif self.agent_pos == 6:
            return 1 + noise
        else:
            return 0 + noise

    def render(self, mode='rgb_array'):
        return self.agent_pos