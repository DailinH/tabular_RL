import gym
from gym import spaces
import numpy as np
from gym.envs import registration


class NoisyChainWalk(gym.Env):
    """
    Environment that consists of L consecutive horizontal grids. 
    Agent starts from the leftmost grid and the goal is to reach
    the rightmost grid.
    """

    metadata= {
        'render.modes': ['rgb_array'],
    }

    def __init__(self, L=5):
        self.L = L      # Length of the array. L is always an 
                        # odd number, and the agent always starts at Array[L/2].
        self.action_space = gym.spaces.Discrete(2) # we allow two actions: 0 (left) and 1 (right).
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(L, 1), dtype=np.int8)
        self.obs_array = np.zeros(shape=(1, L), dtype=np.int8)
        self.agent_pos = 0
        self.max_step = L
        self.step_cnt = 0
        self.states = None

    def step(self, action):
        self.step_cnt += 1
        if action == 0:
            self.agent_pos += ( -1 if self.agent_pos > 0 else 0)
        else:
            self.agent_pos += 1
        # state = self.agent_pos_onehot()
        state = self.agent_pos
        reward = self.reward()
        done = (True if ((self.agent_pos == self.L - 1) or (self.step_cnt == self.max_step)) else False)
        info = {}
        print(self.agent_pos)
        return state, reward, done, info
        

    def reset(self):
        self.states = np.arange(self.L, dtype=np.int8)
        self.step_cnt = 0
        self.agent_pos = 0
        return self.agent_pos
        # return self.agent_pos_onehot()

    def reward(self):
        noise = np.random.normal(0, 1)
        # noise = 0
        r = (1 if (self.agent_pos == self.L - 1) else -0.1)
        return r # + noise

    def render(self, mode='rgb_array'):
        return self.agent_pos
        # return self.agent_pos_onehot()

    def agent_pos_onehot(self):
        obs = np.zeros((self.L, 1))
        obs[self.agent_pos][0] = 1
        return obs

    def state_action_encoder(self, state, action):
        return state + self.L * action