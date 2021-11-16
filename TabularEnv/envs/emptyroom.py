import gym
from gym import spaces
import numpy as np
from gym.envs import registration


class EmptyRoom(gym.Env):
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
        self.action_space = gym.spaces.Discrete(4) # we allow 4 actions: 0 (up) and 1 (right), 2 (down), 3(left). Optimal solutions will always be 1 and 2.
        # self.observation_space = gym.spaces.Box(low=0, high=1, shape=(L, L), dtype=np.int8)
        self.observation_space = gym.spaces.Box(low=0, high=L, shape=(1, 1), dtype=np.int8) # return agent position
        self.agent_pos = [0, 0]
        self.step_cnt = 0
        self.max_step = L**2

    def step(self, action):
        self.step_cnt += 1
        # state = self.agent_pos_onehot()
        if action == 0:
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:
            self.agent_pos[1] = min(self.L - 1, self.agent_pos[1] + 1)
        elif action == 2:
            self.agent_pos[0] = min(self.L - 1, self.agent_pos[0] + 1)
        elif action == 3:
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        state = self.agent_pos[0] * self.L + self.agent_pos[1]
        reward = self.reward()
        done = (True if ((state == (self.L ** 2 - 1)) or (self.step_cnt == self.max_step)) else False)
        info = {}
        return state, reward, done, info
        

    def reset(self):
        self.agent_pos = [0, 0]
        self.step_cnt = 0
        return self.agent_pos
        # return self.agent_pos_onehot()

    def reward(self):
        if self.agent_pos[0] == self.L - 1 and self.agent_pos[1] == self.L - 1:
            return 5
        else:
            return -0.1

    def render(self, mode='rgb_array'):
        return self.agent_pos

    def agent_pos_onehot(self):
        obs = np.zeros((self.L, 1))
        obs[self.agent_pos][0] = 1
        return obs

    def state_action_encoder(self, state, action):
        return state * 4 + action