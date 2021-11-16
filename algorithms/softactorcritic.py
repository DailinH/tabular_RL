from . import agents
import numpy as np
import copy

class SoftActorCriticAgent(agents.Agent):
    def __init__(self, obs_space, action_space, state_action_encoder, beta=100):
        self.name = 'sac'
        self.obs_space = obs_space
        self.action_space = action_space
        # self.model = np.zeros(((np.prod(obs_space.shape) + 1) * action_space.n,))
        self.model = np.zeros(((action_space.n + 1) * action_space.n,))
        self.target_model = copy.deepcopy(self.model)
        self.state_action_encoder = state_action_encoder
        self.beta = 10


    def action(self, obs):
        _max_val = min(self.model)
        act = 0
        for a in range(self.action_space.n):
            encode_val = self.model[self.state_action_encoder(obs, a)]
            if encode_val > _max_val:
                _max_val = encode_val
                act = a
        return act

    def update(self, train_batch, lr = 1, gamma = 0.99):
        beta = self.beta
        # print(self.model)
        for data in train_batch:
            s_t, a_t, s_tp1, r = data
            # self.visualize()
            # let's surely hope softmax doesn't overflow! Nah I'll add the tricks to be safe :)))
            # delta = np.max([self.model[self.state_action_encoder(s_t, a)] for a in range(self.action_space.n)])
            # pi = np.exp(self.model[self.state_action_encoder(s_t, a_t)] - delta) / np.sum(np.exp([self.model[self.state_action_encoder(s_t, a)] for a in range(self.action_space.n)] - delta))
            raw_pi_tp1 = [self.model[self.state_action_encoder(s_tp1, a)] for a in range(self.action_space.n)]
            delta = np.max(raw_pi_tp1)
            a_tp1 = raw_pi_tp1.index(max(raw_pi_tp1))
            pi = np.exp(self.model[self.state_action_encoder(s_tp1, a_tp1)] - delta) / np.sum(np.exp(raw_pi_tp1 - delta))
            V = self.model[self.state_action_encoder(s_tp1, a_tp1)] - (1/beta) * np.log(pi)
            self.model[self.state_action_encoder(s_t, a_t)] += lr*(r + gamma * V - self.model[self.state_action_encoder(s_t, a_t)])
            # self.model[self.state_action_encoder(s_t, a_t)] += lr * (r + gamma/beta * 
            # np.log(np.sum((1/self.action_space.n)*(np.exp(beta * np.asarray([self.model[self.state_action_encoder(s_tp1, a)] for a in range(self.action_space.n)])))))
            #  - self.model[self.state_action_encoder(s_t, a_t)])

    def visualize(self, L = 5):
        for x in range(L):
            for y in range(L):
                s_t = x * L + y
                val = [self.model[self.state_action_encoder(s_t, a_t)] for a_t in range(4)]
                a_t = val.index(max(val))
                print(a_t, end=",")
            print('\n', end="")
        print('\n')
                # print('state {}'.format(s_t), ','.join())

    def update_target(self, tau=0.01):
        self.target_model = (1- tau) * self.target_model + tau * self.model

