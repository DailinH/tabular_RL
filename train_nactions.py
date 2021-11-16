print("aa?")
from rollout import rollout
import numpy as np
import gym
from algorithms import QLearningAgent, SoftQLearningAgent, SoftActorCriticAgent
import random
from utils import toCSV

R_ = []
print("start")
for k in range(100):
    print(k, end='\r')

    env = gym.make("TabularEnv:nactions-v0")
    # env.L = 10

    # agent = QLearningAgent(
    #     action_space=env.action_space,
    #     obs_space=env.observation_space,
    #     state_action_encoder=env.state_action_encoder
    # )
    # agent = SoftQLearningAgent(
    #     action_space=env.action_space,
    #     obs_space=env.observation_space,
    #     state_action_encoder=env.state_action_encoder
    # )
    agent = SoftActorCriticAgent(
        action_space=env.action_space,
        obs_space=env.observation_space,
        state_action_encoder=env.state_action_encoder
    )
    agent.model = np.zeros(((env.action_space.n + 1) * env.action_space.n,))
    agent.beta = 1000
    # train_batch_size = 32
    memory_size = 10000
    R = []
    replay_buffer = []
    for i in range(1000):
        replay_buffer.append(rollout(env, agent, explore=True, EP_MAX=5, EPSILON=0.05))
        replay_buffer = replay_buffer[:memory_size]
        random.shuffle(replay_buffer)
        agent.update(replay_buffer[0], lr=0.1)
        test_data = np.array(rollout(env, agent, explore=False, EP_MAX=1))
        r_sum = sum(test_data[:, -1])
        R.append(r_sum)
    R = np.asarray(R)
    # print(R)
    R_.append(R)
R_ = np.mean(R_, axis=0)
print(R_)
toCSV(R_, ['Episode','Reward'], 'csv/{}actions_{}_{}.csv'.format(env.L, agent.name, agent.beta))

