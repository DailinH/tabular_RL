import numpy as np
def rollout(env, agent, explore=False, EP_MAX=5, EPSILON=0.05):
    """
    is_training: disables exploration when this parameter is False
    returns a dataset of [s_t, a_t, s_tp1, r_t]
    env.render is omitted for training purposes.
    """
    dataset = []
    s_t = env.reset()
    s_tp1, r_t, done = 0, 0, False
    num_episodes = 0

    while num_episodes < EP_MAX:
        if done == True:
            num_episodes += 1
            s_t = env.reset()
            s_tp1, done = 0, False
            continue
        else:
            s_t = s_tp1
        a_t = agent.action(s_t)
        if explore:
            ep = np.random.uniform()
            if ep < EPSILON:
                a_t = np.random.randint(env.action_space.n)
        s_tp1, r_t, done, _ = env.step(a_t)
        dataset.append([s_t, a_t, s_tp1, r_t])
        # print("tuple ", s_t, a_t, s_tp1, r_t, done)
    return dataset