from gym.envs.registration import register

register(
    id='noisychainwalk-v0',
    entry_point='TabularEnv.envs:NoisyChainWalk'
)


register(
    id='noisybintree-v0',
    entry_point='TabularEnv.envs:NoisyBinTree'
)

register(
    id='nactions-v0',
    entry_point='TabularEnv.envs:NActions'
)


register(
    id='emptyroom-v0',
    entry_point='TabularEnv.envs:EmptyRoom'
)
