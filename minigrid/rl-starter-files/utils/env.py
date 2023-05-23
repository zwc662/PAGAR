import gym, gym_minigrid


def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key)#, render_mode=render_mode)
    env.seed(seed)
    env.reset()
    return env

