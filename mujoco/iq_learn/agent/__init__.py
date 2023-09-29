import gym
from agent.sac import SAC
from agent.sac_ppo import SAC_PPO
from agent.softq import SoftQ
from agent.softq_ppo import SoftQ_PPO


def make_agent(env, name, args):
    obs_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, gym.spaces.discrete.Discrete):
        print('--> Using Soft-Q agent')
        action_dim = env.action_space.n
        # TODO: Simplify logic
        getattr(args, name).obs_dim = obs_dim
        getattr(args, name).action_dim = action_dim
        if name == 'protagonist':
            agent = SoftQ_PPO(obs_dim, action_dim, args.train.batch, args)
        else:
            agent = SoftQ(obs_dim, action_dim, args.train.batch, args)
    else:
        print('--> Using SAC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        # TODO: Simplify logic
        getattr(args, name).obs_dim = obs_dim
        getattr(args, name).action_dim = action_dim
        if name == 'protagonist':
            agent = SAC_PPO(obs_dim, action_dim, action_range, args.train.batch, args)
        else:
            agent = SAC(obs_dim, action_dim, action_range, args.train.batch, args)
    agent.name = name
    return agent


 