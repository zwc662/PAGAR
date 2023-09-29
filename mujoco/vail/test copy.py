import os
import gym
import torch
import argparse
import numpy as np
import pickle

from model import Actor, Critic
from utils.utils import get_action
from utils.zfilter import ZFilter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Hopper-v2", 
                    help='name of the environment to run')
parser.add_argument('--iter', type=int, default=10,
                    help='number of episodes to play')
parser.add_argument('--at_least', type=int, default=2000,
                    help='select trajectories of at least score_{args.env_name}')
parser.add_argument('--load_model', type=str, default=None, 
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False, 
                    help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99, 
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98, 
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size', type=int, default=64, 
                    help='hidden unit size of actor, critic networks (default: 64)')
parser.add_argument('--learning_rate', type=float, default=3e-4, 
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--model_update_num', type=int, default=10, 
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=2048, 
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch size to update (default: 64)')
parser.add_argument('--max_iter_num', type=int, default=12000,
                    help='maximal number of main iterations (default: 12000)')
parser.add_argument('--seed', type=int, default=500,
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()


if __name__ == "__main__":
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

    print("state size: ", num_inputs)
    print("action size: ", num_actions)

    actor = Actor(num_inputs, num_actions, args)
    critic = Critic(num_inputs, args)

    running_state = ZFilter((num_inputs,), clip=5)
    
    if args.load_model is not None:
        pretrained_model_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))

        pretrained_model = torch.load(pretrained_model_path)

        actor.load_state_dict(pretrained_model['actor'])
        critic.load_state_dict(pretrained_model['critic'])

        running_state.rs.n = pretrained_model['z_filter_n']
        running_state.rs.mean = pretrained_model['z_filter_m']
        running_state.rs.sum_square = pretrained_model['z_filter_s']

        print("Loaded OK ex. ZFilter N {}".format(running_state.rs.n))

    else:
        assert("Should write pretrained filename in save_model folder. ex) python3 test_algo.py --load_model ppo_max.tar")

    demonstrations = []
    actor.eval(), critic.eval()
    episode = 0
    num_demos = 0
    while True:
        demonstration = []
        state = env.reset()
        norm_state = running_state(state)
        steps = 0
        score = 0
        for _ in range(10000):
            demonstration.append([])
            demonstration[-1].append(np.asarray(state).flatten())
            if args.render:     
                env.render()
            mu, std = actor(torch.Tensor(norm_state).unsqueeze(0))
            action = get_action(mu, std)[0]
           
            demonstration[-1].append(np.asarray(action).flatten())
            
            next_state, reward, done, _ = env.step(action)
 
            demonstration[-1].append(np.asarray([reward]))
            demonstration[-1].append(np.asarray([1. - float(done)]))
            
            norm_next_state = running_state(next_state)
            
            norm_state = norm_next_state

            state = next_state
            score += reward
            
            if done:
                print("{} cumulative reward: {}".format(episode, score))
                if score >= int(args.at_least):
                    demonstrations += demonstration
                    num_demos += 1
                break
        if num_demos == args.iter:
            with open(f'expert_demo_{args.env_name}.p', 'wb') as f:        
                pickle.dump(demonstrations, f)
            
            exit(0)

        episode += 1