import os
import gym
import pickle
import argparse
import numpy as np
from collections import deque
from datetime import datetime
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 
import pickle
from utils.utils import *
from utils.zfilter import ZFilter
from model import Actor, Critic, Discriminator, Q_Value
from train_model import train_actor_critic, train_reward_function, train_protagonist_actor_critic, train_antagonist_actor_critic

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


parser = argparse.ArgumentParser(description='PyTorch GAIL')
parser.add_argument('--env_name', type=str, default="Hopper-v2", 
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None, 
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False, 
                    help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99, 
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98, 
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--num_layers', type=int, default=2, 
                    help='number of linear layers (default: 2)')
parser.add_argument('--hidden_size', type=int, default=100, 
                    help='hidden unit size of actor, critic and reward_function networks (default: 100)')
parser.add_argument('--learning_rate', type=float, default=3e-4, 
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3, 
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--pair_coef', type=float, default=1, 
                    help='pair loss coefficient (default: 1)')
parser.add_argument('--irl_coef', type=float, default=1e3, 
                    help='pair loss coefficient (default: 1e3)')
parser.add_argument('--ppo_coef', type=float, default=1., 
                    help='ppo (trpo) loss coefficient (default: 1)')
parser.add_argument('--constraint_coef', type=float, default=1., 
                    help='constraint loss coefficient (default: 1)')
parser.add_argument('--entropy_coef', type=float, default=1., 
                    help='pair entropy loss coefficient (default: 1)')
parser.add_argument('--clip_param', type=float, default=0.2, 
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--reward_function_update_num', type=int, default=2, 
                    help='update number of reward_function (default: 2)')
parser.add_argument('--actor_critic_update_num', type=int, default=10, 
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=2048, 
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.8,
                    help='accuracy for suspending reward_function about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.8,
                    help='accuracy for suspending reward_function about generated data (default: 0.8)')
parser.add_argument('--max_iter_num', type=int, default=4000,
                    help='maximal number of main iterations (default: 4000)')
parser.add_argument('--seed', type=int, default=500,
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()


def main():
    

    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    running_state = ZFilter((num_inputs,), clip=5)

    print('state size:', num_inputs) 
    print('action size:', num_actions)

    protagonist_actor = Actor(num_inputs, num_actions, args)
    protagonist_critic = Critic(num_inputs, args)

    antagonist_actor = Actor(num_inputs, num_actions, args)
    antagonist_critic = Critic(num_inputs, args)

    reward_function = Discriminator(num_inputs + num_actions, args)
    q_value = Q_Value(num_inputs, num_actions, args)

    protagonist_actor_optim = optim.Adam(protagonist_actor.parameters(), lr=args.learning_rate)
    protagonist_critic_optim = optim.Adam(protagonist_critic.parameters(), lr=args.learning_rate, 
                              weight_decay=args.l2_rate) 
    antagonist_actor_optim = optim.Adam(antagonist_actor.parameters(), lr=args.learning_rate)
    antagonist_critic_optim = optim.Adam(antagonist_critic.parameters(), lr=args.learning_rate, 
                              weight_decay=args.l2_rate) 
    
    reward_function_optim = optim.Adam(reward_function.parameters(), lr=args.learning_rate)
    q_value_optim = optim.Adam(q_value.parameters(), lr=args.learning_rate)

    
    
    # load demonstrations
    expert_demo= pickle.load(open(f'../expert_demo/expert_demo_{args.env_name}.p', "rb"))
    demonstrations = np.array(expert_demo)
    print("demonstrations.shape", demonstrations.shape)
    
    writer = SummaryWriter(args.logdir)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model)) + "_antagonist.pth.tar"
        ckpt = torch.load(saved_ckpt_path)
        antagonist_actor.load_state_dict(ckpt['antagonist_actor'])
        antagonist_critic.load_state_dict(ckpt['antagonist_critic'])
        antagonist_actor_optim.load_state_dict(ckpt['antagonist_actor_optim'])
        antagonist_critic_optim.load_state_dict(ckpt['antagonist_critic_optim'])

        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model)) + "_protagonist.pth.tar"
        ckpt = torch.load(saved_ckpt_path)
        protagonist_actor.load_state_dict(ckpt['protagonist_actor'])
        protagonist_critic.load_state_dict(ckpt['protagonist_critic'])
        protagonist_actor_optim.load_state_dict(ckpt['protagonist_actor_optim'])
        protagonist_critic_optim.load_state_dict(ckpt['protagonist_critic_optim'])

        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model)) + "_reward_function.pth.tar"
        ckpt = torch.load(saved_ckpt_path)
        reward_function.load_state_dict(ckpt['reward_function'])
        reward_function_optim.load_state_dict(ckpt['reward_function_optim'])


        saved_info_path = os.path.join(os.path.dirname(os.path.join(os.getcwd(), 'save_model', str(args.load_model))), "info.pickle")
        info = pickle.load(open(saved_info_path, 'rb'))
        running_state.rs.n = info['z_filter_n']
        running_state.rs.mean = info['z_filter_m']
        running_state.rs.sum_square = info['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    saved_log_path = os.path.join(args.logdir, timestamp)
    writer = SummaryWriter(saved_log_path)

    
    protagonist_episodes = 0
    antagonist_episodes = 0

    train_reward_function_flag = True

    pair_coef = args.pair_coef
    irl_coef = args.irl_coef
    antagonist_score_avgs = []
    protagonist_score_avgs = []
    for iter in range(args.max_iter_num):
        protagonist_actor.eval(), protagonist_critic.eval()
        protagonist_memory = deque()

        protagonist_steps = 0
        protagonist_scores = []

        while protagonist_steps < args.total_sample_size: 
            protagonist_state = env.reset()
            protagonist_score = 0

            norm_protagonist_state = running_state(protagonist_state)
            
            for _ in range(10000): 
                if args.render:
                    env.render()

                protagonist_steps += 1

                mu, std = protagonist_actor(torch.Tensor(norm_protagonist_state).unsqueeze(0))
                protagonist_action = get_action(mu, std)[0]
                protagonist_next_state, reward, done, _ = env.step(protagonist_action)
                #irl_reward = get_q_value(q_value, protagonist_state, protagonist_action)# #protagonist_next_state, protagonist_action)
                protagonist_reward = get_reward(reward_function, norm_protagonist_state, protagonist_action)
                antagonist_prob = log_prob_density(torch.Tensor(np.asarray([protagonist_action])), *antagonist_actor(torch.Tensor(np.asarray(norm_protagonist_state)).unsqueeze(0))).exp().item()
                protagonist_prob = log_prob_density(torch.Tensor(np.asarray([protagonist_action])), mu.detach(), std.detach()).exp().item()
                protagonist_reward = math.log(max(antagonist_prob * max(1./ protagonist_reward - 1., 1.e-6) + protagonist_prob, 1.e-6)) - math.log(protagonist_prob) 
                if done:
                    protagonist_mask = 0
                else:
                    protagonist_mask = 1

                protagonist_memory.append([protagonist_state, protagonist_action, protagonist_reward, protagonist_mask])

                norm_protagonist_next_state = running_state(protagonist_next_state)
                norm_protagonist_state = norm_protagonist_next_state

                protagonist_state = protagonist_next_state

                protagonist_score += reward

                if done:
                    break
            
            protagonist_episodes += 1
            protagonist_scores.append(protagonist_score)
        protagonist_score_avg = np.mean(protagonist_scores)
        protagonist_score_avgs.append(protagonist_score_avg)
        print('{}::{}:: {} protagonist_episode score is {:.2f}'.format(args.env_name, iter, protagonist_episodes, protagonist_score_avg))
        writer.add_scalar(f'log/{args.env_name}_pgail_protagonist_score', float(protagonist_score_avg), iter)

        
        antagonist_actor.eval(), antagonist_critic.eval()
        antagonist_memory = deque()

        antagonist_steps = 0
        antagonist_scores = []

        while antagonist_steps < args.total_sample_size: 
            antagonist_state = env.reset()
            antagonist_score = 0

            norm_antagonist_state = running_state(antagonist_state)
            
            for _ in range(10000): 
                if args.render:
                    env.render()

                antagonist_steps += 1

                mu, std = antagonist_actor(torch.Tensor(np.asarray(norm_antagonist_state)).unsqueeze(0))
                antagonist_action = get_action(mu, std)[0]
                antagonist_next_state, reward, done, _ = env.step(antagonist_action)
                #irl_reward = get_q_value(q_value, antagonist_state, antagonist_action) #antagonist_next_state, antagonist_action)
                antagonist_reward = get_reward(reward_function, norm_antagonist_state, antagonist_action)
                if done:
                    antagonist_mask = 0
                else:
                    antagonist_mask = 1

                antagonist_memory.append([antagonist_state, antagonist_action, antagonist_reward, antagonist_mask])

                norm_antagonist_next_state = running_state(antagonist_next_state)
                norm_antagonist_state = norm_antagonist_next_state
                antagonist_state = antagonist_next_state

                antagonist_score += reward

                if done:
                    break
            
            antagonist_episodes += 1
            antagonist_scores.append(antagonist_score)
        
        antagonist_score_avg = np.mean(antagonist_scores)
        antagonist_score_avgs.append(antagonist_score_avg)
        print('{}::{}:: {} antagonist_episode score is {:.2f}'.format(args.env_name, iter, antagonist_episodes, antagonist_score_avg))
        writer.add_scalar(f'log/{args.env_name}_pgail_antagonist_score', float(antagonist_score_avg), iter)

        
        
        if train_reward_function_flag:
            reward_function.train()

            """
            expert_acc, antagonist_learner_acc = train_vdb(reward_function, antagonist_memory, reward_function_optim, demonstrations, 0, args)
            protagonist_learner_acc = 0.0
            print("Expert: %.2f%% | Protagonist_Learner: %.2f%% | Antagonist_Learner: %.2f%%" % (expert_acc * 100, protagonist_learner_acc * 100, antagonist_learner_acc * 100))
            if expert_acc > args.suspend_accu_exp and antagonist_learner_acc > args.suspend_accu_gen:  
                train_reward_function_flag = False
            """
            #if expert_acc > antagonist_learner_acc:
            #    args.irl_coef += 100
            expert_acc, protagonist_learner_acc, antagonist_learner_acc = train_reward_function(running_state, protagonist_actor, antagonist_actor, reward_function, None, protagonist_memory, antagonist_memory, reward_function_optim, None, demonstrations, args)
            print("Expert: %.2f%% | Protagonist_Learner: %.2f%% | Antagonist_Learner: %.2f%%" % (expert_acc * 100, protagonist_learner_acc * 100, antagonist_learner_acc * 100))
            #if expert_acc > 0.5 or antagonist_learner_acc < 0.5:
            #    args.pair_coef = 0.
            #else:
            #    args.pair_coef = pair_coef
            """
            if expert_acc > 0.5 or antagonist_learner_acc < 0.5:
                args.pair_coef = 0
                #args.irl_coef += 100
                #reward_function_optim = optim.Adam(reward_function.parameters(), lr=args.learning_rate)
                print("Reset Adam")
            else:
                args.pair_coef = pair_coef
            """

        protagonist_actor.train(), protagonist_critic.train(), antagonist_actor.train(), antagonist_critic.train()
        train_protagonist_actor_critic(running_state, protagonist_actor, protagonist_critic, antagonist_actor, antagonist_critic, protagonist_memory, antagonist_memory, protagonist_actor_optim, protagonist_critic_optim, args)
        #train_actor_critic(running_state, protagonist_actor, protagonist_critic, protagonist_memory, protagonist_actor_optim, protagonist_critic_optim, args)

        #train_antagonist_actor_critic(antagonist_actor, antagonist_critic, protagonist_actor, protagonist_critic, antagonist_memory, protagonist_memory, antagonist_actor_optim, antagonist_critic_optim, args)
        train_actor_critic(running_state, antagonist_actor, antagonist_critic, antagonist_memory, antagonist_actor_optim, antagonist_critic_optim, args)

        if iter % 100 == 0:
            antagonist_score_avg = int(antagonist_score_avg)
            protagonist_score_avg = int(protagonist_score_avg)
            model_path = os.path.join(os.getcwd(),'save_model', timestamp)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            protagonist_ckpt_path = os.path.join(model_path, f'ckpt_{args.env_name}_{protagonist_score_avg}_protagonist.pth.tar')
            save_checkpoint({
                'protagonist_actor': protagonist_actor.state_dict(),
                'protagonist_critic': protagonist_critic.state_dict(),
                'protagonist_actor_optim': protagonist_actor_optim.state_dict(),
                'protagonist_critic_optim': protagonist_critic_optim.state_dict()
            }, filename=protagonist_ckpt_path)
            
            antagonist_ckpt_path = os.path.join(model_path, f'ckpt_{args.env_name}_{protagonist_score_avg}_antagonist.pth.tar')
            save_checkpoint({
                'antagonist_actor': antagonist_actor.state_dict(),
                'antagonist_critic': antagonist_critic.state_dict(),
                'antagonist_actor_optim': antagonist_actor_optim.state_dict(),
                'antagonist_critic_optim': antagonist_critic_optim.state_dict()
            }, filename=antagonist_ckpt_path)

            reward_ckpt_path = os.path.join(model_path, f'ckpt_{args.env_name}_{protagonist_score_avg}_reward_function.pth.tar')
            save_checkpoint({
                'reward_function': reward_function.state_dict(),
                'reward_function_optim': reward_function_optim.state_dict()
            }, filename=reward_ckpt_path)
             
             
            
            with open(os.path.join(model_path, 'info.pickle'), 'wb') as f:
                pickle.dump({'z_filter_n':running_state.rs.n,
                    'z_filter_m': running_state.rs.mean,
                    'z_filter_s': running_state.rs.sum_square,
                    'args': args,
                    'protagonist_scores': protagonist_score_avgs,
                    'antagonist_scores': antagonist_score_avgs
                }, f)
        

if __name__=="__main__":
    main()