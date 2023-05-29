"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
"""

import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types

import hydra
import numpy as np
import torch
import torch.nn.functional as F
#import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from wrappers.atari_wrapper import LazyFrames
from make_envs import make_env
from dataset.memory import Memory, Example
from agent import make_agent
from utils.utils import eval_mode, average_dicts, get_concat_samples, evaluate, soft_update, hard_update, clip_grad_value
from utils.logger import Logger
from iq import iq_loss

torch.set_num_threads(2)



import git 


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)

 
 

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    env_args = args.env
    
    antagonist_env = make_env(args)
    antagonist_eval_env = make_env(args)
    # Seed envs
    antagonist_env.seed(args.seed)
    antagonist_eval_env.seed(args.seed + 10)

    protagonist_env = make_env(args)
    protagonist_eval_env = make_env(args)
    # Seed envs
    protagonist_env.seed(args.seed)
    protagonist_eval_env.seed(args.seed + 10)

    

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)
    INITIAL_STATES = 128  # Num initial states to use to calculate value of initial state distribution s_0

    antagonist = make_agent(antagonist_env, 'antagonist', args)
    protagonist = make_agent(protagonist_env, 'protagonist', args)
    protagonist.critic = antagonist.critic
    protagonist.critic_target = antagonist.critic_target
    
    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.isfile(pretrain_path):
            print("=> loading pretrain '{}'".format(args.pretrain))
            antagonist.load(pretrain_path)
        else:
            print("[Attention]: Did not find checkpoint {}".format(args.pretrain))

    # Load expert data
    expert_memory_replay = Example(REPLAY_MEMORY//2, args.seed)

    expert_demonstrations_path = os.path.join(os.path.abspath(__file__).split("iq_learn")[0], f'expert_demo/{args.env.demo}')
    expert_memory_replay.load(hydra.utils.to_absolute_path(expert_demonstrations_path),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    
    antagonist_online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)
    protagonist_online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)
 
    # Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S") + '_' + git.Repo(search_parent_directories=True).head.object.hexsha
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    
    if not os.path.exists(os.path.join(args.log_dir, 'antagonist/')):
        os.mkdir(os.path.join(args.log_dir, 'antagonist/'))
    antagonist_logger = Logger(os.path.join(args.log_dir, 'antagonist/'),
                    log_frequency=args.log_interval,
                    writer=writer,
                    save_tb=True,
                    agent=getattr(args, antagonist.name).name)

    if not os.path.exists(os.path.join(args.log_dir, 'protagonist/')):
        os.mkdir(os.path.join(args.log_dir, 'protagonist/'))
    protagonist_logger = Logger(os.path.join(args.log_dir, 'protagonist/'),
                    log_frequency=args.log_interval,
                    writer=writer,
                    save_tb=True,
                    agent=getattr(args, protagonist.name).name)
    antagonist_steps = 0
    protagonist_steps = 0
    
     
    # track mean reward and scores

    antagonist_scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
    antagonist_rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    antagonist_best_eval_returns = -np.inf
    antagonist_learn_steps = 0
    antagonist_begin_learn = False
    antagonist_episode_reward = 0


    protagonist_scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
    protagonist_rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    protagonist_best_eval_returns = -np.inf
    protagonist_learn_steps = 0
    protagonist_begin_learn = False
    protagonist_episode_reward = 0
    


    # Sample initial states from env
    antagonist_state_0 = [antagonist_env.reset()] * INITIAL_STATES
    if isinstance(antagonist_state_0[0], LazyFrames):
        antagonist_state_0 = np.array(antagonist_state_0) / 255.0
    antagonist_state_0 = torch.FloatTensor(np.array(antagonist_state_0)).to(args.device)
    
    # Sample initial states from env
    protagonist_state_0 = [protagonist_env.reset()] * INITIAL_STATES
    if isinstance(protagonist_state_0[0], LazyFrames):
        protagonist_state_0 = np.array(protagonist_state_0) / 255.0
    protagonist_state_0 = torch.FloatTensor(np.array(protagonist_state_0)).to(args.device)


    for epoch in count():
        ########## antagonist roll-out ##########
        antagonist_state = antagonist_env.reset()
        antagonist_episode_reward = 0
        antagonist_done = False

        antagonist_start_time = time.time()
        for antagonist_episode_step in range(EPISODE_STEPS):

            if antagonist_steps < args.num_seed_steps:
                # Seed replay buffer with random actions
                antagonist_action = antagonist_env.action_space.sample()
            else:
                with torch.no_grad():
                    with eval_mode(antagonist):
                        antagonist_action, antagonist_log_prob = antagonist.choose_stochastic_action(antagonist_state, sample=True)
                    with eval_mode(protagonist):
                        antagonist_protagonist_log_prob = protagonist.log_prob_density(antagonist_state, antagonist_action)
            antagonist_next_state, antagonist_reward, antagonist_done, _ = antagonist_env.step(antagonist_action)
            antagonist_episode_reward += antagonist_reward

            if True:
                with torch.no_grad():
                    with eval_mode(antagonist):
                        _, antagonist_reward = antagonist.get_rewards(antagonist_state, antagonist_action, antagonist_next_state) 
                        antagonist_reward = antagonist_reward.item()
            antagonist_steps += 1

            if antagonist_learn_steps % args.env.eval_interval == 0:
                antagonist_eval_returns, antagonist_eval_timesteps = evaluate(antagonist, antagonist_eval_env, num_episodes=args.eval.eps)
                antagonist_returns = np.mean(antagonist_eval_returns)
                antagonist_learn_steps += 1  # To prevent repeated eval at timestep 0
                antagonist_logger.log('eval/episode_reward', antagonist_returns,antagonist_learn_steps)
                antagonist_logger.log('eval/episode', epoch, antagonist_learn_steps)
                antagonist_logger.dump(antagonist_learn_steps, ty='eval')
                # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

                if antagonist_returns > antagonist_best_eval_returns:
                    # Store best eval returns
                    antagonist_best_eval_returns = antagonist_returns
                    #wandb.run.summary["best_returns"] = best_eval_returns
                    save(antagonist, epoch, args, output_dir='results_best')

            # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
            antagonist_done_no_lim = antagonist_done
            if str(antagonist_env.__class__.__name__).find('TimeLimit') >= 0 and antagonist_episode_step + 1 == antagonist_env._max_episode_steps:
                antagonist_done_no_lim = 0
            antagonist_online_memory_replay.add((
                antagonist_state, 
                antagonist_next_state, 
                antagonist_action, 
                antagonist_reward, 
                antagonist_done_no_lim, 
                antagonist_log_prob,
                antagonist_protagonist_log_prob,
            ))

            if antagonist_online_memory_replay.size() > INITIAL_MEMORY and protagonist_online_memory_replay.size() > INITIAL_MEMORY:
                # Start learning
                if antagonist_begin_learn is False:
                    print('Learn begins!')
                    antagonist_begin_learn = True

                antagonist_learn_steps += 1
                if antagonist_learn_steps == LEARN_STEPS:
                    print('Finished!')
                    #wandb.finish()
                    return

                
                ######
                # IQ-Learn Modification
                antagonist.iq_update = types.MethodType(iq_pagar_update, antagonist)
                antagonist.pagar_update_critic = types.MethodType(pagar_update_critic, antagonist)
                antagonist.iq_update_critic = types.MethodType(iq_update_critic, antagonist)
                antagonist_losses = antagonist.iq_update(protagonist, protagonist_online_memory_replay, 
                                         antagonist_online_memory_replay,
                                         expert_memory_replay, antagonist_logger, antagonist_learn_steps)
                if antagonist_learn_steps % args.log_interval == 0:
                    for key, loss in antagonist_losses.items():
                        writer.add_scalar(key, loss, global_step=antagonist_learn_steps)
                ######
                  
            if antagonist_done:
                break
            antagonist_state = antagonist_next_state

        antagonist_rewards_window.append(antagonist_episode_reward)
        antagonist_logger.log('train/episode', epoch, antagonist_learn_steps)
        antagonist_logger.log('train/episode_reward', antagonist_episode_reward, antagonist_learn_steps)
        antagonist_logger.log('train/duration', time.time() - antagonist_start_time, antagonist_learn_steps)
        antagonist_logger.dump(antagonist_learn_steps, save=antagonist_begin_learn)
        writer.add_scalar("antagonist_episode_reward", antagonist_episode_reward, global_step=antagonist_learn_steps)
        # print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
        save(antagonist, epoch, args, output_dir='results')

        
        ########## Protagonist roll-out ##########
        protagonist_state = protagonist_env.reset()
        protagonist_episode_reward = 0
        protagonist_done = False

        protagonist_start_time = time.time()
        for protagonist_episode_step in range(EPISODE_STEPS):

            if protagonist_steps < args.num_seed_steps:
                # Seed replay buffer with random actions
                protagonist_action = protagonist_env.action_space.sample()
            else:
                with torch.no_grad():
                    with eval_mode(protagonist):
                        protagonist_action, protagonist_log_prob = protagonist.choose_stochastic_action(protagonist_state, sample=True)
                    with eval_mode(antagonist):
                        protagonist_antagonist_log_prob = antagonist.log_prob_density(protagonist_state, protagonist_action)
                        
            protagonist_next_state, protagonist_reward, protagonist_done, _ = protagonist_env.step(protagonist_action)
            protagonist_episode_reward += protagonist_reward
            
            if True:
                with torch.no_grad():
                    with eval_mode(antagonist):
                        _, protagonist_reward = antagonist.get_rewards(protagonist_state, protagonist_action, protagonist_next_state)
                        protagonist_reward = protagonist_reward.item()
            protagonist_steps += 1

            if protagonist_learn_steps % args.env.eval_interval == 0:
                protagonist_eval_returns, protagonist_eval_timesteps = evaluate(protagonist, protagonist_eval_env, num_episodes=args.eval.eps)
                protagonist_returns = np.mean(protagonist_eval_returns)
                protagonist_learn_steps += 1  # To prevent repeated eval at timestep 0
                protagonist_logger.log('eval/episode_reward', protagonist_returns,protagonist_learn_steps)
                protagonist_logger.log('eval/episode', epoch, protagonist_learn_steps)
                protagonist_logger.dump(protagonist_learn_steps, ty='eval')
                # print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))

                if protagonist_returns > protagonist_best_eval_returns:
                    # Store best eval returns
                    protagonist_best_eval_returns = protagonist_returns
                    #wandb.run.summary["best_returns"] = best_eval_returns
                    save(protagonist, epoch, args, output_dir='results_best')

            # only store done true when episode finishes without hitting timelimit (allow infinite bootstrap)
            protagonist_done_no_lim = protagonist_done
            if str(protagonist_env.__class__.__name__).find('TimeLimit') >= 0 and protagonist_episode_step + 1 == protagonist_env._max_episode_steps:
                protagonist_done_no_lim = 0
            protagonist_online_memory_replay.add((
                protagonist_state, 
                protagonist_next_state, 
                protagonist_action, 
                protagonist_reward, 
                protagonist_done_no_lim, 
                protagonist_antagonist_log_prob,
                protagonist_log_prob))

            if antagonist_online_memory_replay.size() > INITIAL_MEMORY and protagonist_online_memory_replay.size() > INITIAL_MEMORY:
                # Start learning
                if protagonist_begin_learn is False:
                    print('Learn begins!')
                    protagonist_begin_learn = True

                protagonist_learn_steps += 1
                if protagonist_learn_steps == LEARN_STEPS:
                    print('Finished!')
                    #wandb.finish()
                    return
                
                ######
                # IQ-Learn Modification
                protagonist.iq_update = types.MethodType(sac_ppo_update, protagonist)
                protagonist.iq_update_critic = types.MethodType(iq_update_critic, protagonist)
                protagonist.pagar_update_critic = types.MethodType(pagar_update_critic, protagonist)
                protagonist_losses = protagonist.iq_update(protagonist_online_memory_replay, 
                                         #antagonist, 
                                         antagonist_online_memory_replay, 
                                         expert_memory_replay, protagonist_logger, protagonist_learn_steps)
                if antagonist_learn_steps % args.log_interval == 0:
                    for key, loss in protagonist_losses.items():
                        writer.add_scalar(key, loss, global_step=protagonist_learn_steps)
                ######
                
                
            if protagonist_done:
                break
            protagonist_state = protagonist_next_state

        protagonist_rewards_window.append(protagonist_episode_reward)
        protagonist_logger.log('train/episode', epoch, protagonist_learn_steps)
        protagonist_logger.log('train/episode_reward', protagonist_episode_reward, protagonist_learn_steps)
        protagonist_logger.log('train/duration', time.time() - protagonist_start_time, protagonist_learn_steps)
        protagonist_logger.dump(protagonist_learn_steps, save=protagonist_begin_learn)
        writer.add_scalar("protagonist_episode_reward", protagonist_episode_reward, global_step=protagonist_learn_steps)
        # print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
        save(protagonist, epoch, args, output_dir='results')
         


def save(agent, epoch, args, output_dir='results'):
    if epoch % args.save_interval == 0:
        if args.method.type == "sqil":
            name = f'sqil_{args.env.name}'
        else:
            name = f'iq_{args.env.name}'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        agent.save(f'{output_dir}/{getattr(args, agent.name).name}_{name}')


# Minimal IQ-Learn objective
def iq_learn_update(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    obs, next_obs, action, reward, done, is_expert = get_concat_samples(
        policy_batch, expert_batch, args)

    loss_dict = {}

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    current_Q = self.critic(obs, action)
    y = (1 - done) * self.gamma * self.getV(next_obs)
    if args.train.use_target:
        with torch.no_grad():
            y = (1 - done) * self.gamma * self.get_targetV(next_obs)

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()

    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    value_loss = (self.getV(obs) - y).mean()
    loss += value_loss

    # Use χ2 divergence (adds a extra term to the loss)
    chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
    loss += chi2_loss
    ######

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return loss


def iq_update_critic(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch[:5]
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch[:5]
    #print(policy_action)
    if args.only_expert_states:
        # Use policy actions instead of experts actions for IL with only observations
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    batch = get_concat_samples(policy_batch, expert_batch, args)
    obs, next_obs, action = batch[0:3]

    agent = self
    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)
    else:
        next_V = self.getV(next_obs)

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch)
        q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch)
        critic_loss = 1/2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    logger.log('train/critic_loss', critic_loss, step)

   
    return critic_loss, loss_dict


def iq_update(self, policy_buffer, expert_buffer, logger, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    critic_loss, losses = self.iq_update_critic(policy_batch, expert_batch, logger, step)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    self.critic_optimizer.step()

    if self.actor and step % self.actor_update_frequency == 0:
        if not getattr(self.args, self.name).vdice_actor:

            if self.args.offline:
                obs = expert_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
                  
            losses.update(actor_alpha_losses)
  
 
    
    if step % self.critic_target_update_frequency == 0:
        if self.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


def pagar_update_critic(self, protagonist, protagonist_batch, antagonist, antagonist_batch, antagonist_logger, step):
    args = self.args
    protagonist_states, protagonist_next_states, protagonist_action, protagonist_reward, protagonist_done, protagonist_antagonist_log_probs, protagonist_log_probs = protagonist_batch[:7]
    
    protagonist_antagonist_current_Q = self.critic(protagonist_states, protagonist_action)
    protagonist_antagonist_current_y = (1 - protagonist_done) * self.gamma * self.getV(protagonist_next_states)
    pair_r1 = - (protagonist_antagonist_current_Q - protagonist_antagonist_current_y) 
    
    antagonist_states, antagonist_next_states, antagonist_action, antagonist_reward, antagonist_done, antagonist_log_probs, antagonist_protagonist_log_probs = antagonist_batch[:7]
    antagonist_current_Q = self.critic(antagonist_states, antagonist_action)
    antagonist_current_y = (1 - antagonist_done) * self.gamma * self.getV(antagonist_next_states)
    pair_r2 = (antagonist_current_Q - antagonist_current_y) 
    

    #pair_r1 = - protagonist_antagonist_rewards.flatten()
    pair_ratio1 = torch.exp(protagonist_antagonist_log_probs - protagonist_log_probs).detach().flatten()
    pair_loss1 = (pair_r1 * pair_ratio1)[torch.isfinite(pair_ratio1)]
    pair_loss1 = pair_loss1[torch.isfinite(pair_loss1)].mean() 
    #pair_ids1 = (pair_ratio1 <=  1. + args.clip_param).float() * (pair_ratio1 >=  1. - args.clip_param).float()
    #pair_clipped_ratio1 = pair_ratio1 * pair_ids1
    #pair_loss1 = (pair_r1 * pair_clipped_ratio1)
    #pair_loss1 = pair_loss1[torch.isfinite(pair_loss1)].sum() / pair_ids1[torch.isfinite(pair_loss1)].sum()
     
        
    pair_kl1 = torch.nn.functional.mse_loss(protagonist.actor(protagonist_states).loc.flatten(), antagonist.actor(protagonist_states).loc.flatten()).detach().item()
    #pair_kl1 = torch.sqrt(protagonist_actor(protagonist_states)[0] - antagonist_actor(protagonist_states)[0])
    #pair_kl1 = pair_kl1[torch.isfinite(pair_kl1)].max().detach().item()
    #print("pair_loss1", pair_kl1, pair_loss1)
    pair_loss1 =  pair_loss1 + pair_kl1 * 4 * args.gamma / (1 - args.gamma) * torch.abs(pair_r1[torch.isfinite(pair_r1)].flatten()).max() 
    pair_loss1 = pair_loss1 - pair_r1[torch.isfinite(pair_r1)].mean()
    
    
    #pair_r2 = antagonist_rewards.flatten()
    pair_ratio2 = (torch.exp(antagonist_protagonist_log_probs - antagonist_log_probs)).detach().flatten()
    pair_loss2 = (pair_r2 * pair_ratio2)[torch.isfinite(pair_ratio2)]
    pair_loss2 = pair_loss2[torch.isfinite(pair_loss2)].mean() 
    #pair_ids2 = (pair_ratio2 <=  1. + args.clip_param).float() * (pair_ratio2 >=  1. - args.clip_param).float()
    #pair_clipped_ratio2 = pair_ratio2 * pair_ids2
    #pair_loss2 = (pair_r2 * pair_clipped_ratio2)
    #pair_loss2 = pair_loss2[torch.isfinite(pair_loss2)].sum() / pair_ids2[torch.isfinite(pair_loss2)].sum()
    
    pair_kl2 = torch.nn.functional.mse_loss(antagonist.actor(antagonist_states).loc.flatten(), protagonist.actor(antagonist_states).loc.flatten()).detach().item()
    #pair_kl2 = torch.sqrt(antagonist_actor(antagonist_states)[0] - protagonist_actor(antagonist_states)[0])
    #pair_kl2 = pair_kl2[torch.isfinite(pair_kl2)].max().detach().item()
    #print("pair_loss2", pair_kl2, pair_loss2)
    pair_loss2 = pair_loss2 - pair_kl2 * 4 * args.gamma / (1 - args.gamma) * torch.abs(pair_r2[torch.isfinite(pair_r2)].flatten()).max() 
    pair_loss2 = pair_loss2 - pair_r2[torch.isfinite(pair_r2)].mean() 
    
    """
    pair_ratio3 = (torch.exp(pair_r2.flatten() - antagonist_log_probs.flatten().detach())) 
    pair_ids3 = (pair_ratio3 <=  1. + args.clip_param).float() * (pair_ratio3 >=  1. - args.clip_param).float()
    pair_clipped_ratio3 = torch.clamp(pair_ratio3, 1 - args.clip_param, 1 + args.clip_param)# * pair_ids3.detach()
    pair_loss3 = - torch.min(pair_r2 * pair_ratio3, pair_r2 * pair_clipped_ratio3).mean()
    #pair_loss3 = pair_loss3[torch.isfinite(pair_loss3)].sum() / pair_ids3[torch.isfinite(pair_loss3)].sum()
    pair_loss3 = pair_loss3 - pair_r1[torch.isfinite(pair_r1)].mean()
    #print("pair_loss3", pair_loss3)

    pair_ratio4 = (torch.exp(-pair_r1.flatten() - protagonist_log_probs.flatten().detach())).flatten()
    pair_ids3 = (pair_ratio3 <=  1. + args.clip_param).float() * (pair_ratio3 >=  1. - args.clip_param).float()
    pair_clipped_ratio4 = torch.clamp(pair_ratio4, 1 - args.clip_param, 1 + args.clip_param)# * pair_ids3.detach()
    pair_loss4 = -torch.min(-pair_r1 * pair_ratio4, -pair_r1 * pair_clipped_ratio4).mean()
    #pair_loss3 = pair_loss3[torch.isfinite(pair_loss3)].sum() / pair_ids3[torch.isfinite(pair_loss3)].sum()
    pair_loss4 = pair_loss4 - pair_r1[torch.isfinite(pair_r1)].mean()
    #print("pair_loss4", pair_loss4)
    """
    pagar_loss = pair_loss1 + pair_loss2 #+ (pair_loss3 if torch.isfinite(pair_loss3).all() else 0.)  + (pair_loss4 if torch.isfinite(pair_loss4).all() else 0.) ##(pair_loss0 if torch.isfinite(pair_loss0).all() else 0.) + #
        
    antagonist_logger.log('train/pagar_loss', pagar_loss, step)
    return pagar_loss, {'pagar_loss': pagar_loss.item()}


def iq_pagar_update(self, protagonist, protagonist_buffer, antagonist_buffer, expert_buffer, logger, step):
    protagonist_batch = protagonist_buffer.get_samples(self.batch_size, self.device)
    antagonist_batch = antagonist_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    critic_loss, losses = self.iq_update_critic(antagonist_batch, expert_batch, logger, step)
    pagar_loss, pagar_losses = self.pagar_update_critic(protagonist, protagonist_batch, self, antagonist_batch, logger, step)
    losses.update(pagar_losses)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    (critic_loss + 1.e-3 * (critic_loss < -0.3) * pagar_loss).backward()
    #critic_loss.backward()
    # step critic
    self.critic_optimizer.step()
 

    if self.actor and step % self.actor_update_frequency == 0:
        if not getattr(self.args, self.name).vdice_actor:

            if self.args.offline:
                obs = expert_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([antagonist_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
                  
            losses.update(actor_alpha_losses)
  
 
    
    if step % self.critic_target_update_frequency == 0:
        if self.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses
 


def iq_ppo_update(self, protagonist_buffer, antagonist_buffer, expert_buffer, logger, step):
    protagonist_batch = protagonist_buffer.get_samples(self.batch_size, self.device)
    antagonist_batch = antagonist_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)
    
    critic_loss, losses = self.iq_update_critic(protagonist_batch, antagonist_batch, logger, step)
    
    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    self.critic_optimizer.step()

    if self.actor and step % self.actor_update_frequency == 0:
        if not getattr(self.args, self.name).vdice_actor:

            if self.args.offline:
                obs = protagonist_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([protagonist_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step, *antagonist_batch)
                  
            losses.update(actor_alpha_losses)
  
 
    
    if step % self.critic_target_update_frequency == 0:
        if self.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses

def sac_ppo_update(self, protagonist_buffer, antagonist_buffer, expert_buffer, logger, step):
    ###################
    #### sac_ppo learn ######
    ###################
    protagonist_batch = protagonist_buffer.get_samples(self.batch_size, self.device)
    antagonist_batch = antagonist_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)
 
    batch = get_concat_samples(protagonist_batch, antagonist_batch, self.args)
    #print(len(batch))
    #assert len(batch) > 7
    #batch[3] = batch[6]
    losses = self.update_critic(batch, logger, step)
     
    if self.actor and step % self.actor_update_frequency == 0:
        if not getattr(self.args, self.name).vdice_actor:

            if self.args.offline:
                obs = protagonist_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([protagonist_batch[0], expert_batch[0]], dim=0)

                actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step, *antagonist_batch)
                losses.update(actor_alpha_losses)
    
    if step % self.critic_target_update_frequency == 0:
        if self.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


def update_critic(self, policy_batch, logger, step):
    obs, next_obs, action, reward, done = \
        policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3], policy_batch[4] 
    #print(obs.shape, next_obs.shape, action.shape, done.shape, reward.shape)
    with torch.no_grad():
        next_action, next_log_prob, _ = self.actor.sample(next_obs)
        #print(next_log_prob.shape)
        qf1_next_target, qf2_next_target = self.critic_target(next_obs, next_action, both = True)
        #print(qf1_next_target.shape)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_log_prob
        #print(min_qf_next_target.shape)
        #print(reward)
        next_q_value = reward + (1 - done) * self.gamma * (min_qf_next_target)
        #print(next_q_value.shape)

    # get current Q estimates
    qf1, qf2 = self.critic(obs, action, both=True)

    q1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    q2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    critic_loss = q1_loss + q2_loss

    
    logger.log('train/critic_loss', critic_loss, step)
    return critic_loss, {
        'critic_loss/critic_1': q1_loss.item(),
        'critic_loss/critic_2': q2_loss.item(),
        'loss/critic': critic_loss.item()}
    
def sac_ppo_update(self, protagonist_buffer, antagonist, antagonist_buffer, expert_buffer, logger, step):
    ###################
    #### sac_ppo learn ######
    ###################
    protagonist_batch = protagonist_buffer.get_samples(self.batch_size, self.device)
    antagonist_batch = antagonist_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)
    losses = {}
    #critic_loss, losses = self.critic_loss(get_concat_samples(protagonist_batch, antagonist_batch, self.args), logger, step)
    
    # Optimize the critic
    #self.critic_optimizer.zero_grad()
    #critic_loss.backward()
    #self.critic_optimizer.step()

    if self.actor and step % self.actor_update_frequency == 0:
        if not getattr(self.args, self.name).vdice_actor:

            if self.args.offline:
                obs = protagonist_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([protagonist_batch[0], expert_batch[0]], dim=0)

                actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step, *antagonist_batch)
                losses.update(actor_alpha_losses)
    
    if step % self.critic_target_update_frequency == 0:
        if self.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses

def iq_sac_ppo_update(self, protagonist_buffer, antagonist_buffer, expert_buffer, logger, step):
    ###################
    #### iq_sac_ppo learn ######
    ###################
    protagonist_batch = protagonist_buffer.get_samples(self.batch_size, self.device)
    antagonist_batch = antagonist_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)
    
    critic_loss, losses = self.critic_loss(get_concat_samples(protagonist_batch, antagonist_batch, self.args), logger, step)
   
    iq_critic_loss1, iq_losses1 = self.iq_update_critic(antagonist_batch, expert_batch, logger, step)
    #iq_critic_loss2, iq_losses2 = self.iq_update_critic(antagonist_batch, expert_batch, logger, step)
    #iq_critic_loss3, iq_losses3 = self.iq_update_critic(protagonist_batch, expert_batch, logger, step)
    
    iq_critic_loss = iq_critic_loss1 #+ iq_critic_loss2 + iq_critic_loss3
    
    for iq_losses in [iq_losses1]:#, iq_losses2, iq_losses3]:  
        for k, v in iq_losses.items():
            if losses.get(k, None) is None:
                losses[k] = v
            else:
                losses[k] += v 
                
    # Optimize the critic
    self.critic_optimizer.zero_grad()
    (iq_critic_loss + critic_loss).backward()
    self.critic_optimizer.step()

    if self.actor and step % self.actor_update_frequency == 0:
        if not getattr(self.args, self.name).vdice_actor:

            if self.args.offline:
                obs = protagonist_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([protagonist_batch[0], expert_batch[0]], dim=0)

                actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step, *antagonist_batch)
                losses.update(actor_alpha_losses)
    
    if step % self.critic_target_update_frequency == 0:
        if self.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


def iq_pagar_sac_ppo_update(self, protagonist_buffer, antagonist, antagonist_buffer, expert_buffer, logger, step):
    ###################
    #### iq_sac_ppo learn ######
    ###################
    protagonist_batch = protagonist_buffer.get_samples(self.batch_size, self.device)
    antagonist_batch = antagonist_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)
    
 
    #critic_loss, losses = self.critic_loss(get_concat_samples(protagonist_batch, antagonist_batch, self.args), logger, step)
    critic_loss, losses = self.iq_update_critic(antagonist_batch, expert_batch, logger, step)
    
    pagar_loss, pagar_losses = self.pagar_update_critic(self, protagonist_batch, antagonist, antagonist_batch, logger, step)
    losses.update(pagar_losses)
    #iq_critic_loss2, iq_losses2 = self.iq_update_critic(antagonist_batch, expert_batch, logger, step)
    #iq_critic_loss3, iq_losses3 = self.iq_update_critic(protagonist_batch, expert_batch, logger, step)
    
    #iq_critic_loss = iq_critic_loss1 #+ iq_critic_loss2 + iq_critic_loss3
    
    #for iq_losses in [iq_losses1]:#, iq_losses2, iq_losses3]:  
    #    for k, v in iq_losses.items():
    #        if losses.get(k, None) is None:
    #            losses[k] = v
    #        else:
    #            losses[k] += v 
                
    # Optimize the critic
    self.critic_optimizer.zero_grad()
    (1.e-3 * (critic_loss < -0.3) * pagar_loss).backward()
    self.critic_optimizer.step()

    if self.actor and step % self.actor_update_frequency == 0:
        if not getattr(self.args, self.name).vdice_actor:

            if self.args.offline:
                obs = protagonist_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([protagonist_batch[0], expert_batch[0]], dim=0)

                actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step, *antagonist_batch)
                losses.update(actor_alpha_losses)
    
    if step % self.critic_target_update_frequency == 0:
        if self.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses



if __name__ == "__main__":
    main()