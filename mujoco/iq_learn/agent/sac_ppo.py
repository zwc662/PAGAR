import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import math
import hydra

from utils.utils import soft_update


class SAC_PPO(object):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        self.name = "protagonist"
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.batch_size = batch_size
        self.action_range = action_range
        self.device = torch.device(args.device)
        self.clip_param = args.clip_param
        self.args = args
        agent_cfg = args.protagonist
        #self.args.method.loss = "value_expert"
        self.critic_tau = agent_cfg.critic_tau
        self.learn_temp = agent_cfg.init_temp
        self.actor_update_frequency = agent_cfg.actor_update_frequency
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency
        self.soft_update = agent_cfg.soft_update

        self.critic = hydra.utils.instantiate(agent_cfg.critic_cfg, args={'method': args.method, 'gamma': args.gamma}).to(self.device)

        self.critic_target = hydra.utils.instantiate(agent_cfg.critic_cfg, args={'method': args.method, 'gamma': args.gamma}).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(agent_cfg.actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=agent_cfg.actor_lr,
                                    betas=agent_cfg.actor_betas)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=agent_cfg.critic_lr,
                                     betas=agent_cfg.critic_betas)
        self.log_alpha_optimizer = Adam([self.log_alpha],
                                        lr=agent_cfg.alpha_lr,
                                        betas=agent_cfg.alpha_betas)
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.critic

    @property
    def critic_target_net(self):
        return self.critic_target

    def log_prob_density(self, state, action):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        return dist.log_prob(action).sum(-1, keepdim=True).detach().cpu().numpy()[0].item()


    def get_rewards(self, state, action, next_state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0) 
        next_state = torch.FloatTensor(next_state).to(self.device).unsqueeze(0)
        current_Q = self.critic(state, action).detach().cpu().numpy()[0]
        current_y = self.getV(state).detach().cpu().numpy()[0]
        next_y = self.getV(next_state).detach().cpu().numpy()[0]
        return current_Q - current_y, current_Q - next_y
 
    def choose_action(self, state, sample=False):
        return self.choose_stochastic_action(state, sample)[0]
    
    def choose_stochastic_action(self, state, sample = False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, log_prob_density, _ = self.actor.sample(state)
        #dist = self.actor(state)
        #action = torch.normal(dist.loc, dist.scale)
        #action = action.data.numpy()
        #print(dist.loc, dist.scale, action)
        #log_prob_density = dist.log_prob(action).sum(-1, keepdim=True).flatten()
        #print((-(action - dist.loc).pow(2) / (2 * dist.scale.pow(2)) - 0.5 * math.log(2 * math.pi)))
        #print(log_prob_density)
        #    (action <= -1.0) * 0.5 * (1 + torch.erf((action - dist.loc) * dist.scale.reciprocal() / math.sqrt(2))) + \
        #        (action >= 1.0) * (1. - 0.5 * (1 + torch.erf((action - dist.loc) * dist.scale.reciprocal() / math.sqrt(2)))) + \
        #            (action > -1.0) * (action < 1.0) * (-(action - dist.loc).pow(2) / (2 * dist.scale.pow(2)) - 0.5 * math.log(2 * math.pi))
        #)
        
        #action = dist.sample() if sample else dist.mean
        # assert action.ndim == 2 and action.shape[0] == 1
        #print(log_prob_density, action)
        #print(log_prob_density)
        return action.detach().cpu().numpy()[0], log_prob_density.detach().cpu().numpy()[0].item()

    def getV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        current_Q = self.critic(obs, action)
        current_V = current_Q - self.alpha.detach() * log_prob
        return current_V

    def get_targetV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        target_Q = self.critic_target(obs, action)
        target_V = target_Q - self.alpha.detach() * log_prob
        return target_V

    def update(self, protagonist_replay_buffer, antagonist_replay_buffer, logger, step):
        protagonist_policy_batch = protagonist_replay_buffer.get_samples(
            self.batch_size, self.device)

        losses = self.update_critic(protagonist_policy_batch, logger, step)

        if step % self.actor_update_frequency == 0:
            antagonist_policy_batch = antagonist_replay_buffer.get_samples(self.batch_size, self.device)
                
            actor_alpha_losses = self.update_actor_and_alpha(protagonist_policy_batch[0], logger, step, antagonist_policy_batch)
            losses.update(actor_alpha_losses)

        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic, self.critic_target,
                        self.critic_tau)

        return losses
    
    def critic_loss(self, policy_batch, logger, step):
        obs, next_obs, action, reward, done, log_prob = \
            policy_batch[0], policy_batch[1], policy_batch[2], policy_batch[3], policy_batch[4], policy_batch[5] 
         
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
    
    def update_critic(self, policy_batch, logger, step):
        critic_loss, losses = self.critic_loss(policy_batch, logger, step)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        
        return losses
    
    def surrogate_loss(self, advants, states, old_policy, actions):
        
        new_policy = self.actor(states).log_prob(actions).sum(-1, keepdim=True)
        old_policy = old_policy 

        ratio = torch.exp(new_policy - old_policy)
        surrogate_loss = ratio * advants
        
        return surrogate_loss, ratio 

    def update_ppo(self, policy_batch, logger, step):
        obs, action, done, advant, old_log_prob = \
            policy_batch[0], policy_batch[2], policy_batch[4], policy_batch[3], policy_batch[5]   
        #advant = antagonist_log_prob.detach()
        advant = (advant - advant.mean()) / advant.std()
        if False and len(antagonist_policy_batch) <= 6:
            # For testing:
            ### If policy batch contains no more than 6 items, it implies that the samples are drawn from the protagonist itself instead of the antagonist
            ### Then the 6th item is not antagonist log prob, but protagonist log prob, and advant is original reward, not antagonist advantage
            ### In this case, the protagonist advantage should be computed to replace the dummy advant
            with torch.no_grad():
                advant = (self.critic_target(obs, action) - self.getV(obs)).detach()

        loss, ratio = self.surrogate_loss(((1 - done) * advant).detach(), obs,
                                    old_log_prob.detach(), action)
        clipped_ratio = torch.clamp(ratio,
                                    1.0 - self.clip_param,
                                    1.0 + self.clip_param)
        clipped_loss = clipped_ratio * advant
        ppo_loss = -torch.min(loss, clipped_loss).mean()
        logger.log('train/ppo_loss', ppo_loss, step)
        
        return ppo_loss

    def update_actor(self, obs, logger, step):
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q = self.critic(obs, action)

        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train/actor_loss', actor_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/actor_entropy', -log_prob.mean(), step)

        return actor_loss, log_prob
    
    def update_actor_and_alpha_with_critic(self, obs, critic, logger, step, *antagonist_policy_batch):
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q1, actor_Q2 = critic(obs, action, both = True)

        actor_loss = (self.alpha.detach() * log_prob - torch.min(actor_Q1, actor_Q2)).mean()
        
        ppo_loss = 0
        if antagonist_policy_batch:
            ppo_loss = self.update_ppo(antagonist_policy_batch, logger, step)

        
        logger.log('train/actor_loss', actor_loss, step)
        logger.log('train/ppo_loss', ppo_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/actor_entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        (actor_loss + ppo_loss).backward()
        self.actor_optimizer.step()

        losses = {
            'loss/actor': (ppo_loss + actor_loss).item(),
            'actor_loss/target_entropy': self.target_entropy,
            'actor_loss/entropy': -log_prob.mean().item(),
            'ppo_loss': ppo_loss}

        # self.actor.log(logger, step)
        if self.learn_temp:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train/alpha_loss', alpha_loss, step)
            logger.log('train/alpha_value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            losses.update({
                'alpha_loss/loss': alpha_loss.item(),
                'alpha_loss/value': self.alpha.item(),
            })
        return losses

    def update_actor_and_alpha(self, obs, logger, step, *antagonist_policy_batch):
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q1, actor_Q2 = self.critic(obs, action, both = True)

        actor_loss = (self.alpha.detach() * log_prob - torch.min(actor_Q1, actor_Q2)).mean()
        
        ppo_loss = 0
        if antagonist_policy_batch:
            ppo_loss = self.update_ppo(antagonist_policy_batch, logger, step)

        
        logger.log('train/actor_loss', actor_loss, step)
        logger.log('train/ppo_loss', ppo_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/actor_entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        (actor_loss + ppo_loss).backward()
        self.actor_optimizer.step()

        losses = {
            'loss/actor': (ppo_loss + actor_loss).item(),
            'actor_loss/target_entropy': self.target_entropy,
            'actor_loss/entropy': -log_prob.mean().item(),
            'ppo_loss': ppo_loss}

        # self.actor.log(logger, step)
        if self.learn_temp:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train/alpha_loss', alpha_loss, step)
            logger.log('train/alpha_value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            losses.update({
                'alpha_loss/loss': alpha_loss.item(),
                'alpha_loss/value': self.alpha.item(),
            })
        return losses

    # Save model parameters
    def save(self, path, suffix=""):
        actor_path = f"{path}{suffix}_actor"
        critic_path = f"{path}{suffix}_critic"

        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load(self, path, suffix=""):
        actor_path = f'{path}/{self.args.protagonist.name}{suffix}_actor'
        critic_path = f'{path}/{self.args.protagonist.name}{suffix}_critic'
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.FloatTensor(action).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()

    def sample_actions(self, obs, num_actions):
        """For CQL style training."""
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
            obs.shape[0] * num_actions, obs.shape[1])
        action, log_prob, _ = self.actor.sample(obs_temp)
        return action, log_prob.view(obs.shape[0], num_actions, 1)

    def _get_tensor_values(self, obs, actions, network=None):
        """For CQL style training."""
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def cqlV(self, obs, network, num_random=10):
        """For CQL style training."""
        # importance sampled version
        action, log_prob = self.sample_actions(obs, num_random)
        current_Q = self._get_tensor_values(obs, action, network)

        random_action = torch.FloatTensor(
            obs.shape[0] * num_random, action.shape[-1]).uniform_(-1, 1).to(self.device)

        random_density = np.log(0.5 ** action.shape[-1])
        rand_Q = self._get_tensor_values(obs, random_action, network)
        alpha = self.alpha.detach()

        cat_Q = torch.cat(
            [rand_Q - alpha * random_density, current_Q - alpha * log_prob.detach()], 1
        )
        cql_V = torch.logsumexp(cat_Q / alpha, dim=1).mean() * alpha
        return cql_V
