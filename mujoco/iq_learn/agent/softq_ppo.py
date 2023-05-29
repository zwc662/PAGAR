import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
import hydra

from wrappers.atari_wrapper import LazyFrames


class SoftQ_PPO(object):
    def __init__(self, num_inputs, action_dim, batch_size, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.protagonist
        self.actor = None
        self.critic_tau = agent_cfg.critic_tau

        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency
        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.q_net = hydra.utils.instantiate(
            agent_cfg.critic_cfg, args=args, device=self.device).to(self.device)
        self.target_net = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args, device=self.device).to(
            self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.critic_optimizer = Adam(self.q_net.parameters(), lr=agent_cfg.critic_lr,
                                     betas=agent_cfg.critic_betas)
        self.train()
        self.target_net.train()

    def train(self, training=True):
        self.training = training
        self.q_net.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.q_net

    @property
    def critic_target_net(self):
        return self.target_net

    def choose_action(self, state, sample=False):
        if isinstance(state, LazyFrames):
            state = np.array(state) / 255.0
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(state)
            dist = F.softmax(q/self.alpha, dim=1)
            # if sample:
            dist = Categorical(dist)
            action = dist.sample()  # if sample else dist.mean
            # else:
            #     action = torch.argmax(dist, dim=1)

        return action.detach().cpu().numpy()[0]

    def log_prob_density(self, state, action):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = torch.FloatTensor(action).to(self.device).unsqueeze(0)
        return dist.log_prob(action).sum(-1, keepdim=True).detach().cpu().numpy()[0].item()

    def getV(self, obs):
        q = self.q_net(obs)
        v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return v

    def critic(self, obs, action, both=False):
        q = self.q_net(obs, both)
        if isinstance(q, tuple) and both:
            q1, q2 = q
            critic1 = q1.gather(1, action.long())
            critic2 = q2.gather(1, action.long())
            return critic1, critic2

        return q.gather(1, action.long())

    def get_targetV(self, obs):
        q = self.target_net(obs)
        target_v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return target_v

    def update(self, protagonist_replay_buffer, antagonist_replay_buffer, logger, step):
        protagonist_obs, protagonist_next_obs, protagonist_action, protagonist_reward, protagonist_done, protagonist_antagonist_reward, protagonist_antagonist_log_prob, protagonist_log_prob = protagonist_replay_buffer.get_samples(
            self.batch_size, self.device)


        antagonist_obs, antagonist_next_obs, antagonist_action, antagonist_reward, antagonist_done, antagonist_reward, antagonist_log_prob, antagonist_protagonist_log_prob = antagonist_replay_buffer.get_samples(self.batch_size, self.device)[0]
        ppo_loss = self.update_ppo(antagonist_obs, antagonist_action, antagonist_reward, antagonist_log_prob, logger, step)
        logger.log('train_critic/ppo_loss', ppo_loss , step)

        with torch.no_grad():
            next_v = self.get_targetV(protagonist_next_obs)
            y = protagonist_antagonist_reward.detach() + (1 - protagonist_done) * self.gamma * next_v

        critic_loss = F.mse_loss(self.critic(protagonist_obs, protagonist_action), y)
        logger.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        (ppo_loss + critic_loss).backward()
        self.critic_optimizer.step()

        return {
            'loss/critic': (ppo_loss + critic_loss).item()}
 

    def update_critic(self, obs, action, reward, next_obs, done, logger,
                      step):

        with torch.no_grad():
            next_v = self.get_targetV(next_obs)
            y = reward + (1 - done) * self.gamma * next_v

        critic_loss = F.mse_loss(self.critic(obs, action), y)
        logger.log('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            'loss/critic': critic_loss.item()}

    def surrogate_loss(self, advants, states, old_policy, actions):
        new_policy = self.actor(states).log_prob(actions).sum(-1, keepdim=True)
        old_policy = old_policy 

        ratio = torch.exp(new_policy - old_policy)
        surrogate_loss = ratio * advants
        
        return surrogate_loss, ratio 

    def update_ppo(self, obs, action, advants, old_policy, logger, step):
        loss, ratio = self.surrogate_loss(advants, obs,
                                    old_policy.detach(), action)
        clipped_ratio = torch.clamp(ratio,
                                    1.0 - self.args.protagonist.clip_param,
                                    1.0 + self.args.protagonist.clip_param)
        clipped_loss = clipped_ratio * advants
        ppo_loss = -torch.min(loss, clipped_loss).mean()
        logger.log('train/protagonist_ppo_loss', ppo_loss, step)
        
        return ppo_loss

    # Save model parameters
    def save(self, path, suffix=""):
        critic_path = f"{path}{suffix}"
        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.q_net.state_dict(), critic_path)

    # Load model parameters
    def load(self, path, suffix=""):
        critic_path = f'{path}/{self.args.protagonist.name}{suffix}'
        print('Loading models from {}'.format(critic_path))
        self.q_net.load_state_dict(torch.load(critic_path, map_location=self.device))

    def infer_q(self, state, action):
        if isinstance(state, LazyFrames):
            state = np.array(state) / 255.0
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = torch.FloatTensor([action]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        if isinstance(state, LazyFrames):
            state = np.array(state) / 255.0
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()
