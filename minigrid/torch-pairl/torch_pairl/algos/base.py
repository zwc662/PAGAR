from abc import ABC, abstractmethod
import torch

import sys, os
python_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(python_path)


from torch_pairl.format import default_preprocess_obss
from torch_pairl.utils import DictList, ParallelEnv


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, protagonist_envs, antagonist_envs, protagonist_acmodel, antagonist_acmodel, discmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, ac_recurrence, disc_recurrence, preprocess_obss, reshape_reward, entropy_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        entropy_reward: bool
            whether add entropy regularizer to the reward
        """

        # Store parameters

        self.protagonist_env = ParallelEnv(protagonist_envs)
        self.antagonist_env = ParallelEnv(antagonist_envs)
        self.protagonist_acmodel = protagonist_acmodel
        self.antagonist_acmodel = antagonist_acmodel
        self.discmodel = discmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.protagonist_ac_recurrence = ac_recurrence
        self.antagonist_ac_recurrence = ac_recurrence
        self.disc_recurrence = disc_recurrence
        self.preprocess_obss = default_preprocess_obss
        if preprocess_obss:
            self.preprocess_obss = preprocess_obss
            print("Customized observation preprocess")
        self.reshape_reward = reshape_reward
        self.entropy_reward = entropy_reward

        # Control parameters
        assert (self.protagonist_acmodel.recurrent or self.protagonist_ac_recurrence == 1) and (self.antagonist_acmodel.recurrent or self.antagonist_ac_recurrence == 1) and (self.discmodel.recurrent or self.disc_recurrence == 1)
        assert self.num_frames_per_proc % self.protagonist_ac_recurrence == 0 and self.num_frames_per_proc % self.antagonist_ac_recurrence == 0 and self.num_frames_per_proc % self.disc_recurrence == 0

        # Configure acmodel

        self.protagonist_acmodel.to(self.device)
        self.protagonist_acmodel.train()
        self.antagonist_acmodel.to(self.device)
        self.antagonist_acmodel.train()
        self.discmodel.to(self.device)
        self.discmodel.train()
        # Store helpers values

        assert len(protagonist_envs) == len(antagonist_envs)
        self.num_procs = len(protagonist_envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        
        self.num_demo_frames_per_proc = None
        self.num_demo_frames = None

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.protagonist_obs = self.protagonist_env.reset()
        self.protagonist_obss = [None] * (shape[0])
        if self.protagonist_acmodel.recurrent:
            self.protagonist_ac_memory = torch.zeros(shape[1], self.protagonist_acmodel.memory_size, device=self.device)
            self.protagonist_ac_memories = torch.zeros(*shape, self.protagonist_acmodel.memory_size, device=self.device)
            self.protagonist_antagonist_ac_memory = torch.zeros(shape[1], self.protagonist_acmodel.memory_size, device=self.device)
            self.protagonist_antagonist_ac_memories = torch.zeros(*shape, self.protagonist_acmodel.memory_size, device=self.device)
        self.antagonist_obs = self.antagonist_env.reset()
        self.antagonist_obss = [None] * (shape[0])
        if self.antagonist_acmodel.recurrent:
            self.antagonist_ac_memory = torch.zeros(shape[1], self.antagonist_acmodel.memory_size, device=self.device)
            self.antagonist_ac_memories = torch.zeros(*shape, self.antagonist_acmodel.memory_size, device=self.device)
            self.antagonist_protagonist_ac_memory = torch.zeros(shape[1], self.antagonist_acmodel.memory_size, device=self.device)
            self.antagonist_protagonist_ac_memories = torch.zeros(*shape, self.antagonist_acmodel.memory_size, device=self.device)
         

        if self.discmodel.recurrent:
            self.protagonist_disc_memory = torch.zeros(shape[1], self.discmodel.memory_size, device=self.device)
            self.protagonist_disc_memories = torch.zeros(*shape, self.discmodel.memory_size, device=self.device)
            self.antagonist_disc_memory = torch.zeros(shape[1], self.discmodel.memory_size, device=self.device)
            self.antagonist_disc_memories = torch.zeros(*shape, self.discmodel.memory_size, device=self.device)
             
        self.protagonist_mask = torch.ones(shape[1], device=self.device)
        self.protagonist_masks = torch.zeros(*shape, device=self.device)
        self.antagonist_mask = torch.ones(shape[1], device=self.device)
        self.antagonist_masks = torch.zeros(*shape, device=self.device)
        self.protagonist_actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.antagonist_actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.protagonist_values = torch.zeros(*shape, device=self.device)
        self.antagonist_values = torch.zeros(*shape, device=self.device)
        self.protagonist_rewards = torch.zeros(*shape, device=self.device)
        self.antagonist_rewards = torch.zeros(*shape, device=self.device)
        self.protagonist_ori_rewards = torch.zeros(*shape, device=self.device)
        self.antagonist_ori_rewards = torch.zeros(*shape, device=self.device)
        self.protagonist_advantages = torch.zeros(*shape, device=self.device)
        self.antagonist_advantages = torch.zeros(*shape, device=self.device)
        self.protagonist_log_probs = torch.zeros(*shape, device=self.device)
        self.antagonist_protagonist_log_probs = torch.zeros(self.protagonist_log_probs.shape, device=self.device)
        self.antagonist_log_probs = torch.zeros(*shape, device=self.device)
        self.protagonist_antagonist_log_probs = torch.zeros(self.antagonist_log_probs.shape, device=self.device)
        self.protagonist_logits = torch.zeros(*shape, protagonist_envs[0].action_space.n, device=self.device)
        self.antagonist_protagonist_logits = torch.zeros(self.protagonist_logits.shape, device=self.device)
        self.antagonist_logits = torch.zeros(*shape, antagonist_envs[0].action_space.n,  device=self.device)
        self.protagonist_antagonist_logits = torch.zeros(self.antagonist_logits.shape, device=self.device)
        
        
        # Initialize log values

        self.protagonist_log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.protagonist_log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.protagonist_log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.protagonist_log_done_counter = 0
        self.protagonist_log_return = [0] * self.num_procs
        self.protagonist_log_reshaped_return = [0] * self.num_procs
        self.protagonist_log_num_frames = [0] * self.num_procs

        self.antagonist_log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.antagonist_log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.antagonist_log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.antagonist_log_done_counter = 0
        self.antagonist_log_return = [0] * self.num_procs
        self.antagonist_log_reshaped_return = [0] * self.num_procs
        self.antagonist_log_num_frames = [0] * self.num_procs

    def init_demonstrations(self, demons_dict):
        demos = DictList()
        self.num_demo_frames_per_proc = len(demons_dict['obs'])
        self.num_demo_frames = self.num_demo_frames_per_proc * 1
         
        demos.obs = demons_dict['obs']

        if self.discmodel.recurrent:
            demos.disc_memory = torch.zeros(len(demos.obs), self.discmodel.memory_size).to(self.device)
        demos.mask = 1. - torch.tensor(demons_dict['done']).unsqueeze(1).to(self.device)
        demos.action = torch.tensor(demons_dict['action']).to(self.device)
        demos.obs = self.preprocess_obss(demos.obs, device=self.device)
        return demos
    
    def collect_demonstrations(self, demos):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        protagonist_ac_memory = torch.zeros(1, self.protagonist_ac_memory.shape[-1], device= self.device) 
        protagonist_log_probs = torch.zeros(len(demos), device= self.device) 
        protagonist_logits = torch.zeros(len(demos), self.protagonist_logits.shape[-1], device= self.device) 
        
        antagonist_ac_memory = torch.zeros(1,self.antagonist_ac_memory.shape[-1], device= self.device) 
        antagonist_log_probs = torch.zeros(len(demos), device= self.device) 
        antagonist_logits = torch.zeros(len(demos), self.antagonist_logits.shape[-1], device= self.device) 

        disc_memory = torch.zeros(self.discmodel.memory_size, device= self.device) 

        for i in range(self.num_demo_frames):
            # Do one agent-environment interaction
            preprocessed_obs = demos.obs[i:i+1]
            action = demos.action[i]
            with torch.no_grad():
                if self.protagonist_acmodel.recurrent:
                    protagonist_dist, _, protagonist_ac_memory = self.protagonist_acmodel(preprocessed_obs, protagonist_ac_memory * demos.mask[i])
                else:
                    protagonist_dist, _ = self.protagonist_acmodel(preprocessed_obs)
                protagonist_log_probs[i] = protagonist_dist.log_prob(action)
                protagonist_logits[i] = protagonist_dist.logits

                if self.antagonist_acmodel.recurrent:
                    antagonist_dist, _, antagonist_ac_memory = self.antagonist_acmodel(preprocessed_obs, antagonist_ac_memory * demos.mask[i])
                else:
                    antagonist_dist, _ = self.antagonist_acmodel(preprocessed_obs)
                antagonist_log_probs[i] = antagonist_dist.log_prob(action)
                antagonist_logits[i] = antagonist_dist.logits

                if self.discmodel.recurrent:
                    disc_prob, disc_memory = self.discmodel(preprocessed_obs, action, disc_memory * demos.mask[i])
                    demos.disc_memory[i] = disc_memory
                else:
                    disc_prob = self.discmodel(preprocessed_obs, action, disc_memory * demos.mask[i])
                #disc_prob = disc_dist.log_prob(action).exp()
                

        
        demos.protagonist_logits = protagonist_logits.reshape(-1, self.protagonist_env.action_space.n)
        demos.antagonist_logits = antagonist_logits.reshape(-1, self.protagonist_env.action_space.n)
        demos.protagonist_log_prob = protagonist_log_probs.reshape(-1)
        demos.antagonist_log_prob = antagonist_log_probs.reshape(-1)
        return demos
 


    def collect_protagonist_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        
        
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.protagonist_obs, device=self.device)
            with torch.no_grad():
                if self.protagonist_acmodel.recurrent:
                    dist, value, ac_memory = self.protagonist_acmodel(preprocessed_obs, self.protagonist_ac_memory * self.protagonist_mask.unsqueeze(1))
                else:
                    dist, value = self.protagonist_acmodel(preprocessed_obs)
                if self.antagonist_acmodel.recurrent:
                    antagonist_dist, _, antagonist_ac_memory = self.antagonist_acmodel(preprocessed_obs, self.antagonist_protagonist_ac_memory * self.protagonist_mask.unsqueeze(1))
                else:
                    antagonist_dist, _ = self.antagonist_acmodel(preprocessed_obs)
            #print(antagonist_dist)
            action = dist.sample()

            obs, ori_reward, done, info = self.protagonist_env.step(action.cpu().numpy())
            with torch.no_grad():
                disc_prob, disc_memory = self.discmodel(preprocessed_obs, action, self.protagonist_disc_memory * self.protagonist_mask.unsqueeze(1))
            #disc_prob = disc_dist.log_prob(action).exp()

            reward = (torch.clamp((1 / disc_prob - 1).flatten() * antagonist_dist.log_prob(action).exp().flatten(), min = 1e-6).log() - (dist.log_prob(action) if self.entropy_reward else 0.)).detach().cpu().numpy().tolist() 
             
             
            # Update experiences values

            self.protagonist_obss[i] = self.protagonist_obs
            self.protagonist_obs = obs
            if self.protagonist_acmodel.recurrent:
                self.protagonist_ac_memories[i] = self.protagonist_ac_memory
                self.protagonist_ac_memory = ac_memory
            if self.antagonist_acmodel.recurrent:
                self.antagonist_protagonist_ac_memories[i] = self.antagonist_protagonist_ac_memory
                self.antagonist_protagonist_ac_memory = antagonist_ac_memory
            if self.discmodel.recurrent:
                self.protagonist_disc_memories[i] = self.protagonist_disc_memory
                self.protagonist_disc_memory = disc_memory
            self.protagonist_masks[i] = self.protagonist_mask
            self.protagonist_mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.protagonist_actions[i] = action
            self.protagonist_values[i] = value
            if self.reshape_reward is not None:
                self.protagonist_rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
                self.protagonist_ori_rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, ori_reward_, done_)
                    for obs_, action_, ori_reward_, done_ in zip(obs, action, ori_reward, done)
                ], device=self.device)
            else:
                self.protagonist_rewards[i] = torch.tensor(reward, device=self.device)
                self.protagonist_ori_rewards[i] = torch.tensor(ori_reward, device=self.device)
            self.protagonist_log_probs[i] = dist.log_prob(action)
            self.protagonist_logits[i] = dist.logits 
            self.antagonist_protagonist_log_probs[i] = antagonist_dist.log_prob(action)
            self.antagonist_protagonist_logits[i] = antagonist_dist.logits 
            # Update log values

            self.protagonist_log_episode_return += torch.tensor(ori_reward, device=self.device, dtype=torch.float)
            self.protagonist_log_episode_reshaped_return += self.protagonist_ori_rewards[i]
            self.protagonist_log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.protagonist_log_done_counter += 1
                    self.protagonist_log_return.append(self.protagonist_log_episode_return[i].item())
                    self.protagonist_log_reshaped_return.append(self.protagonist_log_episode_reshaped_return[i].item())
                    self.protagonist_log_num_frames.append(self.protagonist_log_episode_num_frames[i].item())

            self.protagonist_log_episode_return *= self.protagonist_mask
            self.protagonist_log_episode_reshaped_return *= self.protagonist_mask
            self.protagonist_log_episode_num_frames *= self.protagonist_mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.protagonist_obs, device=self.device)
        with torch.no_grad():
            if self.protagonist_acmodel.recurrent:
                _, next_value, _ = self.protagonist_acmodel(preprocessed_obs, self.protagonist_ac_memory * self.protagonist_mask.unsqueeze(1))
            else:
                _, next_value = self.protagonist_acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.protagonist_masks[i+1] if i < self.num_frames_per_proc - 1 else self.protagonist_mask
            next_value = self.protagonist_values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.protagonist_advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.protagonist_rewards[i] + self.discount * next_value * next_mask - self.protagonist_values[i]
            self.protagonist_advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.protagonist_obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.protagonist_acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.ac_memory = self.protagonist_ac_memories.transpose(0, 1).reshape(-1, *self.protagonist_ac_memories.shape[2:])
        if self.antagonist_acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.ac_memory_ = self.antagonist_protagonist_ac_memories.transpose(0, 1).reshape(-1, *self.antagonist_protagonist_ac_memories.shape[2:])
        
        if self.discmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.disc_memory = self.protagonist_disc_memories.transpose(0, 1).reshape(-1, *self.protagonist_disc_memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.protagonist_masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.protagonist_actions.transpose(0, 1).reshape(-1)
        exps.value = self.protagonist_values.transpose(0, 1).reshape(-1)
        exps.reward = self.protagonist_rewards.transpose(0, 1).reshape(-1)
        exps.ori_reward = self.protagonist_ori_rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.protagonist_advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.logits = self.protagonist_logits.transpose(0, 1).reshape(-1, self.protagonist_env.action_space.n)
        exps.logits_ = self.antagonist_protagonist_logits.transpose(0, 1).reshape(-1, self.protagonist_env.action_space.n)
        exps.log_prob = self.protagonist_log_probs.transpose(0, 1).reshape(-1)
        exps.log_prob_ = self.antagonist_protagonist_log_probs.transpose(0, 1).reshape(-1)
        # Preprocess experiences
        #print(exps.log_prob, exps.log_prob_)
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.protagonist_log_done_counter, self.num_procs)

        logs = {
            "protagonist_return_per_episode": self.protagonist_log_return[-keep:],
            "protagonist_reshaped_return_per_episode": self.protagonist_log_reshaped_return[-keep:],
            "protagonist_num_frames_per_episode": self.protagonist_log_num_frames[-keep:],
            "protagonist_num_frames": self.num_frames
        }

        self.protagonist_log_done_counter = 0
        self.protagonist_log_return = self.protagonist_log_return[-self.num_procs:]
        self.protagonist_log_reshaped_return = self.protagonist_log_reshaped_return[-self.num_procs:]
        self.protagonist_log_num_frames = self.protagonist_log_num_frames[-self.num_procs:]
        #print("protagonist episode return: ", logs["protagonist_return_per_episode"])
        return exps, logs

    

    def collect_antagonist_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        
        
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.antagonist_obs, device=self.device)
            with torch.no_grad():
                if self.antagonist_acmodel.recurrent:
                    dist, value, ac_memory = self.antagonist_acmodel(preprocessed_obs, self.antagonist_ac_memory * self.antagonist_mask.unsqueeze(1))
                else:
                    dist, value = self.antagonist_acmodel(preprocessed_obs)
                if self.protagonist_acmodel.recurrent:
                    protagonist_dist, _, protagonist_ac_memory = self.protagonist_acmodel(preprocessed_obs, self.protagonist_antagonist_ac_memory * self.antagonist_mask.unsqueeze(1))
                else:
                    protagonist_dist, _ = self.protagonist_acmodel(preprocessed_obs)

            action = dist.sample()

            obs, ori_reward, done, info = self.antagonist_env.step(action.cpu().numpy())
            with torch.no_grad():
                disc_prob, disc_memory = self.discmodel(preprocessed_obs, action, self.antagonist_disc_memory * self.antagonist_mask.unsqueeze(1))
            #disc_prob = disc_dist.log_prob(action).exp()
            reward = ((1 - disc_prob).log().flatten() -disc_prob.log().flatten()+ (dist.log_prob(action).flatten() if not self.entropy_reward else 0)).detach().cpu().numpy().tolist()#
            # Update experiences values

            self.antagonist_obss[i] = self.antagonist_obs
            self.antagonist_obs = obs
            if self.antagonist_acmodel.recurrent:
                self.antagonist_ac_memories[i] = self.antagonist_ac_memory
                self.antagonist_ac_memory = ac_memory
            if self.protagonist_acmodel.recurrent:
                self.protagonist_antagonist_ac_memories[i] = self.protagonist_antagonist_ac_memory
                self.protagonist_antagonist_ac_memory = protagonist_ac_memory
            if self.discmodel.recurrent:
                self.antagonist_disc_memories[i] = self.antagonist_disc_memory
                self.antagonist_disc_memory = disc_memory
            self.antagonist_masks[i] = self.antagonist_mask
            self.antagonist_mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.antagonist_actions[i] = action
            self.antagonist_values[i] = value
            if self.reshape_reward is not None:
                self.antagonist_rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
                self.antagonist_ori_rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, ori_reward_, done_)
                    for obs_, action_, ori_reward_, done_ in zip(obs, action, ori_reward, done)
                ], device=self.device)
            else:
                self.antagonist_rewards[i] = torch.tensor(reward, device=self.device)
                self.antagonist_ori_rewards[i] = torch.tensor(ori_reward, device=self.device)
            self.antagonist_log_probs[i] = dist.log_prob(action)
            self.antagonist_logits[i] = dist.logits
            self.protagonist_antagonist_log_probs[i] = protagonist_dist.log_prob(action)
            self.protagonist_antagonist_logits[i] = protagonist_dist.logits
            # Update log values

            self.antagonist_log_episode_return += torch.tensor(ori_reward, device=self.device, dtype=torch.float)
            self.antagonist_log_episode_reshaped_return += self.antagonist_ori_rewards[i]
            self.antagonist_log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.antagonist_log_done_counter += 1
                    self.antagonist_log_return.append(self.antagonist_log_episode_return[i].item())
                    self.antagonist_log_reshaped_return.append(self.antagonist_log_episode_reshaped_return[i].item())
                    self.antagonist_log_num_frames.append(self.antagonist_log_episode_num_frames[i].item())

            self.antagonist_log_episode_return *= self.antagonist_mask
            self.antagonist_log_episode_reshaped_return *= self.antagonist_mask
            self.antagonist_log_episode_num_frames *= self.antagonist_mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.antagonist_obs, device=self.device)
        with torch.no_grad():
            if self.antagonist_acmodel.recurrent:
                _, next_value, _ = self.antagonist_acmodel(preprocessed_obs, self.antagonist_ac_memory * self.antagonist_mask.unsqueeze(1))
            else:
                _, next_value = self.antagonist_acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.antagonist_masks[i+1] if i < self.num_frames_per_proc - 1 else self.antagonist_mask
            next_value = self.antagonist_values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.antagonist_advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.antagonist_rewards[i] + self.discount * next_value * next_mask - self.antagonist_values[i]
            self.antagonist_advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.antagonist_obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.antagonist_acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.ac_memory = self.antagonist_ac_memories.transpose(0, 1).reshape(-1, *self.antagonist_ac_memories.shape[2:])
        if self.protagonist_acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.ac_memory_ = self.protagonist_antagonist_ac_memories.transpose(0, 1).reshape(-1, *self.protagonist_antagonist_ac_memories.shape[2:])
        
        if self.discmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.disc_memory = self.antagonist_disc_memories.transpose(0, 1).reshape(-1, *self.antagonist_disc_memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.antagonist_masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.antagonist_actions.transpose(0, 1).reshape(-1)
        exps.value = self.antagonist_values.transpose(0, 1).reshape(-1)
        exps.reward = self.antagonist_rewards.transpose(0, 1).reshape(-1)
        exps.ori_reward = self.antagonist_ori_rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.antagonist_advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.logits = self.antagonist_logits.transpose(0, 1).reshape(-1, self.antagonist_env.action_space.n)
        exps.logits_ = self.protagonist_antagonist_logits.transpose(0, 1).reshape(-1, self.antagonist_env.action_space.n)
        exps.log_prob = self.antagonist_log_probs.transpose(0, 1).reshape(-1)
        exps.log_prob_ = self.protagonist_antagonist_log_probs.transpose(0, 1).reshape(-1)
        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.antagonist_log_done_counter, self.num_procs)

        logs = {
            "antagonist_return_per_episode": self.antagonist_log_return[-keep:],
            "antagonist_reshaped_return_per_episode": self.antagonist_log_reshaped_return[-keep:],
            "antagonist_num_frames_per_episode": self.antagonist_log_num_frames[-keep:],
            "antagonist_num_frames": self.num_frames
        }

        self.antagonist_log_done_counter = 0
        self.antagonist_log_return = self.antagonist_log_return[-self.num_procs:]
        self.antagonist_log_reshaped_return = self.antagonist_log_reshaped_return[-self.num_procs:]
        self.antagonist_log_num_frames = self.antagonist_log_num_frames[-self.num_procs:]
        #print("antagonist episode return: ", logs["antagonist_return_per_episode"])
        return exps, logs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.antagonist_obs, device=self.device)
            with torch.no_grad():
                if self.antagonist_acmodel.recurrent:
                    antagonist_dist, antagonist_value, antagonist_ac_memory = self.antagonist_acmodel(preprocessed_obs, self.antagonist_ac_memory * self.antagonist_mask.unsqueeze(1))
                else:
                    antagonist_dist, antagonist_value = self.antagonist_acmodel(preprocessed_obs)
            action = antagonist_dist.sample()

            obs, ori_reward, done, info = self.antagonist_env.step(action.cpu().numpy())
            with torch.no_grad():
                disc_prob, disc_memory = self.discmodel(preprocessed_obs, action, self.antagonist_disc_memory * self.antagonist_mask.unsqueeze(1))
            #disc_prob = disc_dist.log_prob(action).exp()
            
            reward = ori_reward #((1 - disc_prob).log() - disc_prob.log()).flatten().detach().cpu().numpy().tolist()
            # Update experiences values



            self.antagonist_obss[i] = self.antagonist_obs
            self.antagonist_obs = obs
            if self.antagonist_acmodel.recurrent:
                self.antagonist_ac_memories[i] = self.antagonist_ac_memory
                self.antagonist_ac_memory = antagonist_ac_memory
            if self.discmodel.recurrent:
                self.antagonist_disc_memories[i] = self.antagonist_disc_memory
                self.antagonist_disc_memory = disc_memory
            self.antagonist_masks[i] = self.antagonist_mask
            self.antagonist_mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.antagonist_actions[i] = action
            self.antagonist_values[i] = antagonist_value
            if self.reshape_reward is not None:
                self.antagonist_rewards[i] = torch.tensor([
                    self.antagonist_reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
                self.antagonist_ori_rewards[i] = torch.tensor([
                    self.antagonist_reshape_reward(obs_, action_, ori_reward_, done_)
                    for obs_, action_, ori_reward_, done_ in zip(obs, action, ori_reward, done)
                ], device=self.device)
            else:
                self.antagonist_rewards[i] = torch.tensor(reward, device=self.device)
                self.antagonist_ori_rewards[i] = torch.tensor(ori_reward, device=self.device)
            self.antagonist_log_probs[i] = antagonist_dist.log_prob(action)

            # Update log values

            self.antagonist_log_episode_return += torch.tensor(ori_reward, device=self.device, dtype=torch.float)
            self.antagonist_log_episode_reshaped_return += self.antagonist_ori_rewards[i]
            self.antagonist_log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.antagonist_log_done_counter += 1
                    self.antagonist_log_return.append(self.antagonist_log_episode_return[i].item())
                    self.antagonist_log_reshaped_return.append(self.antagonist_log_episode_reshaped_return[i].item())
                    self.antagonist_log_num_frames.append(self.antagonist_log_episode_num_frames[i].item())

            self.antagonist_log_episode_return *= self.antagonist_mask
            self.antagonist_log_episode_reshaped_return *= self.antagonist_mask
            self.antagonist_log_episode_num_frames *= self.antagonist_mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.antagonist_obs, device=self.device)
        with torch.no_grad():
            if self.antagonist_acmodel.recurrent:
                _, next_value, _ = self.antagonist_acmodel(preprocessed_obs, self.antagonist_ac_memory * self.antagonist_mask.unsqueeze(1))
            else:
                _, next_value = self.antagonist_acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.antagonist_masks[i+1] if i < self.num_frames_per_proc - 1 else self.antagonist_mask
            next_value = self.antagonist_values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.antagonist_advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.antagonist_rewards[i] + self.discount * next_value * next_mask - self.antagonist_values[i]
            self.antagonist_advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.antagonist_obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.antagonist_acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.ac_memory = self.antagonist_ac_memories.transpose(0, 1).reshape(-1, *self.antagonist_ac_memories.shape[2:])
        if self.discmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.disc_memory = self.antagonist_disc_memories.transpose(0, 1).reshape(-1, *self.antagonist_disc_memories.shape[2:])
        # T x P -> P x T -> (P * T) x 1
        exps.mask = self.antagonist_masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.antagonist_actions.transpose(0, 1).reshape(-1)
        exps.value = self.antagonist_values.transpose(0, 1).reshape(-1)
        exps.reward = self.antagonist_rewards.transpose(0, 1).reshape(-1)
        exps.ori_reward = self.antagonist_ori_rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.antagonist_advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.antagonist_log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.antagonist_log_done_counter, self.num_procs)

        logs = {
            "antagonist_return_per_episode": self.antagonist_log_return[-keep:],
            "antagonist_reshaped_return_per_episode": self.antagonist_log_reshaped_return[-keep:],
            "antagonist_num_frames_per_episode": self.antagonist_log_num_frames[-keep:],
            "antagonist_num_frames": self.num_frames
        }

        self.antagonist_log_done_counter = 0
        self.antagonist_log_return = self.antagonist_log_return[-self.num_procs:]
        self.antagonist_log_reshaped_return = self.antagonist_log_reshaped_return[-self.num_procs:]
        self.antagonist_log_num_frames = self.antagonist_log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_ac_parameters(self):
        pass

    @abstractmethod
    def update_disc_parameters(self):
        pass
