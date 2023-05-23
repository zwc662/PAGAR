import numpy as np
import torch
import torch.nn.functional as F

import sys, os
python_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'torch-pairl/')
sys.path.append(python_path)
import torch_pairl
from torch_pairl.algos import BaseAlgo

from utils import clip_grad_value


class PGAILAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""
    def __init__(self, protagonist_envs, antagonist_envs, protagonist_acmodel, antagonist_acmodel, discmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, ac_recurrence=4, disc_recurrence = 4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, pair_coef = 1.e-3, preprocess_obss=None,
                 reshape_reward=None, entropy_reward = False):
        
        num_frames_per_proc = num_frames_per_proc or 128
        
        super(PGAILAlgo, self).__init__(protagonist_envs, antagonist_envs, protagonist_acmodel, antagonist_acmodel, discmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, ac_recurrence, disc_recurrence, preprocess_obss, reshape_reward, entropy_reward)

        self.pair_coef = pair_coef
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        
        assert self.batch_size % self.antagonist_ac_recurrence == 0 and self.batch_size % self.protagonist_ac_recurrence == 0 and self.batch_size % self.disc_recurrence == 0
        
        self.protagonist_ac_optimizer = torch.optim.Adam(self.protagonist_acmodel.parameters(), lr, eps=adam_eps)
        self.antagonist_ac_optimizer = torch.optim.Adam(self.antagonist_acmodel.parameters(), lr, eps=adam_eps)
        self.disc_optimizer = torch.optim.Adam(self.discmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0


    def update_antagonist_ac_parameters(self, antagonist_exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes(self.antagonist_ac_recurrence):
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.antagonist_acmodel.recurrent:
                    memory = antagonist_exps.ac_memory[inds]

                for i in range(self.antagonist_ac_recurrence):
                    # Create a sub-batch of experience

                    sb = antagonist_exps[inds + i]

                    # Compute loss

                    if self.antagonist_acmodel.recurrent:
                        dist, value, memory = self.antagonist_acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.antagonist_acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.antagonist_acmodel.recurrent and i < self.antagonist_ac_recurrence - 1:
                        antagonist_exps.ac_memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.antagonist_ac_recurrence
                batch_value /= self.antagonist_ac_recurrence
                batch_policy_loss /= self.antagonist_ac_recurrence
                batch_value_loss /= self.antagonist_ac_recurrence
                batch_loss /= self.antagonist_ac_recurrence

                # Update actor-critic

                self.antagonist_ac_optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.antagonist_acmodel.parameters()) ** 0.5
                clip_grad_value(self.antagonist_acmodel.parameters(), self.max_grad_norm)
                self.antagonist_ac_optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "antagonist_entropy": np.mean(log_entropies),
            "antagonist_value": np.mean(log_values),
            "antagonist_policy_loss": np.mean(log_policy_losses),
            "antagonist_value_loss": np.mean(log_value_losses),
            "antagonist_ac_grad_norm": np.mean(log_grad_norms)
        }

        return logs
    
    def update_protagonist_ac_parameters(self, protagonist_exps, antagonist_exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes(self.protagonist_ac_recurrence):
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.protagonist_acmodel.recurrent:
                    memory = protagonist_exps.ac_memory[inds]
                if self.antagonist_acmodel.recurrent:
                    protagonist_antagonist_memory = antagonist_exps.ac_memory_[inds]

                for i in range(self.protagonist_ac_recurrence):
                    # Create a sub-batch of experience
                    sb = protagonist_exps[inds + i]
                    # Compute loss
                    if self.protagonist_acmodel.recurrent:
                        dist, value, memory = self.protagonist_acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.protagonist_acmodel(sb.obs)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()
                    

                    loss = policy_loss / self.protagonist_ac_recurrence - \
                        self.entropy_coef * entropy / self.protagonist_ac_recurrence + \
                            self.value_loss_coef * value_loss / self.protagonist_ac_recurrence

                    antagonist_sb = antagonist_exps[inds + i] 
                    if self.antagonist_acmodel.recurrent:
                        protagonist_antagonist_dist, _, protagonist_antagonist_memory = self.protagonist_acmodel(antagonist_sb.obs, protagonist_antagonist_memory * antagonist_sb.mask)
                    else:
                        protagonist_antagonist_dist, _ = self.protagonist_acmodel(antagonist_sb.obs)
                    protagonist_antagonist_entroy = protagonist_antagonist_dist.entropy().mean()
                    protagonist_antagonist_ratio = torch.exp(protagonist_antagonist_dist.log_prob(antagonist_sb.action) - antagonist_sb.log_prob.detach())
                    protagonist_antagonist_sur1 = protagonist_antagonist_ratio * antagonist_sb.advantage.detach()
                    protagonist_antagonist_sur2 = torch.clamp(protagonist_antagonist_ratio, 1. - self.clip_eps, 1. + self.clip_eps) * antagonist_sb.advantage.detach()
                    protagonist_antagonist_policy_loss = -torch.min(protagonist_antagonist_sur1, protagonist_antagonist_sur2).mean()

                    loss = loss + protagonist_antagonist_policy_loss / self.protagonist_ac_recurrence - \
                        self.entropy_coef * protagonist_antagonist_entroy / self.protagonist_ac_recurrence

                    # Update batch values

                    batch_entropy += entropy.item() + protagonist_antagonist_entroy.item()
                    batch_value += value.mean().item()  
                    batch_policy_loss += policy_loss.item() + protagonist_antagonist_policy_loss.item()
                    batch_value_loss += value_loss.item() 
                    batch_loss += loss


                    if self.protagonist_acmodel.recurrent and i < self.protagonist_ac_recurrence - 1:
                        protagonist_exps.ac_memory[inds + i + 1] = memory.detach()
                        antagonist_exps.ac_memory_[inds + i + 1] = protagonist_antagonist_memory.detach()
                 
                # Update batch values

                batch_entropy /= self.protagonist_ac_recurrence
                batch_value /= self.protagonist_ac_recurrence
                batch_policy_loss /= 2 * self.protagonist_ac_recurrence
                batch_value_loss /= self.protagonist_ac_recurrence
      
                # Update actor-critic

                self.protagonist_ac_optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.protagonist_acmodel.parameters()) ** 0.5
                clip_grad_value(self.protagonist_acmodel.parameters(), self.max_grad_norm)
                self.protagonist_ac_optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "protagonist_entropy": np.mean(log_entropies),
            "protagonist_value": np.mean(log_values),
            "protagonist_policy_loss": np.mean(log_policy_losses),
            "protagonist_value_loss": np.mean(log_value_losses),
            "protagonist_ac_grad_norm": np.mean(log_grad_norms)
        }

        return logs

    def update_ac_parameters(self, protagonist_exps, antagonist_exps):
        logs3 = self.update_protagonist_ac_parameters(protagonist_exps, antagonist_exps)
        logs4 = self.update_antagonist_ac_parameters(antagonist_exps)
        return logs3, logs4

    def update_disc_parameters(self, protagonist_exps, antagonist_exps, demos):
        # Collect experiences
        antagonist_exps_acc = []
        protagonist_exps_acc = []
        demos_acc = []
        for _ in range(self.epochs):
            # Initialize log values
 
            log_irl_losses = []
            log_pair_losses = []
            log_grad_norms = []

            antagonist_exps_inds = np.arange(0, len(antagonist_exps), self.disc_recurrence)
            demos_inds = np.arange(0, len(demos), self.disc_recurrence)
            protagonist_exps_inds = np.arange(0, len(protagonist_exps), self.disc_recurrence)
            
            if True:
                # Initialize batch values
                batch_irl_loss = 0
                batch_pair_loss = 0
                # Initialize memory

                if self.discmodel.recurrent:
                    antagonist_exps_memory = antagonist_exps.disc_memory[antagonist_exps_inds]
                    protagonist_exps_memory = protagonist_exps.disc_memory[protagonist_exps_inds]
                    demos_memory = demos.disc_memory[demos_inds]
                
                for i in range(self.disc_recurrence):
                    # Create a sub-batch of experience
                    antagonist_exps_sb = antagonist_exps[antagonist_exps_inds + i]
                    protagonist_exps_sb = protagonist_exps[protagonist_exps_inds + i]
                    demos_sb = demos[demos_inds + i]
                    # Compute loss

                    if self.discmodel.recurrent:
                        antagonist_exps_learner, antagonist_exps_memory = self.discmodel(antagonist_exps_sb.obs, antagonist_exps_sb.action.to(self.device), antagonist_exps_memory.to(self.device) * antagonist_exps_sb.mask.to(self.device))
                        demos_learner, demos_memory = self.discmodel(demos_sb.obs, demos_sb.action, demos_memory * demos_sb.mask)
                        protagonist_exps_learner, protagonist_exps_memory = self.discmodel(protagonist_exps_sb.obs, protagonist_exps_sb.action.to(self.device), protagonist_exps_memory.to(self.device) * antagonist_exps_sb.mask.to(self.device))
                    else:
                        antagonist_exps_learner = self.discmodel(antagonist_exps_sb.obs, antagonist_exps_sb.action.to(self.device))
                        demos_learner = self.discmodel(demos_sb.obs, demos_sb.action)
                        protagonist_exps_learner = self.discmodel(protagonist_exps_sb.obs, protagonist_exps_sb.action.to(self.device))
                    

                    #antagonist_exps_learner = antagonist_exps_learner.log_prob(antagonist_exps_sb.action.to(self.device)).exp().reshape(-1, 1)
                    #demos_learner = demos_learner.log_prob(demos_sb.action).exp().reshape(-1, 1)
                    #protagonist_exps_learner = protagonist_exps_learner.log_prob(protagonist_exps_sb.action.to(self.device)).exp().reshape(-1, 1)
                    
                    antagonist_exps_acc.append(antagonist_exps_learner.mean().detach().item())
                    protagonist_exps_acc.append(protagonist_exps_learner.mean().detach().item())
                    demos_acc.append(demos_learner.mean().detach().item())
                    criterion = torch.nn.BCELoss()
                    loss = criterion(antagonist_exps_learner, torch.ones((antagonist_exps_learner.shape[0], 1)).to(self.device)) + \
                        criterion(demos_learner, torch.zeros((demos_learner.shape[0], 1)).to(self.device))
                    # Update batch values

                   
                    batch_irl_loss += loss
                    # Update memories for next epoch

                    if self.discmodel.recurrent and i < self.disc_recurrence - 1:
                        antagonist_exps.disc_memory[antagonist_exps_inds + i + 1] = antagonist_exps_memory.detach()
                        protagonist_exps.disc_memory[protagonist_exps_inds + i + 1] = protagonist_exps_memory.detach()
                        demos.disc_memory[demos_inds +i + 1] = demos_memory.detach()
                    
                    
                    pair_r1 = - ((protagonist_exps_sb.log_prob_.exp() / protagonist_exps_learner - protagonist_exps_sb.log_prob_.exp()).log())
                     
                    pair_ratio1 = torch.exp(protagonist_exps_sb.log_prob_ - protagonist_exps_sb.log_prob).detach()
                    #print(pair_ratio1)
                    pair_loss1 = (pair_r1 * pair_ratio1)
                    pair_loss1 = pair_loss1[torch.isfinite(pair_loss1)].mean() 
                    pair_tv1 = torch.square((protagonist_exps_sb.logits.exp() - protagonist_exps_sb.logits_.exp()).abs().sum(dim = 1).mean()).detach().item()
                    pair_loss1 = pair_loss1 + pair_tv1 * 4 * self.discount / (1 - self.discount) * torch.abs(pair_r1.flatten()).max()
                    pair_loss1 = pair_loss1 - pair_r1[torch.isfinite(pair_r1)].mean()

                    pair_r2 = ((antagonist_exps_sb.log_prob_.exp() / antagonist_exps_learner - antagonist_exps_sb.log_prob_.exp()).log())
                    pair_ratio2 = torch.exp(antagonist_exps_sb.log_prob_ - antagonist_exps_sb.log_prob).detach()
                    pair_loss2 = (pair_r2 * pair_ratio2)
                    pair_loss2 = pair_loss2[torch.isfinite(pair_loss2)].mean() 
                    pair_tv2 = torch.square((antagonist_exps_sb.logits.exp() - antagonist_exps_sb.logits_.exp()).abs().sum(dim=1).mean()).detach().item()
                    pair_loss2 = pair_loss2 - pair_tv2 * 4 * self.discount / (1 - self.discount) * torch.abs(pair_r2.flatten()).max()
                    pair_loss2 = pair_loss2 - pair_r2[torch.isfinite(pair_r2)].mean() 
                    
                    pair_ratio3 = (torch.exp(pair_r2 - antagonist_exps_sb.log_prob.detach()))
                    pair_ids3 = (pair_ratio3 <=  1. + self.clip_eps).float() * (pair_ratio3 >=  1. - self.clip_eps).float()
                    pair_clipped_ratio3 = pair_ratio3 * pair_ids3.detach()
                    pair_loss3 = - (pair_r2 * pair_clipped_ratio3)
                    pair_loss3 = pair_loss3[torch.isfinite(pair_loss3)].sum() / pair_ids3[torch.isfinite(pair_loss3)].sum()
                    pair_loss3 = pair_loss3 - pair_r1[torch.isfinite(pair_r1)].mean()

                    pair_ratio4 = (torch.exp(- pair_r1 - protagonist_exps_sb.log_prob.detach()))
                    pair_ids4 = (pair_ratio4 <=  1. + self.clip_eps).float() * (pair_ratio4 >=  1. - self.clip_eps).float()
                    pair_clipped_ratio4 = pair_ratio4 * pair_ids4.detach()
                    pair_loss4 =  - (pair_r1 * pair_clipped_ratio4)
                    pair_loss4 = pair_loss4[torch.isfinite(pair_loss4)].sum() / pair_ids4[torch.isfinite(pair_loss4)].sum()  
                    pair_loss4 = pair_loss4 - pair_r2[torch.isfinite(pair_r1)].mean()  

                    r = (demos_sb.antagonist_log_prob.exp().flatten() /demos_learner.flatten() - demos_sb.antagonist_log_prob.exp().flatten()).log()
                    ratio = ((demos_sb.antagonist_log_prob.exp() - demos_sb.protagonist_log_prob.exp()).flatten().detach()) 
                    #pair_loss0 =  (r * ratio)
                    #ratio = (protagonist_expert_learner - antagonist_expert_learner).detach()  / r.exp().detach()
                    pair_loss0 =  (r * ratio)[ratio < 0]
                    pair_loss0 = - pair_loss0[torch.isfinite(pair_loss0)].log().mean().exp()
                    
                    pair_loss = pair_loss1 + pair_loss2 + (pair_loss3 if torch.isfinite(pair_loss3).all() else 0.) + (pair_loss4 if torch.isfinite(pair_loss4).all() else 0.) 
                    #print(pair_loss1, pair_loss2, pair_loss3, pair_loss0)
                    batch_pair_loss += pair_loss
            
                
                # Update batch values
                
                batch_irl_loss /= self.disc_recurrence
                batch_pair_loss /= self.disc_recurrence
                # Update actor-critic

                self.disc_optimizer.zero_grad()
                self.pair_coef = self.pair_coef * np.exp((- batch_irl_loss.detach().cpu().numpy().item() + 1.2))
                (batch_pair_loss * self.pair_coef + batch_irl_loss).backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.discmodel.parameters()) ** 0.5
                clip_grad_value(self.discmodel.parameters(), self.max_grad_norm)
                self.disc_optimizer.step()

                # Update log values
 
                log_irl_losses.append(batch_irl_loss.data.detach().cpu().numpy())
                log_pair_losses.append(batch_pair_loss.data.detach().cpu().numpy())
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "protagonist_exps_acc": np.mean(protagonist_exps_acc),
            "antagonist_exps_acc": np.mean(antagonist_exps_acc),
            "demos_acc": np.mean(demos_acc),
            "irl_loss": np.mean(log_irl_losses),
            "pair_loss": np.mean(log_pair_losses),
            "disc_grad_norm": np.mean(log_grad_norms)
        }

        return logs
 
    def _get_batches_starting_indexes(self, recurrence, num_frames = None, capacity = None):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """
        num_frames = self.num_frames if num_frames is None else num_frames
        capacity = self.num_frames_per_proc if capacity is None else capacity
        indexes = np.arange(0, num_frames, recurrence)
        indexes = np.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + recurrence) % capacity != 0]
            indexes += recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
    