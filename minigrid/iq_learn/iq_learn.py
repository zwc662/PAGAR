import numpy as np
import torch
import torch.nn.functional as F

import sys, os
python_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'torch-irl/')
sys.path.append(python_path)
 
from torch_irl.algos.base import BaseAlgo
from torch.autograd import Variable, grad



class IQAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, discmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, ac_recurrence=4, disc_recurrence = 4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128
        
        super().__init__(envs, acmodel, discmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, ac_recurrence, disc_recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.div = 'chi'
        self.alpha = 0.5
        self.loss = "value"
        self.regularize = True
        self.lambda_gp = 10
        self.grad_pen = False

        assert self.batch_size % self.ac_recurrence == 0 and self.batch_size % self.disc_recurrence == 0
        
        self.ac_optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.disc_optimizer = torch.optim.Adam(self.discmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0

    def jacobian(self, outputs, inputs):
        """Computes the jacobian of outputs with respect to inputs

        :param outputs: tensor for the output of some function
        :param inputs: tensor for the input of some function (probably a vector)
        :returns: a tensor containing the jacobian of outputs with respect to inputs
        """
        batch_size, output_dim = outputs.shape
        jacobian = []
        for i in range(output_dim):
            v = torch.zeros_like(outputs)
            v[:, i] = 1.
            dy_i_dx = grad(outputs,
                           inputs,
                           grad_outputs=v,
                           retain_graph=True,
                           create_graph=True)[0]  # shape [B, N]
            jacobian.append(dy_i_dx)

        jacobian = torch.stack(jacobian, dim=-1).requires_grad_()
        return jacobian

    # Full IQ-Learn objective with other divergences and options
    def grad_pen(self, obs1, action1, memory1, obs2, action2, memory2, lambda_=1):
        expert_obs = obs1
        policy_obs = obs2
        batch_size = expert_obs.size()[0]

        expert_memory = memory1
        policy_memory = memory2

        # Calculate interpolation
        if expert_obs.ndim == 4:
            alpha = torch.rand(batch_size, 1, 1, 1)  # B, C, H, W input
        else:
            alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(expert_obs).to(expert_obs.device)

        interpolated_obs = alpha * expert_obs + (1 - alpha) * policy_obs
        interpolated_obs = Variable(interpolated_obs, requires_grad=True)
        interpolated_obs = interpolated_obs.to(expert_obs.device)

        interpolated_memory = alpha * expert_memory + (1 - alpha) * policy_memory
        interpolated_memory = Variable(interpolated_memory, requires_grad=True)
        interpolated_memory = interpolated_memory.to(expert_memory.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.forward(interpolated_obs, interpolated_memory)
        # Calculate gradients of probabilities with respect to examples
        gradients = self.jacobian(prob_interpolated, interpolated_obs)

        # Gradients have shape (batch_size, input_dim, output_dim)
        out_size = gradients.shape[-1]
        gradients_norm = gradients.reshape([batch_size, -1, out_size]).norm(2, dim=1)

        # Return gradient penalty
        return lambda_ * ((gradients_norm - 1) ** 2).mean()
     
    

    def update_parameters(self, exps, demos):
        # Collect experiences
        loss_dict = {
            'softq_loss': [], 
            'value_loss': [],
            'chi2_loss': [],
            'regularize_loss': [],
            'total_loss': []
        }

        for _ in range(self.epochs):
            # Initialize log values
  

            exps_inds = np.arange(0, len(exps), self.ac_recurrence)
            demos_inds = np.arange(0, len(demos), self.disc_recurrence)

            if True:
                # Initialize batch values
                batch_loss = 0

                # Initialize memory

                if self.discmodel.recurrent:
                    exps_memory = exps.disc_memory[exps_inds]
                    demos_memory = demos.disc_memory[demos_inds]
                for i in range(self.disc_recurrence):
                    # Create a sub-batch of experience
                    exps_sb = exps[exps_inds + i]
                    demos_sb = demos[demos_inds + i]
                    # Compute loss

                    if self.discmodel.recurrent:
                        exps_learner, exps_val, exps_memory = self.discmodel(exps_sb.obs, exps_sb.action.to(self.device).long(), exps_memory.to(self.device) * exps_sb.mask.to(self.device))
                        demos_learner, demos_val, demos_memory = self.discmodel(demos_sb.obs, demos_sb.action.to(self.device).long(), demos_memory * demos_sb.mask)
                    else:
                        exps_learner, exps_val = self.discmodel(exps_sb.obs, exps_sb.action.to(self.device))
                        demos_learner, demos_val = self.discmodel(demos_sb.obs, demos_sb.action)

                    exps_log_prob = exps_learner.log()#.log_prob(exps_sb.action.to(self.device)).exp().reshape(-1, 1)
                    demos_log_prob = demos_learner.log()#log_prob(demos_sb.action).exp().reshape(-1, 1)
                    
                    # keep track of value of initial states
                    v0 = demos_val.mean()
                    
                    #  calculate 1st term for IQ loss
                    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
                    demos_y = demos_sb[:-1].mask * self.discount * (demos_val - demos_log_prob)[1:].detach()
                    demos_reward = (demos_val[:-1] - demos_y) 

                    exps_y = exps_sb[:-1].mask * self.discount * (exps_val - exps_log_prob)[1:].detach()
                    exps_reward = (exps_val[:-1] - exps_y) 


                    with torch.no_grad():
                        # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
                        if self.div == "hellinger":
                            phi_grad = 1/(1+demos_reward)**2
                        elif self.div == "kl":
                            # original dual form for kl divergence (sub optimal)
                            phi_grad = torch.exp(-demos_reward-1)
                        elif self.div == "kl2":
                            # biased dual form for kl divergence
                            phi_grad = F.softmax(-demos_reward, dim=0) * demos_reward.shape[0]
                        elif self.div == "kl_fix":
                            # our proposed unbiased form for fixing kl divergence
                            phi_grad = torch.exp(-demos_reward)
                        elif self.div == "js":
                            # jensen–shannon
                            phi_grad = torch.exp(-demos_reward)/(2 - torch.exp(-demos_reward))
                        else:
                            phi_grad = 1
                    batch_loss = -(phi_grad * demos_reward).mean()
                    loss_dict['softq_loss'].append(batch_loss.item())

                    # calculate 2nd term for IQ loss, we show different sampling strategies
                    if self.loss == "value_expert":
                        # sample using only expert states (works offline)
                        # E_(ρ)[Q(s,a) - γV(s')]
                        value_loss = (demos_val[:-1] - demos_log_prob[:-1] - demos_y).mean()
                        batch_loss += value_loss
                        loss_dict['value_loss'].append(value_loss.item())

                    elif self.loss == "value":
                        # sample using expert and policy states (works online)
                        # E_(ρ)[V(s) - γV(s')]
                        value_loss = ((demos_val[:-1] - demos_log_prob[:-1] - demos_y).mean() + (exps_val[:-1] - exps_log_prob[:-1] - exps_y).mean()) * 0.5
                        batch_loss += value_loss
                        loss_dict['value_loss'].append(value_loss.item())

                    elif self.loss == "v0":
                        # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
                        # (1-γ)E_(ρ0)[V(s0)]
                        v0_loss = (1 - self.discount) * v0
                        batch_loss += v0_loss
                        loss_dict['v0_loss'].append(v0_loss.item())

                    # alternative sampling strategies for the sake of completeness but are usually suboptimal in practice
                    # elif args.method.loss == "value_policy":
                    #     # sample using only policy states
                    #     # E_(ρ)[V(s) - γV(s')]
                    #     value_loss = (current_v - y)[~is_expert].mean()
                    #     loss += value_loss
                    #     loss_dict['value_policy_loss'] = value_loss.item()

                    # elif args.method.loss == "value_mix":
                    #     # sample by weighted combination of expert and policy states
                    #     # E_(ρ)[Q(s,a) - γV(s')]
                    #     w = args.method.mix_coeff
                    #     value_loss = (w * (current_v - y)[is_expert] +
                    #                   (1-w) * (current_v - y)[~is_expert]).mean()
                    #     loss += value_loss
                    #     loss_dict['value_loss'] = value_loss.item()

                    else:
                        raise ValueError(f'This sampling method is not implemented')

                    if self.grad_pen:
                        # add a gradient penalty to loss (Wasserstein_1 metric)
                        gp_loss = self.grad_pen(
                            demos.obs, demos.acts, demos_memory,
                            exps.obs, exps.acts, exps_memory,
                            self.lambda_gp)
                        loss_dict['gp_loss'].append(gp_loss.item())
                        batch_loss += gp_loss

                    if self.div == "chi":  # TODO: Deprecate method.chi argument for method.div
                        # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
                        chi2_loss = 1/(4 * self.alpha) * (demos_reward**2).mean()
                        batch_loss += chi2_loss
                        loss_dict['chi2_loss'].append(chi2_loss.item())
                   

                    if self.regularize:
                        # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
    
                        chi2_loss = 1/(4 * self.alpha) * ((demos_reward**2).mean() + (exps_reward**2).mean()) * 0.5
                        batch_loss += chi2_loss
                        loss_dict['regularize_loss'].append(chi2_loss.item())

                    loss_dict['total_loss'].append(batch_loss.item())
 
                # Update batch values
 
                batch_loss /= self.disc_recurrence

                # Update actor-critic

                self.disc_optimizer.zero_grad()
                batch_loss.backward()
                #grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.discmodel.parameters()) ** 0.5
                #torch.nn.utils.clip_grad_norm_(self.discmodel.parameters(), self.max_grad_norm)
                self.disc_optimizer.step()

                # Update log values
 
                #log_losses.append(batch_loss.data.detach().cpu().numpy())
                #log_grad_norms.append(grad_norm)

        # Log some values
        for k,v in loss_dict.items():
            loss_dict[k] = np.mean(v)

        return loss_dict

    def update_ac_parameters(self, exps):
        # Collect experiences

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes(self.ac_recurrence):
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.ac_memory[inds]

                for i in range(self.ac_recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value = self.acmodel(sb.obs)

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

                    if self.acmodel.recurrent and i < self.ac_recurrence - 1:
                        exps.ac_memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.ac_recurrence
                batch_value /= self.ac_recurrence
                batch_policy_loss /= self.ac_recurrence
                batch_value_loss /= self.ac_recurrence
                batch_loss /= self.ac_recurrence

                # Update actor-critic

                self.ac_optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.ac_optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": np.mean(log_entropies),
            "value": np.mean(log_values),
            "policy_loss": np.mean(log_policy_losses),
            "value_loss": np.mean(log_value_losses),
            "ac_grad_norm": np.mean(log_grad_norms)
        }

        return logs


    def update_disc_parameters(self, exps, demos):
        # Collect experiences
        exps_acc = []
        demos_acc = []
        for _ in range(self.epochs):
            # Initialize log values
 
            log_losses = []
            log_grad_norms = []

            exps_inds = np.arange(0, len(exps), self.ac_recurrence)
            demos_inds = np.arange(0, len(demos), self.disc_recurrence)

            if True:
                # Initialize batch values
                batch_loss = 0

                # Initialize memory

                if self.discmodel.recurrent:
                    exps_memory = exps.disc_memory[exps_inds]
                    demos_memory = demos.disc_memory[demos_inds]
                for i in range(self.disc_recurrence):
                    # Create a sub-batch of experience
                    exps_sb = exps[exps_inds + i]
                    demos_sb = demos[demos_inds + i]
                    # Compute loss

                    if self.discmodel.recurrent:
                        exps_learner, exps_memory = self.discmodel(exps_sb.obs, exps_sb.action.to(self.device), exps_memory.to(self.device) * exps_sb.mask.to(self.device))
                        demos_learner, demos_memory = self.discmodel(demos_sb.obs, demos_sb.action, demos_memory * demos_sb.mask)
                    else:
                        exps_learner = self.discmodel(exps_sb.obs, exps_sb.action.to(self.device))
                        demos_learner = self.discmodel(demos_sb.obs, demos_sb.action)
                    exps_learner = exps_learner.log_prob(exps_sb.action.to(self.device)).exp().reshape(-1, 1)
                    demos_learner = demos_learner.log_prob(demos_sb.action).exp().reshape(-1, 1)

                    exps_acc.append(exps_learner.mean().detach().item())
                    demos_acc.append(demos_learner.mean().detach().item())
                    criterion = torch.nn.BCELoss()
                    loss = criterion(exps_learner, torch.ones((exps_learner.shape[0], 1)).to(self.device)) + \
                        criterion(demos_learner, torch.zeros((demos_learner.shape[0], 1)).to(self.device))
                    # Update batch values

                   
                    batch_loss += loss
                    # Update memories for next epoch

                    if self.discmodel.recurrent and i < self.disc_recurrence - 1:
                        exps.disc_memory[exps_inds + i + 1] = exps_memory.detach()
                        demos.disc_memory[demos_inds +i + 1] = demos_memory.detach()

                # Update batch values
 
                batch_loss /= self.disc_recurrence

                # Update actor-critic

                self.disc_optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.discmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.discmodel.parameters(), self.max_grad_norm)
                self.disc_optimizer.step()

                # Update log values
 
                log_losses.append(batch_loss.data.detach().cpu().numpy())
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "exps_acc": np.mean(exps_acc),
            "demos_acc": np.mean(demos_acc),
            "disc_loss": np.mean(log_losses),
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
