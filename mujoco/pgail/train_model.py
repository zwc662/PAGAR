import torch
from torch.distributions import Categorical, Normal
import numpy as np
from utils.utils import get_entropy, log_prob_density, get_reward, get_q_value, get_action, clip_grad_value, clip_grad_norm


def train_discrim(discrim, protagonist_memory, antagonist_memory, discrim_optim, demonstrations, args):
    protagonist_memory = np.array(protagonist_memory) 
    protagonist_states = np.vstack(protagonist_memory[:-1, 0]) 
    protagonist_nxt_states = np.vstack(protagonist_memory[1:, 0]) 
    protagonist_actions = np.vstack(protagonist_memory[:-1, 1]) 

    protagonist_states = torch.Tensor(protagonist_states)
    protagonist_nxt_states = torch.Tensor(protagonist_nxt_states)
    protagonist_actions = torch.Tensor(protagonist_actions)

    antagonist_memory = np.array(antagonist_memory) 
    antagonist_states = np.vstack(antagonist_memory[:-1, 0]) 
    antagonist_nxt_states = np.vstack(antagonist_memory[1:, 0]) 
    antagonist_actions = np.vstack(antagonist_memory[:-1, 1]) 

    antagonist_states = torch.Tensor(antagonist_states)
    antagonist_nxt_states = torch.Tensor(antagonist_nxt_states)
    antagonist_actions = torch.Tensor(antagonist_actions)
    

    demonstration_states = np.vstack(demonstrations[:-1, : int(antagonist_states.shape[-1])])
    demonstration_nxt_states = np.vstack(demonstrations[1:, : int(antagonist_states.shape[-1])])
    demonstration_actions = np.vstack(demonstrations[:-1, int(antagonist_states.shape[-1]): ])

    demonstration_states = torch.Tensor(demonstration_states)
    demonstration_nxt_states = torch.Tensor(demonstration_nxt_states)
    demonstration_actions = torch.Tensor(demonstration_actions)
    

    irl_criterion = torch.nn.BCELoss()

    for _ in range(args.reward_function_update_num):
        protagonist_learner = discrim(torch.cat([protagonist_states, protagonist_actions, protagonist_nxt_states], dim=1))
        #demonstration = torch.Tensor(demonstrations)
        antagonist_learner = discrim(torch.cat([antagonist_states, antagonist_actions, antagonist_nxt_states], dim=1)) 
        #demonstration = 
        expert = discrim(torch.Tensor(torch.cat([demonstration_states, demonstration_actions, demonstration_nxt_states], dim=1)))
        irl_loss = irl_criterion(protagonist_learner, torch.ones((protagonist_states.shape[0], 1))) + \
                        irl_criterion(expert, torch.zeros((demonstration_states.shape[0], 1)))  
        #pair_loss = criterion(protagonist_learner, torch.ones((protagonist_states.shape[0], 1))) + \
                        #criterion(antagonist_learner, torch.zeros((antagonist_states.shape[0], 1))) 
        pair_loss = protagonist_learner.mean() - antagonist_learner.mean()

        discrim_loss = irl_loss + args.pair_coef * pair_loss        
        discrim_optim.zero_grad()
        discrim_loss.backward()
        discrim_optim.step()

    expert_acc = ((discrim(torch.Tensor(torch.cat([demonstration_states, demonstration_actions, demonstration_nxt_states], dim=1))) > 0.5).float()).mean()
    protagonist_learner_acc = ((discrim(torch.cat([protagonist_states, protagonist_actions, protagonist_nxt_states], dim=1)) > 0.5).float()).mean()
    antagonist_learner_acc = ((discrim(torch.cat([antagonist_states, antagonist_actions, antagonist_nxt_states], dim=1))  > 0.5).float()).mean()

    return expert_acc, protagonist_learner_acc, antagonist_learner_acc


def train_reward_function(running_state, protagonist_actor, antagonist_actor, reward_function, q_value, protagonist_memory, antagonist_memory, reward_function_optim, q_value_optim, demonstrations, args):
    protagonist_memory = np.array(protagonist_memory) 
    protagonist_states = np.vstack(protagonist_memory[:, 0]) 
    protagonist_states = running_state(protagonist_states, update = False)
    #protagonist_nxt_states = np.vstack(protagonist_memory[1:, 0]) 
    protagonist_actions = np.vstack(protagonist_memory[:, 1]) 
    #print(protagonist_states.shape, protagonist_actions.shape)
    protagonist_states = torch.Tensor(protagonist_states)
    #protagonist_nxt_states = torch.Tensor(protagonist_nxt_states)
    protagonist_actions = torch.Tensor(protagonist_actions)
    protagonist_log_probs = torch.Tensor(log_prob_density(protagonist_actions, *protagonist_actor(protagonist_states))).detach()
 

    antagonist_memory = np.array(antagonist_memory) 
    antagonist_states = np.vstack(antagonist_memory[:, 0]) 
    antagonist_states = running_state(antagonist_states, update = False)
    #antagonist_nxt_states = np.vstack(antagonist_memory[1:, 0]) 
    antagonist_actions = np.vstack(antagonist_memory[:, 1]) 

    antagonist_states = torch.Tensor(antagonist_states)
    #antagonist_nxt_states = torch.Tensor(antagonist_nxt_states)
    antagonist_actions = torch.Tensor(antagonist_actions)
    antagonist_log_probs = torch.Tensor(log_prob_density(antagonist_actions, *antagonist_actor(antagonist_states))).detach()
    
    
    demonstration_states = np.vstack(demonstrations[:, 0])
    demonstration_states = running_state(demonstration_states, update = False) 
    
    demonstration_actions = np.vstack(demonstrations[:, 1])
    
    demonstration_states = torch.Tensor(demonstration_states)
    demonstration_actions = torch.Tensor(demonstration_actions)
    
 

    likelihood_criterion = torch.nn.BCELoss()
    constraint_criterion = torch.nn.MSELoss()
    for _ in range(args.reward_function_update_num):
        #protagonist_learner = log_prob_density(protagonist_actions, *(q_value(protagonist_states)))
        protagonist_learner = reward_function(torch.cat((protagonist_states, protagonist_actions), dim = 1))
        protagonist_antagonist_log_probs = torch.Tensor(log_prob_density(protagonist_actions, *antagonist_actor(protagonist_states))) 
        #antagonist_learner = log_prob_density(antagonist_actions, *(q_value(antagonist_states)))
        antagonist_learner = reward_function(torch.cat((antagonist_states, antagonist_actions), dim = 1))
        antagonist_protagonist_log_probs = torch.Tensor(log_prob_density(antagonist_actions, *protagonist_actor(antagonist_states))) 
        
        expert_learner = reward_function(torch.cat((demonstration_states, demonstration_actions), dim = 1))
        #expert_mu, expert_std = q_value(demonstration_states) 
        #expert_learner = log_prob_density(demonstration_actions, expert_mu, expert_std)
        protagonist_expert_learner = torch.Tensor(log_prob_density(demonstration_actions, *protagonist_actor(demonstration_states))).exp()        
        antagonist_expert_learner = torch.Tensor(log_prob_density(demonstration_actions, *antagonist_actor(demonstration_states))).exp()

        #expert_q_value = torch.log(expert_q_value_norm * expert_prob)
        #print(expert_q_value)
        #expert_nxt_qvalue = q_value(demonstration_nxt_states, demonstration_nxt_actions)[demonstration_nxt_actions]
     
        #likelihood_loss =  - (1. - expert_learner + 1e-20).log().mean() - (antagonist_learner + 1e-20).log().mean()
        likelihood_loss = likelihood_criterion(expert_learner, torch.zeros(demonstration_states.shape[0], 1)) + \
            likelihood_criterion(antagonist_learner, torch.ones(antagonist_states.shape[0], 1)) #+ \
                #+ likelihood_criterion(protagonist_learner, torch.ones(protagonist_states.shape[0], 1))
            #likelihood_criterion(expert_learner, torch.zeros(demonstration_states.shape[0], 1)) + likelihood_criterion(protagonist_learner, torch.ones(protagonist_states.shape[0], 1))
        #                    likelihood_criterion(expert_learner / (expert_learner + antagonist_expert_learner), torch.ones(demonstration_states.shape[0], 1)) #+ \
                                #likelihood_criterion(protagonist_learner / (protagonist_learner + protagonist_log_probs.exp()), torch.zeros(protagonist_states.shape[0], 1)) + \
                                    #likelihood_criterion(antagonist_learner / (antagonist_learner + antagonist_log_probs.exp()), torch.zeros(antagonist_states.shape[0], 1))

        constraint_loss = 0# constraint_criterion(expert_learner[:-1].log(), (expert_q_value[:-1] - args.gamma * expert_q_value[1:]).exp())  
        entropy_loss = 0 #expert_std.mean()
        #print(likelihood_loss, constraint_loss, entropy_loss)
        irl_loss = likelihood_loss  + args.constraint_coef * constraint_loss + args.entropy_coef * entropy_loss
        
                #pair_loss = likelihood_criterion(protagonist_learner / (protagonist_learner + protagonist_antagonist_learner), torch.zeros(protagonist_states.shape[0], 1)) + \
        #                likelihood_criterion(antagonist_learner / (antagonist_learner + antagonist_protagonist_learner), torch.ones(antagonist_states.shape[0], 1))
        pair_r1 = - ((protagonist_antagonist_log_probs.exp() / protagonist_learner - protagonist_antagonist_log_probs.exp()).log())
        pair_ratio1 = torch.exp(protagonist_antagonist_log_probs - protagonist_log_probs).detach()
        pair_loss1 = (pair_r1 * pair_ratio1)
        pair_loss1 = pair_loss1[torch.isfinite(pair_loss1)].mean() 
        #pair_ids1 = (pair_ratio1 <=  1. + args.clip_param).float() * (pair_ratio1 >=  1. - args.clip_param).float()
        #pair_clipped_ratio1 = pair_ratio1 * pair_ids1
        #pair_loss1 = (pair_r1 * pair_clipped_ratio1)
        #pair_loss1 = pair_loss1[torch.isfinite(pair_loss1)].sum() / pair_ids1[torch.isfinite(pair_loss1)].sum()
        
         
        pair_kl1 = torch.nn.functional.mse_loss(protagonist_actor(protagonist_states)[0], antagonist_actor(protagonist_states)[0]).detach().item()
        #pair_kl1 = torch.sqrt(protagonist_actor(protagonist_states)[0] - antagonist_actor(protagonist_states)[0])
        #pair_kl1 = pair_kl1[torch.isfinite(pair_kl1)].max().detach().item()

        pair_loss1 =  pair_loss1 + pair_kl1 * 4 * args.gamma / (1 - args.gamma) * torch.abs(pair_r1.flatten()).max() 
        pair_loss1 = pair_loss1 - pair_r1[torch.isfinite(pair_r1)].mean()
        
        
        pair_r2 = ((antagonist_log_probs.exp() / antagonist_learner - antagonist_log_probs.exp()).log())
        pair_ratio2 = (torch.exp(antagonist_protagonist_log_probs - antagonist_log_probs)).detach()
        pair_loss2 = (pair_r2 * pair_ratio2)
        pair_loss2 = pair_loss2[torch.isfinite(pair_loss2)].mean() 
        #pair_ids2 = (pair_ratio2 <=  1. + args.clip_param).float() * (pair_ratio2 >=  1. - args.clip_param).float()
        #pair_clipped_ratio2 = pair_ratio2 * pair_ids2
        #pair_loss2 = (pair_r2 * pair_clipped_ratio2)
        #pair_loss2 = pair_loss2[torch.isfinite(pair_loss2)].sum() / pair_ids2[torch.isfinite(pair_loss2)].sum()
        
        pair_kl2 = torch.nn.functional.mse_loss(antagonist_actor(antagonist_states)[0], protagonist_actor(antagonist_states)[0]).detach().item()
        #pair_kl2 = torch.sqrt(antagonist_actor(antagonist_states)[0] - protagonist_actor(antagonist_states)[0])
        #pair_kl2 = pair_kl2[torch.isfinite(pair_kl2)].max().detach().item()

        pair_loss2 = pair_loss2 - pair_kl2 * 4 * args.gamma / (1 - args.gamma) * torch.abs(pair_r2.flatten()).max() 
        pair_loss2 = pair_loss2 - pair_r2[torch.isfinite(pair_r2)].mean() 

        
        pair_ratio3 = (torch.exp(pair_r2 - antagonist_log_probs.detach()))
        pair_ids3 = (pair_ratio3 <=  1. + args.clip_param).float() * (pair_ratio3 >=  1. - args.clip_param).float()
        pair_clipped_ratio3 = torch.clamp(pair_ratio3, 1 - args.clip_param, 1 + args.clip_param)# * pair_ids3.detach()
        pair_loss3 = - torch.min(pair_r2 * pair_ratio3, pair_r2 * pair_clipped_ratio3).mean()
        #pair_loss3 = pair_loss3[torch.isfinite(pair_loss3)].sum() / pair_ids3[torch.isfinite(pair_loss3)].sum()
        pair_loss3 = pair_loss3 - pair_r1[torch.isfinite(pair_r1)].mean()
        

        pair_ratio4 = (torch.exp(-pair_r1 - protagonist_log_probs.detach()))
        pair_ids3 = (pair_ratio3 <=  1. + args.clip_param).float() * (pair_ratio3 >=  1. - args.clip_param).float()
        pair_clipped_ratio4 = torch.clamp(pair_ratio4, 1 - args.clip_param, 1 + args.clip_param)# * pair_ids3.detach()
        pair_loss4 = -torch.min(-pair_r1 * pair_ratio4, -pair_r1 * pair_clipped_ratio4).mean()
        #pair_loss3 = pair_loss3[torch.isfinite(pair_loss3)].sum() / pair_ids3[torch.isfinite(pair_loss3)].sum()
        pair_loss4 = pair_loss4 - pair_r1[torch.isfinite(pair_r1)].mean()
        """
        pair_loss = pair_loss1 + pair_loss2
        
        pair_loss0 = 0
        
         
            protagonist_expert_learner, _, _ = reward_function(torch.cat((demonstration_states, protagonist_expert_actions[i]), dim = 1))
            protagonist_expert_r = (antagonist_protagonist_expert_log_probs[i].exp() / protagonist_expert_learner - antagonist_protagonist_expert_log_probs[i].exp()).log()

            antagonist_expert_learner, _, _ = reward_function(torch.cat((demonstration_states, antagonist_expert_actions[i]), dim = 1))
            antagonist_expert_r = (antagonist_expert_log_probs[i].exp() / antagonist_expert_learner - antagonist_expert_log_probs[i].exp()).log()
            
            pair_loss0_i = (protagonist_expert_log_probs[i].exp() * protagonist_expert_r - antagonist_expert_log_probs[i].exp() * antagonist_expert_r)
            pair_loss0_i = pair_loss0_i[torch.isfinite(pair_loss0_i)]
            pair_loss0 += pair_loss0_i.mean()
        pair_loss0 /= 20
        """

        #r = (antagonist_expert_learner /expert_learner - antagonist_expert_learner).log()
        #ratio = ((protagonist_expert_learner - antagonist_expert_learner).detach()) 
        #pair_loss0 =  (r * ratio)
        
        #ratio = (protagonist_expert_learner - antagonist_expert_learner).detach()  / r.exp().detach()
        #pair_loss0 =  (r * ratio) 
        #pair_loss0 = pair_loss0[pair_loss0 < 0]
        #pair_loss0 = pair_loss0[torch.isfinite(pair_loss0)].log().mean().exp()
        #print(pair_loss1, pair_loss2, pair_loss0)
        
        pair_loss = pair_loss1 + pair_loss2 + (pair_loss4 if torch.isfinite(pair_loss4).all() else 0.)  + (pair_loss3 if torch.isfinite(pair_loss3).all() else 0.) 
        
        
        
        
        #print(pair_loss1, pair_loss2, pair_loss3)
        #print(r.mean(), ratio.mean())
        #pair_loss = (protagonist_learner.log() - (1. - protagonist_learner).log()).mean() - (antagonist_learner.log() - (1. - antagonist_learner).log()).mean()
        #pair_loss = protagonist_learner.log().mean() - antagonist_learner.log().mean()
        
        print(likelihood_loss, pair_loss)
        pair_irl_loss = irl_loss * args.irl_coef + args.pair_coef * pair_loss
        reward_function_optim.zero_grad()
        #q_value_optim.zero_grad()
        pair_irl_loss.backward()
        #clip_grad_norm(reward_function.parameters(), 0.99, norm_type = 'inf')
        clip_grad_value(reward_function.parameters(), 1)
        
        reward_function_optim.step()
        #q_value_optim.step()

    #print("Finished updating")
    #expert_acc = ((log_prob_density(demonstration_actions, *(q_value(demonstration_states))).exp()).float()).mean()
    #protagonist_learner_acc = ((log_prob_density(protagonist_actions, *(q_value(protagonist_states))).exp()).float()).mean()
    #antagonist_learner_acc = ((log_prob_density(antagonist_actions, *(q_value(antagonist_states))).exp()).float()).mean()

    expert_acc = ((reward_function(torch.Tensor(torch.cat([demonstration_states, demonstration_actions], dim=1)))[0]).float()).mean()
    protagonist_learner_acc = ((reward_function(torch.cat([protagonist_states, protagonist_actions], dim=1))[0]).float()).mean()
    antagonist_learner_acc = ((reward_function(torch.cat([antagonist_states, antagonist_actions], dim=1))[0]).float()).mean()
    
    #expert_acc = (reward_function(demonstration_states, demonstration_actions)[0]).float().mean()
    #protagonist_learner_acc = ((reward_function(protagonist_states, protagonist_actions))[0]).float().mean()
    #antagonist_learner_acc = (reward_function(antagonist_states, antagonist_actions))[0].float().mean()


    return expert_acc, protagonist_learner_acc, antagonist_learner_acc 




def train_actor_critic(running_state, actor, critic, memory, actor_optim, critic_optim, args):
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0]) 
    states = running_state(states, update = False) 
    actions = list(memory[:, 1]) 
    rewards = list(memory[:, 2]) 
    masks = list(memory[:, 3]) 

    old_values = critic(torch.Tensor(states))
    returns, advants = get_gae(rewards, masks, old_values, args)
    
    mu, std = actor(torch.Tensor(states))
    old_policy = log_prob_density(torch.Tensor(actions), mu, std)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)

        for i in range(n // args.batch_size): 
            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            
            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()
            
            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param, 
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            critic_optim.zero_grad()
            #loss.backward(retain_graph=True) 
            #critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()
            critic_optim.step()


def train_protagonist_actor_critic(running_state, actor, critic, antagonist_actor, antagonist_critic, memory, antagonist_memory, actor_optim, critic_optim, args):
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0]) 
    states = running_state(states, update = False) 
    actions = np.vstack(memory[:, 1]) 
    rewards = np.vstack(memory[:, 2]) 
    masks = np.vstack(memory[:, 3]) 

    old_values = critic(torch.Tensor(states))
    #antagonist_values = antagonist_critic(torch.Tensor(states)).detach()

    #returns, advants = get_gae(rewards, masks, old_values, args)
    #antagonist_returns, antagonist_advants = get_gae(rewards, masks, antagonist_values, args)
 
    old_policy = log_prob_density(torch.Tensor(actions), *actor(torch.Tensor(states)))
    #antagonist_policy = log_prob_density(torch.Tensor(actions), *antagonist_actor(torch.Tensor(states)))
    #ratio = torch.exp(antagonist_policy.detach() - old_policy.detach())
    #ratio = torch.ones(rewards.shape)
    returns, advants = get_gae(rewards, masks, old_values, args)

     
   
    antagonist_memory = np.array(antagonist_memory) 
    antagonist_states = np.vstack(antagonist_memory[:, 0]) 
    antagonist_states = running_state(antagonist_states, update = False)  
    antagonist_actions = np.vstack(antagonist_memory[:, 1]) 
    antagonist_rewards = np.vstack(antagonist_memory[:, 2]) 
    antagonist_masks = np.vstack(antagonist_memory[:, 3]) 

    antagonist_old_values = antagonist_critic(torch.Tensor(antagonist_states)).detach()
    #antagonist_values = antagonist_critic(torch.Tensor(states)).detach()

    #returns, advants = get_gae(rewards, masks, old_values, args)
    #antagonist_returns, antagonist_advants = get_gae(rewards, masks, antagonist_values, args)
 
    antagonist_old_policy = log_prob_density(torch.Tensor(antagonist_actions), *antagonist_actor(torch.Tensor(antagonist_states)))
    #policy = log_prob_density(torch.Tensor(antagonist_actions), *actor(torch.Tensor(antagonist_states)))
    #antagonist_ratio = torch.exp(antagonist_old_policy.detach() - policy.detach()) 
    #_, antagonist_advants = get_cross_gae(antagonist_rewards, antagonist_ratio, antagonist_masks, antagonist_old_values, args)
    
    _, antagonist_advants = get_gae(antagonist_rewards, antagonist_masks, antagonist_old_values, args)
   

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    
    antagonist_n = len(antagonist_states)
    antagonist_arr = np.arange(antagonist_n)
  
    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)
         
        np.random.shuffle(antagonist_arr)
       

        for i in range(n // args.batch_size): 
            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            
            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            #antagonist_advants_samples = antagonist_advants.unsqueeze(1)[batch_index]

            oldvalue_samples = old_values[batch_index].detach()
            
            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param, 
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values.flatten(), returns_samples.flatten())
            critic_loss2 = criterion(values.flatten(), returns_samples.flatten())
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()
            

            #loss, ratio, entropy = surrogate_antagonist_loss(actor, advants_samples, antagonist_actor, antagonist_advants_samples, inputs,
            #                             old_policy.detach(), antagonist_policy.detach(), actions_samples,
            #                             batch_index)
            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            i = int(i * antagonist_n / n)
            antagonist_batch_index = antagonist_arr[args.batch_size * i : args.batch_size * (i + 1)]
            antagonist_batch_index = torch.LongTensor(antagonist_batch_index )
            antagonist_inputs = torch.Tensor(antagonist_states)[antagonist_batch_index]
            antagonist_actions_samples = torch.Tensor(antagonist_actions)[antagonist_batch_index]
            #actor_antagonist_action_log_probs = torch.Tensor(log_prob_density(antagonist_actions_samples, *actor(antagonist_inputs))) 
            antagonist_advants_samples = antagonist_advants.unsqueeze(1)[antagonist_batch_index]

            antagonist_loss, antagonist_ratio, antagonist_entropy = surrogate_loss(actor, antagonist_advants_samples, antagonist_inputs,
                                         antagonist_old_policy.detach(),antagonist_actions_samples,
                                         antagonist_batch_index)

            antagonist_clipped_ratio = torch.clamp(antagonist_ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            antagonist_clipped_loss = antagonist_clipped_ratio * antagonist_advants_samples
            antagonist_actor_loss = -torch.min(antagonist_loss, antagonist_clipped_loss).mean()
            kl_loss = torch.nn.functional.mse_loss(actor(antagonist_inputs)[0], antagonist_actor(antagonist_inputs)[0])

            actor_loss = actor_loss + args.ppo_coef * (0. if antagonist_advants_samples.numel() == 0 else - antagonist_loss.mean() + (kl_loss * 4 * args.gamma / (1 - args.gamma) * torch.abs(antagonist_advants_samples.flatten().detach()).max().item()))
            if antagonist_advants_samples.numel() == 0:
                print(f"antagonist_advants_samples: {antagonist_advants_samples.shape} numel() == 0")
            entropy = entropy + antagonist_entropy
            #kl_loss = actor_antagonist_action_log_probs.mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy #- 0.01 * kl_loss
            #print("protagonist actor loss:", actor_loss, critic_loss, entropy, kl_loss)
            actor_optim.zero_grad()
            critic_optim.zero_grad()

            loss.backward(retain_graph=True) 
            critic_optim.step()

            #loss.backward()
            actor_optim.step()



def train_antagonist_actor_critic(running_state, actor, critic, protagonist_actor, protagonist_critic, memory, protagonist_memory, actor_optim, critic_optim, args):
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0]) 
    states = running_state(states, update = False)
    actions = np.vstack(memory[:, 1]) 
    rewards = np.vstack(memory[:, 2]) 
    masks = np.vstack(memory[:, 3]) 

    old_values = critic(torch.Tensor(states))
    #antagonist_values = antagonist_critic(torch.Tensor(states)).detach()

    #returns, advants = get_gae(rewards, masks, old_values, args)
    #antagonist_returns, antagonist_advants = get_gae(rewards, masks, antagonist_values, args)
 
    old_policy = log_prob_density(torch.Tensor(actions), *actor(torch.Tensor(states)))
 
    returns, advants = get_cross_gae(rewards, 0.* rewards, masks, old_values, args)

     
   
    protagonist_memory = np.array(protagonist_memory) 
    protagonist_states = np.vstack(protagonist_memory[:, 0]) 
    protagonist_states = running_state(protagonist_states, update = False) 
    protagonist_actions = np.vstack(protagonist_memory[:, 1]) 
    protagonist_rewards = np.vstack(protagonist_memory[:, 2]) 
    protagonist_masks = np.vstack(protagonist_memory[:, 3]) 

    protagonist_old_values = critic(torch.Tensor(protagonist_states))
    #protagonist_values = protagonist_critic(torch.Tensor(states)).detach()

    #returns, advants = get_gae(rewards, masks, old_values, args)
    #protagonist_returns, protagonist_advants = get_gae(rewards, masks, protagonist_values, args)
 
    protagonist_old_policy = log_prob_density(torch.Tensor(protagonist_actions), *protagonist_actor(torch.Tensor(protagonist_states)))
    
    _, protagonist_advants = get_cross_gae(protagonist_rewards, 0.* protagonist_rewards, protagonist_masks, protagonist_old_values, args)
 
    
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)
    
    protagonist_n = len(protagonist_states)
    protagonist_arr = np.arange(protagonist_n)
  
    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)
         
        np.random.shuffle(protagonist_arr)
       

        for i in range(n // args.batch_size): 
            batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            
            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            #protagonist_advants_samples = protagonist_advants.unsqueeze(1)[batch_index]

            oldvalue_samples = old_values[batch_index].detach()
            
            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param, 
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values.flatten(), returns_samples.flatten())
            critic_loss2 = criterion(values.flatten(), returns_samples.flatten())
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()
            

            #loss, ratio, entropy = surrogate_protagonist_loss(actor, advants_samples, protagonist_actor, protagonist_advants_samples, inputs,
            #                             old_policy.detach(), protagonist_policy.detach(), actions_samples,
            #                             batch_index)
            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)
            

            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()
         
            protagonist_batch_index = protagonist_arr[args.batch_size * i : args.batch_size * (i + 1)]
            protagonist_batch_index = torch.LongTensor(protagonist_batch_index)
            
            protagonist_inputs = torch.Tensor(protagonist_states)[protagonist_batch_index]
            protagonist_actions_samples = torch.Tensor(protagonist_actions)[protagonist_batch_index]
            protagonist_advants_samples = protagonist_advants.unsqueeze(1)[protagonist_batch_index]
            #protagonist_advants_samples = protagonist_advants.unsqueeze(1)[batch_index]

            protagonist_loss, protagonist_ratio, protagonist_entropy = surrogate_loss(actor, protagonist_advants_samples, protagonist_inputs,
                                         protagonist_old_policy.detach(), protagonist_actions_samples,
                                         protagonist_batch_index)
            
            protagonist_clipped_ratio = torch.clamp(protagonist_ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            protagonist_clipped_loss = protagonist_clipped_ratio * protagonist_advants_samples
            actor_loss = actor_loss + torch.max(protagonist_loss, protagonist_clipped_loss).mean()
            entropy = entropy + protagonist_entropy
            
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy 

            actor_optim.zero_grad()
            critic_optim.zero_grad()

            loss.backward(retain_graph=True) 
            critic_optim.step()

            #loss.backward()
            actor_optim.step()

def get_gae(rewards, masks, values, args):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
    
    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = rewards[t] + (args.gamma * previous_value * masks[t]) - \
                                        values.data[t]
        previous_value = values.data[t]
        
        running_advants = running_delta + (args.gamma * args.lamda * \
                                            running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

 

def get_cross_gae(rewards, ratio, masks, values, args):
    rewards = torch.Tensor(rewards)
    ratio = torch.Tensor(ratio)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)
 
    running_returns = 0
    previous_value = 0
    running_advants = 0
    
    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = (1 - ratio[t]) * rewards[t] + (args.gamma * previous_value * masks[t]) - \
                                        values.data[t]
        previous_value = values.data[t]
        
        running_advants = running_delta + (args.gamma * args.lamda * \
                                            running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy

def surrogate_antagonist_loss(actor, advants, antagonist_actor, antagonist_advants, states, old_policy, antagonist_policy, actions, batch_index):
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]
    antagonist_policy = antagonist_policy[batch_index]
    
    ratio = torch.exp(new_policy - old_policy)

    surrogate_loss = ratio * (advants - antagonist_advants * torch.exp(antagonist_policy))
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy