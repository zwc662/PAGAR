import torch
import numpy as np
from utils.utils import get_entropy, log_prob_density

def train_discrim(running_state, discrim, memory, discrim_optim, demonstrations, args):
    memory = np.array(memory) 
    states = np.vstack(memory[:, 0])
    states = running_state(states, update = False) 
    actions = list(memory[:, 1]) 

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)

    
    demonstration_states = np.vstack(demonstrations[:, 0])
    demonstration_states = running_state(demonstration_states, update = False) 
    demonstration_actions = np.vstack(demonstrations[:, 1])
    
    demonstration_states = torch.Tensor(demonstration_states)
    demonstration_actions = torch.Tensor(demonstration_actions)
        
    criterion = torch.nn.BCELoss()

    for _ in range(args.discrim_update_num):
        learner = discrim(torch.cat([states, actions], dim=1))
        expert = discrim(torch.cat([demonstration_states, demonstration_actions], dim = 1))

        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                        criterion(expert, torch.zeros((demonstrations.shape[0], 1)))
                
        discrim_optim.zero_grad()
        discrim_loss.backward()
        discrim_optim.step()

    expert_acc = ((discrim(torch.cat([demonstration_states, demonstration_actions], dim = 1)) < 0.5).float() ).mean()
    learner_acc = ((discrim(torch.cat([states, actions], dim=1))  > 0.5 ).float()).mean()

    return expert_acc, learner_acc

def train_soft_actor_critic(running_state, actor, critic, memory, actor_optim, critic_optim, args):
    # Sample a batch from memory
    batch = np.random.choice(len(memory) - 1, 256)
    memory = np.array(memory) 
    state_batch = np.vstack(memory[batch, 0]) 
    next_state_batch = np.vstack(memory[batch + 1, 0])
    state_batch = running_state(state_batch, update = False) 
    next_state_batch = running_state(next_state_batch, update = False) 
    action_batch = list(memory[batch, 1]) 
    reward_batch = list(memory[batch, 2]) 
    mask_batch = list(memory[batch, 3]) 

    state_batch = torch.FloatTensor(state_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    action_batch = torch.FloatTensor(action_batch)
    reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
    mask_batch = torch.FloatTensor(mask_batch).unsqueeze(1)
    
    criterion = torch.nn.MSELoss()
    
    v = critic(state_batch)
    v_next = critic(next_state_batch)
    q_value = reward_batch + mask_batch * args.gamma * (v_next)
    rewards = q_value - v
    clipped_rewards = torch.clamp(rewards, 1 - args.clip_param, 1 - args.clip_param)
    critic_loss1 = criterion(clipped_rewards, reward_batch)
    critic_loss2 = criterion(rewards, reward_batch)
    critic_loss = torch.max(critic_loss1, critic_loss2).mean()

    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    action_mu, action_std = actor(torch.Tensor(state_batch))
    state_log_pi = log_prob_density(torch.Tensor(action_batch), action_mu, action_std)
    next_action_mu, next_action_std = actor(torch.Tensor(next_state_batch))
     
    v_next = critic(next_state_batch)
    q_value = reward_batch + mask_batch * args.gamma * (v_next)

    actor_loss = ((0.9 * state_log_pi) - q_value).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    
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

def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy