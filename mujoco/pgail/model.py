import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
 

def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)
class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()
        self.layers = []
        self.activs = []
        for i in range(args.num_layers):
            self.layers.append(nn.Linear(args.hidden_size if i > 0 else num_inputs, args.hidden_size))
            self.activs.append(nn.Tanh())
            self.add_module(f"layer12_{i}", self.layers[-1])
            self.add_module(f"activ12_{i}", self.activs[-1])

        self.fc3 = nn.Linear(args.hidden_size, num_outputs)
        #self.fc4 = nn.Linear(args.hidden_size, num_outputs)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

        #self.fc4.weight.data.mul_(0.1)
        #self.fc4.bias.data.mul_(0.0)

    def forward(self, x):
        for layer, activ in zip(self.layers, self.activs):
            x = activ(layer(x))
        
        mu = self.fc3(x)
        #logstd = self.fc4(x) 
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std


class Critic(nn.Module):
    def __init__(self, num_inputs, args):
        super(Critic, self).__init__()
        
        self.layers = []
        self.activs = []
        for i in range(args.num_layers):
            self.layers.append(nn.Linear(args.hidden_size if i > 0 else num_inputs, args.hidden_size))
            self.activs.append(nn.Tanh())
            self.add_module(f"layer12_{i}", self.layers[-1])
            self.add_module(f"activ12_{i}", self.activs[-1])
        self.fc3 = nn.Linear(args.hidden_size, 1)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        for layer, activ in zip(self.layers, self.activs):
            x = activ(layer(x))
        
        v = self.fc3(x)
        return v

"""


class Discriminator(nn.Module):
    def __init__(self, num_inputs, args):
        super().__init__()
        self.layers = []
        self.activs = []
        for i in range(args.num_layers):
            self.layers.append(nn.Linear(args.hidden_size if i > 0 else num_inputs, args.hidden_size))
            self.activs.append(nn.Tanh())
            self.add_module(f"layer_{i}", self.layers[-1])
            self.add_module(f"activ_{i}", self.activs[-1])
        self.output = nn.Linear(args.hidden_size, 1)
 
    def forward(self, x):
        for layer, activ in zip(self.layers, self.activs):
            x = activ(layer(x))
        x = torch.sigmoid(self.output(x))
        return x

"""

class Discriminator(nn.Module):
    def __init__(self, num_inputs, args):
        super().__init__()
        
        self.layers = []
        self.activs = []
        for i in range(args.num_layers):
            self.layers.append(nn.Linear(args.hidden_size if i > 0 else num_inputs, args.hidden_size))
            self.activs.append(nn.Tanh())
            self.add_module(f"layer_{i}", self.layers[-1])
            self.add_module(f"activ_{i}", self.activs[-1])
        self.output = nn.Linear(args.hidden_size, 1)
 
    def forward(self, x):
        for layer, activ in zip(self.layers, self.activs):
            x = activ(layer(x))
        x = torch.sigmoid(self.output(x))
        return x

class Q_Value(nn.Module):
    def __init__(self, state_size, action_space, args):
        super(Q_Value, self).__init__()
        self.action_space = action_space
        self.fc1 = nn.Linear(state_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, action_space)
        #self.fc4 = nn.Linear(args.hidden_size, action_space)
        #self.fc5 = nn.Linear(args.hidden_size, 1)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        
        #self.fc4.weight.data.mul_(0.1)
        #self.fc4.bias.data.mul_(0.0)

        #self.fc5.weight.data.mul_(0.1)
        #self.fc5.bias.data.mul_(0.0)
         
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        mu = x[:, :self.action_space]
        logstd = x[:, self.action_space:]
        #normalization = torch.exp(x[:, -1:]) 
        #normalization = self.fc5(x) + 1.0e-6
        std = torch.exp(logstd)
        #dist = Normal(loc = mu, scale = std)
        return mu, std#, normalization


 