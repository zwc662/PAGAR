import torch
import torch.nn as nn


def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, num_outputs)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std


class Critic(nn.Module):
    def __init__(self, num_inputs, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


class VDB(nn.Module):
    def __init__(self, num_inputs, args):
        super(VDB, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.z_size)
        self.fc3 = nn.Linear(args.hidden_size, args.z_size)
        self.fc4 = nn.Linear(args.z_size, args.hidden_size)
        self.fc5 = nn.Linear(args.hidden_size, 1)
        
        self.fc5.weight.data.mul_(0.1)
        self.fc5.bias.data.mul_(0.0)

    def encoder(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def discriminator(self, z):
        h = torch.tanh(self.fc4(z))
        return torch.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        prob = self.discriminator(z)
        return prob, mu, logvar

"""
class VDB(nn.Module):
    def __init__(self, num_inputs, args):
        super(VDB, self).__init__()
        self.layers1= []
        self.activs1 = []
        for i in range(args.num_layers):
            self.layers1.append(nn.Linear(args.hidden_size if i > 0 else num_inputs, args.hidden_size))
            self.activs1.append(nn.Tanh())
            self.add_module(f"fc1_{i}", self.layers1[-1])
            self.add_module(f"activ1_{i}", self.activs1[-1])
        self.fc2 = nn.Linear(args.hidden_size, args.z_size)
        self.fc3 = nn.Linear(args.hidden_size, args.z_size)
        self.layers4 = []
        self.activs4 = []
        for i in range(args.num_layers):
            self.layers4.append(nn.Linear(args.hidden_size if i > 0 else args.z_size, args.hidden_size))
            self.activs4.append(nn.Tanh())
            self.add_module(f"fc4_{i}", self.layers4[-1])
            self.add_module(f"activ4_{i}", self.activs4[-1])
        
        self.bn1 = nn.LayerNorm([1])
        self.fc5 = nn.Linear(args.hidden_size, 1)
        self.fc5.weight.data.mul_(0.1)
        self.fc5.bias.data.mul_(0.0)

        self.apply(init_normal)

    def encoder(self, x):
        for layer, activ in zip(self.layers1, self.activs1):
            x = activ(layer(x))
        #h = torch.tanh(self.fc1(x))
        return self.fc2(x), self.fc3(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def discriminator(self, x):
        for layer, activ in zip(self.layers4, self.activs4):
            x = activ(layer(x))
        h = self.fc5(x)
        return torch.sigmoid(h)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        prob = self.discriminator(z)
        return prob, mu, logvar

"""
 