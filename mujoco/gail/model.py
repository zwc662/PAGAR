import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, num_outputs)
        #self.fc4 = nn.Linear(args.hidden_size, num_outputs)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

        #self.fc4.weight.data.mul_(0.1)
        #self.fc4.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        #logstd = self.fc4(x) 
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


class RobustDiscriminator(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, args.hidden_size)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        #prob = torch.sigmoid(self.fc3(x))
        dist = torch.distributions.Categorical(self.fc(x), self.num_outputs)
        return dist #prob