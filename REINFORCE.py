import torch
import torch.nn as nn
import numpy
from torch.autograd import Variable
import gym


class PolicyNet(nn.Module):
    def __init__(self, inputs, actions, hidden_size):
        super(PolicyNet, self).__init__()
        self.num_actions = actions
        self.layer1 = nn.Linear(inputs, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 2*hidden_size)
        self.layer4 = nn.Linear(2*hidden_size, hidden_size)
        self.layer5 = nn.Linear(hidden_size, actions)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.relu(x)
        x = self.layer3(x)
        x = nn.functional.relu(x)
        x = self.layer4(x)
        x = nn.functional.relu(x)
        x = self.layer5(x)
        return x

    def action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        actions = self.forward(Variable(state))
        #prob = numpy.random.choice(self.num_actions, p=numpy.squeeze(actions.detach().numpy()))
        log_prob = torch.log(actions.squeeze(0))
        return actions, log_prob

    def policy_gradients(self, rewards, log_prob, net):
        optim = torch.optim.Adam(net.parameters(), lr=1e-3)
        Rewards = []
        for i in range(len(rewards)):
            G =0
            p=0
            for reward in rewards[i:]:
                G = G + 0.95 * p * reward
                p = p + 1
            Rewards.append(G)

        Rewards = torch.tensor(Rewards)

        discounted_reward = (Rewards - Rewards.mean()) / (Rewards.std()+1e+8)

        gradients = []
        for log_prob, G in zip(log_prob, discounted_reward):
            gradients.append(-log_prob * G)

        optim.zero_grad()
        policy_gradient = Variable(torch.stack(gradients).sum(), requires_grad=True)
        policy_gradient.backward()
        optim.step()
