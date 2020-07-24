import torch
import torch.nn as nn
import numpy
import gym


class PolicyNet(nn.Module):
    def __init__(self, inputs, actions, hidden_size):
        super(PolicyNet, self).__init__()
        self.num_actions = actions
        self.layer1 = nn.Linear(inputs, hidden_size)
        self.layer2 = nn.Linear(hidden_size, actions)
        self.prelu = nn.PReLU(hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.prelu(x)
        x = self.layer2(x)

        return nn.functional.softmax(x, dim=1)

    def action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        actions = self.forward(state)
        prob = numpy.random.choice(self.num_actions, p=numpy.squeeze(actions.detach().numpy()))
        log_prob = torch.log(actions.squeeze(0)[prob])
        return log_prob, prob

    def policy_gradients(self, rewards, log_prob, net):
        optim = torch.optim.Adam(net.parameters(), lr=0.01)
        Rewards = []
        for i in range(len(rewards)):
            G = p = 0
            for reward in rewards[i:]:
                G = G + 0.9 * p * reward
                p = p + 1
            Rewards.append(G)
        Rewards = torch.tensor(Rewards)

        discounted_reward = (Rewards - Rewards.mean()) / Rewards.std()

        gradients = []
        for log_prob, G in zip(log_prob, Rewards):
            gradients.append(-log_prob * G)

        optim.zero_grad()
        policy_gradient = torch.stack(gradients).sum()
        policy_gradient.backward()
        optim.step()
