import torch
import torch.nn as nn
import numpy
from torch.autograd import Variable
import gym

def train(model, name):
    model.train()
    env = gym.make(name)
    env.reset()
    num_episodes = 20000
    max_steps = 10000
    optim = torch.optim.SGD(model.parameters(), lr=5)
    numsteps = []
    rewards = []
    avg_numsteps = []

    min_reward = -1000

    for episode in range(num_episodes):
        state = env.reset()
        probs = []
        rewards = []

        for steps in range(max_steps):
            action, log_prob = model.action(state)
            env.render()
            state, reward, finished, _ = env.step(action.squeeze(0).detach().numpy())
            env.render()
            probs.append(log_prob)
            rewards.append(reward)
            if finished:
                break

        if finished:
            Rewards = []
            for i in range(len(rewards)):
                G = 0
                p = 0
                for reward in rewards[i:]:
                    G = G + 0.9 * p * reward
                    p = p + 1
                Rewards.append(G)

            Rewards = torch.tensor(Rewards)

            discounted_reward = (Rewards - Rewards.mean()) / (Rewards.std() + 1e-9)

            gradients = []
            for log_prob, G in zip(log_prob, discounted_reward):
                gradients.append(-log_prob * G)

            optim.zero_grad()
            policy_gradient = Variable(torch.stack(gradients).sum(), requires_grad=True)
            policy_gradient.backward()
            optim.step()
            numsteps.append(steps)
            avg_numsteps.append(numpy.mean(numsteps[-10:]))
            rewards.append(numpy.sum(rewards))

            print("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode,
                                                                                           numpy.sum(rewards),
                                                                                           numpy.round(numpy.mean(
                                                                                               rewards[-10:]),
                                                                                                       decimals=3),
                                                                                           steps))

        if numpy.sum(rewards) > min_reward:
            torch.save(model.state_dict(), '/home/atharva/policyNet.pth')
            min_reward = numpy.sum(rewards)


def test(model, name):
    env = gym.make(name)
    model.eval()
    state = env.reset()
    with torch.no_grad():
        while True:
            action, log_prob = model(state)
            state, reward, finished, _ = env.step(action.squeeze(0).numpy())
            env.render()
            if finished:
                break


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

