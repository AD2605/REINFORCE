import gym
from REINFORCE import PolicyNet
import numpy
import torch

env = gym.make('BipedalWalker-v3')
env.reset()
num_episodes = 20000
max_steps = 10000

policyNet = PolicyNet(inputs=env.observation_space.shape[0],
                      actions=4,
                      hidden_size=128)

numsteps = []
rewards = []
avg_numsteps = []

min_reward = -1000

for episode in range(num_episodes):
    state = env.reset()
    probs = []
    rewards = []

    for steps in range(max_steps):
        action, log_prob = policyNet.action(state)
        state, reward, finished, _ = env.step(action.squeeze(0).detach().numpy())
        env.render()
        probs.append(log_prob)
        rewards.append(reward)
        if finished:
            break


    if finished:
        policyNet.policy_gradients(rewards, probs, policyNet)
        numsteps.append(steps)
        avg_numsteps.append(numpy.mean(numsteps[-10:]))
        rewards.append(numpy.sum(rewards))

        print("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode,
                numpy.sum(rewards), numpy.round(numpy.mean(rewards[-10:]), decimals=3), steps))

    if numpy.sum(rewards) > min_reward:
        torch.save(policyNet.state_dict(), '/home/atharva/policyNet.pth')
        min_reward = numpy.sum(rewards)
