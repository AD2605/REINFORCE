import gym
from REINFORCE import PolicyNet
import numpy

env = gym.make('BipedalWalker-v3')
num_episodes = 10000
max_steps = 10000
policyNet = PolicyNet(env.observation_space.shape[0], env.action_space, 128)

numsteps = []
rewards = []
avg_numsteps = []

for episode in range(num_episodes):
    state = env.reset()
    probs = []
    rewards = []

    for steps in range(max_steps):
        env.render()
        action, log_prob = policyNet.action(state)
        new_state, reward, finished, _ = env.step(action)
        probs.append(log_prob)
        rewards.append(reward)

    if finished:
        policyNet.policy_gradients(reward, log_prob)
        numsteps.append(steps)
        avg_numsteps.append(numpy.mean(numsteps[-10:]))
        rewards.append(numpy.sum(rewards))

        if episode % 1 == 0:
            print("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, numpy.round(
                numpy.sum(rewards), decimals=3), numpy.round(numpy.mean(rewards[-10:]), decimals=3), steps))
        break