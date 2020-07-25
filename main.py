from REINFORCE import PolicyNet, train

model = PolicyNet(24, 4, 256)
train(model, 'BipedalWalker-v3')
