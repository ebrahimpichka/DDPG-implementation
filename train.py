import gym
import numpy as np
from ddpg_agent import DDPGAgent
# from utils import plotLearning

env = gym.make('LunarLanderContinuous-v2')
print(env.action_space.sample())
agent = DDPGAgent(input_dims=[8], num_actions=2, tau=0.001, gamma=0.99, max_size=1000000, hidden1_dims=400,
                 hidden2_dims=300, batch_size=64, critic_lr=0.0003, actor_lr=0.0003)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(1000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        #env.render()
    score_history.append(score)

    #if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
