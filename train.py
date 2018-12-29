"""
Continuous Control : Reacher env in Unity ML-Agents Environments

Deep Deterministic Policy Gradients (DDPG) algorithm implementation.
cf. https://arxiv.org/abs/1509.02971

Second project for Udacity's Deep Reinforcement Learning (DRL) program.
Modified the code provided by Udacity DRL Team, 2018.
"""

import numpy as np
from collections import deque
import pickle
import torch
from agent import Agent
from unityagents import UnityEnvironment

"""
Params
======
    n_episodes (int): maximum number of training episodes
    eps_start (float): starting value of epsilon, for exploration action space
    eps_end (float): minimum value of epsilon
    eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    goal_score (float): average score to be required
    checkpoint (str): filename for saving weights
    scores_file (str): filename for saving scores
    env_file_name (str): your path to Reacher.app (Ver.1 or Ver.2)
    hidden_layers (list): size of hidden_layers
    drop_p (float): probability of an element to be zeroed
    use_bn (bool): use batch norm or not. default True
    use_reset (bool): weights initialization used in original paper. default True
    noise (str): choose noise type, gauss(Gaussian) or OU(Ornstein-Uhlenbeck process)
    mode (str): if you use a single agent version, set mode="single". default "multi"     
"""

n_episodes=200
eps_start=1.0
eps_end=0.01
eps_decay=0.9999
goal_score=30

checkpoint_actor="DDPG_actor.pth"
checkpoint_critic="DDPG_critic.pth"
scores_file="scores_ddpg.txt"

# version 1
#env_file_name="/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64"
# version 2
env_file_name="/data/Reacher_Linux_NoVis/Reacher.x86_64"

hidden_layers=[256,128]
drop_p=0
use_bn=True
use_reset=True
noise="OU"
mode="multi"

########  Environment Setting  ########
env = UnityEnvironment(file_name=env_file_name)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
#######################################


###########  Agent Setting  ###########
agent = Agent(state_size, action_size, seed=0, hidden_layers=hidden_layers, drop_p=drop_p, \
              use_bn=use_bn, use_reset=use_reset, noise=noise, mode=mode)
print('-------- Model structure --------')
print('info : reset weights: {}, noise type: {}'.format(use_reset, noise))
print('-------- Actor --------')
print(agent.actor_local)
print('-------- Critic -------')
print(agent.critic_local)
print('---------------------------------')   
#######################################

scores_agent = []                                       # list containing scores from each episode and agent
scores_window = deque(maxlen=100)                       # last 100 scores
eps = eps_start                                         # initialize epsilon
best_score = -np.inf
is_First = True

print('Interacting with env ...')   
for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment
    states = env_info.vector_observations                  # get the current state                             
    agent.reset()
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    while True:
        actions = agent.act(states, eps)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        agent.step(states, actions, rewards, next_states, dones)
        states = next_states                               # roll over states to next time step
        scores += rewards                                  # update the score (for each agent)
        if np.any(dones):                                  # exit loop if episode finished
            break
    score = np.mean(scores)
    scores_window.append(score)         # save most recent score
    scores_agent.append(score)          # save most recent score
    eps = max(eps_end, eps_decay*eps)   # decrease epsilon
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
    if i_episode % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    if np.mean(scores_window)>=goal_score and np.mean(scores_window)>=best_score:
        if is_First:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            is_First = False
        else:
            print('\nAverage Score: {:.2f}'.format(np.mean(scores_window)))
        torch.save(agent.actor_local.state_dict(), checkpoint_actor)
        torch.save(agent.critic_local.state_dict(), checkpoint_critic)
        best_score = np.mean(scores_window)
        

f = open(scores_file, 'wb')
pickle.dump(scores_agent, f)

# End