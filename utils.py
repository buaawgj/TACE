import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


def select_action(policy, o, epsilon=0, flag=False):
    o = torch.from_numpy(np.array(o)).float().unsqueeze(0)
    probs = policy(o)
    dist = Categorical(probs)
    action = dist.sample()
        
    if flag:
        print("action: ", action)
        print("action prob: ", probs[0, action[0]])
            
    return action.item(), dist.log_prob(action)

def rollout(env, agent, max_path_length=np.inf, test=False):
    observations = []
    actions = []
    rewards = []
    log_probs = []
    is_terminals = []
    goal = None
    
    env.reset()
    o = env.observe()
    
    for t in range(max_path_length): 
        a = agent.select_action(o)                
        next_o, reward, done, info = env.step(a)
        
        if not test:
            # saving reward and is_terminals
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
        elif test:
            agent.terminated = done
        
        observations.append(o)
        rewards.append(reward)
        actions.append(a)

        # log_probs.append(log_prob)
        
        if t == max_path_length - 1:
            print('state: ', o)
            print('action: ', a) 
    
        if done:
            goal = env.which_goal()
            print('goal: ', goal)
            print('state: ', o)
            print('action: ', a)
            break
        
        o = next_o
    
    path = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        is_terminals=is_terminals,
        log_probs=log_probs,
        goal=goal,
        info=info,
    )
    
    return path