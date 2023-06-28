import os
import sys
import time
import argparse
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import toy_2d_2g as toy_env
from ppo_mmd import PPO_mmd
from utils import rollout
from plot_visitations import plot_visitations
from monitor import Monitor


def train():
    ####### initialize environment hyperparameters ######
    num_policies = 3                      # the number of agents

    has_continuous_action_space = False   # continuous action space; else discrete

    max_ep_len = 240                      # max timesteps in one episode
    max_training_episodes = int(2.2e3)  # break training loop if timeteps > max_training_timesteps

    # print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    # log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    # save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    ################ PPO hyperparameters ################

    update_freq = 8                       # update policy every n timesteps
    K_epochs = 65                         # update policy for K epochs in one PPO update

    eps_clip = 0.2                        # clip parameter for PPO
    gamma = 0.99                          # discount factor

    # lr_actor = 0.0001                   # learning rate for actor network
    # lr_critic = 0.0002                  # learning rate for critic network
    
    lr_actor = 0.000018                   # learning rate for actor network
    lr_critic = 0.00012                   # learning rate for critic network

    random_seed = 0                       # set random seed if required (0 = no random seed)
    use_mmd = False                       # set default value of the using_mmd

    ################ Create Environment ##################

    env = toy_env.Grid()
    env = Monitor(env, max_ep_len, update_freq)
    state_dim = 2
    action_dim = 4
    known_trajs = []
    
    for idx in range(num_policies):
        # initialize a PPO agent
        use_mmd = False if idx == 0 else True 
        
        # create a ppo agent
        ppo_agent = PPO_mmd(
            use_mmd, known_trajs, state_dim, action_dim, 
            lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
            has_continuous_action_space,)
        
        # initialize the writer
        env.init_results_writer(idx)
        
        # episode step for accumulate reward 
        epinfobuf = deque(maxlen=100)
        epinfos = deque(maxlen=update_freq)
        
        # Initialize parameters
        episode = 0
        ppo_agent.clear_trajs()
        
        # check learning time
        start_time = time.time()
        # record the step number
        step_num = 0
        # record the max_100_ep_rew
        results = []

        # training loop
        while episode <= max_training_episodes:
            # collect rollouts
            path = rollout(env, ppo_agent, max_ep_len)
            ppo_agent.append_traj(path)
            maybeepinfo = path['info'].get('episode')
            if maybeepinfo: 
                epinfos.append(maybeepinfo)
                epinfobuf.append(maybeepinfo)
                
            # Record the number of steps.   
            step_num += len(path['observations'])
            episode += 1

            # update PPO agent
            if episode % update_freq == 0:
                paths = ppo_agent.return_trajs()
                loss, mmd_distances = ppo_agent.update(episode)
                ppo_agent.clear_trajs()
                
                # write data
                env.write_data(epinfos, mmd_distances)
                
                # check time interval
                time_interval = round(time.time() - start_time, 2)
                # calc mean return
                mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]), 2)
                results.append(mean_100_ep_return)
                
                # Print log
                print('Used Step: ', step_num,
                    '| Loss: ', loss,
                    '| Mean ep 100 return: ', mean_100_ep_return,
                    '| Used Time:',time_interval)
                
        known_trajs = collect_demonstrations(env, ppo_agent, max_ep_len, known_trajs)
        
def collect_demonstrations(env, ppo_agent, max_ep_len, known_trajectories, test_trajectories=5): 
    # Rui: just run and render
    rec = len(known_trajectories)
    
    # run 100 iterations in total!
    while len(known_trajectories) - rec < test_trajectories:
        # produce trajectories
        path = rollout(env, ppo_agent, max_ep_len, test=True)

        known_trajectories.append(path)
        print("append")
        print("path: ", path['observations'])
    
    return known_trajectories
            

if __name__ == "__main__":
    train()