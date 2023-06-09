import os
import sys
import torch
import numpy as np
import torch.nn as nn

from mmd import MMD_calculate

EPS = 1e-6


def decide_trajectory(path, good_path_buffer):
    """
    Decide which good trajectory is closest to the current trajectory.
    Use the position coordinate information of the trajectory to calculate the MMD.
    """
    min_dis = 10000
    min_dis_path = None
    for good_path in good_path_buffer:
        distance = MMD_calculate(path, good_path)
        if distance < min_dis:
            min_dis = distance
            min_dis_path = good_path 
    return min_dis, min_dis_path 
    
def distance_to_buffer(current_trajectory, trajectory_buffer, h=3.):
    """
    Calculate the distances between the current trajectory and the trajectory buffer.
    """
    min_distance = 1000.
    min_trajectory = None
    idx = None
    
    for i, expert_trajectory in enumerate(trajectory_buffer):
        distance = MMD_calculate(current_trajectory, expert_trajectory, h=h)
        if distance < min_distance:
            min_distance = distance
            min_trajectory = expert_trajectory
            idx = i
             
    return min_distance, expert_trajectory, idx

def normalize_value(value):
    # Normalize the value of a trajectory.
    # Do not need to sum all values of a state-action pair.
    # Maybe we do not need this function.
    normalize_value = value / (values.sum() + EPS)
    return normalize_value

def adjust_value(values, min_val=None, max_val=None):
    # Adjust the scale of values.
    if not max_val and not min_val:
        return (values - min_val) / (max_val - min_val)
    else: 
        print("The maximal or minimal values are None!")
        raise
    
def obtain_state(ob, i):
    return ob[:i]

def extract_info(trajectory, key, loc):
    state_infos = []
    obs = trajectory[key]
    for ob in obs:
        state = obtain_state(ob, loc)
        state_infos.append(state)
    
    return state_infos

def compute_return(trajectory, gamma=1., key='rewards'):
    # Compute the cumulative return value of the trajectory. 
    rewards = trajectory[key]
    traj_return = 0.
    for i in range(len(rewards) - 1, -1, -1):
        traj_return = rewards[i] + gamma * traj_return 
            
    return traj_return