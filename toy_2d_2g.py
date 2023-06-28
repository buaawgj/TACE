#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Maze environment with the deceptive reward
"""

import numpy as np


#Toy 2D Grid environment adapted from: https://github.com/dtak/hip-mdp-public/blob/master/grid_simulator/grid.py

# 目标区域的中心坐标
goals = [[26, 26],
         [15, 0],
         [0, 15]]

# 目标的奖励值
goal_reward = [100, 5, 8]

# 目标区域的半径d
goal_radii = [[2.5, 2.5],
              [2.5, 2.5,],
              [2.5, 2.5,],]

# 迷宫的边长
x_range = [0, 26]
y_range = [0, 26]

# 动作的数量：上下左右
num_actions = 4

# 格子的边长
step_size = 0.4


class Grid(object):
    """
    This is a 2D Grid environment for simple RL tasks.  
    """

    def __init__(
        self, start_state=[0, 0], step_size=step_size, goal_reward=goal_radii, 
        x_range=x_range, y_range=y_range, num_actions=num_actions, **kwargs
        ):
        """
        Initialize the environment: creating the box, setting a start and goal region
        Later -- might include other obstacles etc.
        """  
        # 动作的数量：上下左右
        self.num_actions = num_actions
        # 迷宫的边长
        self.x_range = x_range
        self.y_range = y_range
#        self.goal_radius = 0.5
#        self.goal = [1.0,0]
        self.spec = "gridworld"
        self.reset(start_state, step_size, goal_reward, **kwargs)  
        
    def reset(self, start_state=[0, 0], step_size=step_size, goal_reward=goal_reward,
              goal=goals, goal_radius=goal_radii, x_std=0, y_std=0, **kwargs
              ):
        """
        Reset Environment.
        """
        self.t = 0
        self.step_size = step_size
        self.start_state = start_state
        self.state = start_state
        self.goal = goal
        self.x_std = x_std
        self.y_std = y_std
        self.goal_reward = goal_reward
        self.goal_radius = goal_radius
        # else:
        #     self.goal_radius = [[4, 4],
        #                         [1, 1],]

    def print_goals(self):
        return self.goal
        
    def observe(self):
        """Return the agent's position."""
        return self.state
    
    def get_action_effect(self, action):
        """
        Set the effect direction of the action -- actual movement will 
        involve step size and possibly involve action error.
        """
        if action == 0:
            return [1, 0]
        elif action == 1:
            return [-1, 0]
        elif action == 2:
            return [0, 1]
        elif action == 3:
            return [0, -1]
        
    def get_next_state(self, state, action):
        """
        Take action from state, and return the next state.
        """
        action_effect = self.get_action_effect(action)
        new_x = state[0] + (self.step_size * action_effect[0]) + np.random.normal(0, self.x_std)
        new_y = state[1] + (self.step_size * action_effect[1]) + np.random.normal(0, self.y_std)
        
        next_state = [new_x, new_y]
        return next_state
    
    def _valid_crossing(self, state=None, next_state=None, action=None):
        """Determine if the agent has crossed the maze boundary."""
        if state is None:
            state = self.state
            action = self.action
        if next_state is None:
            next_state = self.get_next_state(state, action)
            
        #Check for moving out of box in x direction 
        if next_state[0] < np.min(self.x_range) or next_state[0] > np.max(self.x_range):
#            print "Going out of x bounds"
            return False
        elif next_state[1] < np.min(self.y_range) or next_state[1] > np.max(self.y_range):
#            print "Going out of y bounds"
            return False
        else:
            return True
        
    def _in_goal(self, state=None):
        """Determine if the agent has arrived the goal."""
        if state is None:
            state = self.state  
        each_goal = []
        for goal_i, radius_i in zip(self.goal, self.goal_radius):
            if (abs(np.array(state[0]) - np.array(goal_i[0])) <= radius_i[0]):
                if (abs(np.array(state[1]) - np.array(goal_i[1])) <= radius_i[1]):
                    print("Reached goal: ", goal_i)
                    each_goal.append(1)
            else:
                each_goal.append(0)
        # if np.sum(each_goal) >= 1:
        #     return True
        # else:
        #     return False
        return each_goal
        
    def which_goal(self, state=None):
        """Determine which the goal the agent has arrived at."""
        if state is None:
            state = self.state  
        each_goal = []
        for goal_i, radius_i in zip(self.goal, self.goal_radius):
            if (abs(np.array(state[0]) - np.array(goal_i[0])) <= radius_i[0]):
                if (abs(np.array(state[1]) - np.array(goal_i[1])) <= radius_i[1]):
                    return goal_i 
                
        return None        
                
    def calc_reward(self, state=None, action=None, next_state=None, **kw):
        """Calculate the reward of each step."""
        each_goal = self._in_goal(state=next_state)
        if state is None:
            state = self.state
            action = self.action
        if next_state is None:
            next_state = self.get_next_state(state, action)
        if self._valid_crossing(state=state, next_state=next_state, action=action) and \
            np.sum(each_goal):
            for idx, goal in enumerate(each_goal):
                if goal != 0:
                    return self.goal_reward[idx]
        elif self._valid_crossing(state=state, next_state=next_state, action=action) and \
            not np.sum(each_goal):
            return -0.5
        else: 
            return -1.
                   
    def step(self, action, **kwargs):
        """Perform actions in the maze."""
        info = {}
        self.t += 1
        self.action = action
        reward = self.calc_reward()
        if self._valid_crossing():
            self.state = self.get_next_state(self.state, action)
            
        goal = self.which_goal()
        info["goal_idx"] = goal
        return self.observe(), reward, np.sum(self._in_goal()), info