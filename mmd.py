import numpy as np


def cartesian_product(a, b):
    """
    The vectors a and b are required to be two-dimensional numpy array, and the main 
    direction is the first dimension.
    """
    a = np.array(a)
    b = np.array(b)
    
    if a.ndim == 1:
        a = a.reshape((-1, 2))
    if b.ndim == 1:
        b = b.reshape((-1, 2))
        
    l_a = a.shape[0]
    l_b = b.shape[0]
    
    n_x_a = np.array([])
    for obj in a:
        if np.size(n_x_a) == 0:
            n_x_a = np.tile(np.array(obj).reshape((-1, 2)), (l_b, 1))
        else:
            n_x_obj = np.tile(np.array(obj).reshape((-1, 2)), (l_b, 1))
            n_x_a = np.concatenate([n_x_a, n_x_obj], axis=0)
            
    b_tile = np.tile(b, (l_a, 1))
    return n_x_a, b_tile

def exp_kernel(x, y, h=1.):
    """
    Adjust the parameter h during training.
    """
    if x.ndim == 1:
        x = x.reshape((2, -1))
    if y.ndim == 1:
        y = y.reshape((2, -1))
    return np.exp(-1. * np.sum((x - y)**2, 1) / h**2 / 2.0)

def MMD_calculate(path_a, path_b, h=2.5, gap=3):
    path_a = slice_path(path_a, gap=gap)
    path_b = slice_path(path_b, gap=gap)
    
    n_path_a_0, path_a_tile_0 = cartesian_product(path_a, path_a)
    n_path_a_1, path_b_tile_1 = cartesian_product(path_a, path_b)
    n_path_b_2, path_b_tile_2 = cartesian_product(path_b, path_b)
    
    term_0 = np.mean(exp_kernel(n_path_a_0, path_a_tile_0, h))
    term_1 = -2 * np.mean(exp_kernel(n_path_a_1, path_b_tile_1, h))
    term_2 = np.mean(exp_kernel(n_path_b_2, path_b_tile_2, h))

    # print("term_0: ", term_0)
    # print("term_1: ", term_1)
    # print("term_2: ", term_2)
    
    return term_0 + term_1 + term_2 


def make_goal_pair(goals):
    goal_pairs = []
    goals = list(goals)
    for idx, goal in enumerate(goals):
        for j in range(idx + 1, len(goals)):
            goal_pairs.append((goal, goals[j]))
            
    return goal_pairs

def slice_path(path, gap=40):
    if len(path) > 10 * gap:
        return path[:: gap]
    elif len(path) > 5 * gap:
        return path[:: gap // 2]
    elif len(path) > 3 * gap:
        return path[:: gap // 4]
    else:
        return path