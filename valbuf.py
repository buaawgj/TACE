import numpy as np


class ValBuffer:
    """ The buffer of values with the fifo function. """
    def __init__(self, max_num):
        self.max_num = max_num

        # the {min, max} credit value assigned to a transition in the FIFOBuffer
        self.min_val = None
        self.max_val = None

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)
        
        self.values = np.zeros((int(self.max_num),), dtype=np.float32) # a list used to store values of trajectories

    def __len__(self):
        return self.filled_i
    
    @property
    def obtain_values(self):
        return self.values.item()

    def add_values(self, values):
        values = np.array(values, dtype=np.float32)

        nentries = len(values)
        # Roll the elements in the end to the start of the numpy array to overwrite them. 
        if self.curr_i + nentries > self.max_num:
            rollover = self.max_num - self.curr_i # num of indices to roll over
            self.values = np.roll(self.obs, rollover, axis=0)
            self.curr_i = 0
            self.filled_i = self.max_num

        self.values[self.curr_i : self.curr_i + nentries] = values

        self.curr_i += nentries
        if self.filled_i < self.max_num:
            self.filled_i += nentries
        if self.curr_i == self.max_num:
            self.curr_i = 0

        # update credit values
        if self.values.size != 0:
            self.min_val = self.values[:len(self)].min().item()
            self.max_val = self.values[:len(self)].max().item()

    def uniform_value(self, val):
        # Adjust the scale of values.
        if self.max_val and self.min_val:
            return (val - self.min_val) / (self.max_val - self.min_val)
        else: 
            print("min val: ", self.min_val)
            print("max val: ", self.max_val)
            print("The maximal or minimal values are None!")
            return val
        
    def normal_value(self, val):
        if len(self.values) > 100:
            mean = self.values.mean()
            std = self.values.std()
            
            normal_val = (val - mean) / (std + 1e-7) 
            return normal_val
        
        else:
            return val