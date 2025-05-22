from collections import deque
import random

class ReplayMemory():

    def __init__(self, maxLen):
        self.memory = deque([], maxlen=maxLen)
        
    def append(self, transisition):
        self.memory.append(transisition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)