import random
from collections import deque

class ReplayBuffer:
    """Experience Replay Buffer. Pops elements out when adding more than maximum capacity.
    
        Args:
            capacity (int): maximum capacity of buffer, will hold (s, a, r, s') transitions.
    """
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        """Add experience to memory."""
        self.memory.append([*args])
    
    def sample(self, batch_size):
        """Sample batch of experiences from memory with replacement."""
        return random.choices(self.memory, k=batch_size)
    
    def __len__(self):
        return len(self.memory)