import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # of the storage/buffer
        self.storage = [] # storage 
        self.position = 0 # where to add next item        
    
    def add(self, state, action, reward, next_state, terminated, truncated):
        experience = (state, action, reward, next_state, terminated, truncated)

        if len(self.storage) < self.capacity:
            self.storage.append(experience)
        else:
            self.storage[self.position % self.capacity] = experience 
        
        self.position +=1 
    
    def sample(self, batch_size):
        if len(self.storage) < batch_size:
            batch_size = len(self.storage)

        indices = np.random.choice(len(self.storage), batch_size, replace=False)
        return [self.storage[i] for i in indices]
    

    def __len__(self):
        return len(self.storage)
    
       