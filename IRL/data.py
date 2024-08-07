import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TransitionStorage:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_states = []

    def add_data(self, observation, action, reward, next_state):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)

    def get_data(self):
        return (np.array(self.observations), 
                np.array(self.actions), 
                np.array(self.rewards), 
                np.array(self.next_states))

    def __len__(self):
        return len(self.observations)

    def save_to_disk(self, filename):
        np.savez(filename, 
                 observations=np.array(self.observations),
                 actions=np.array(self.actions),
                 rewards=np.array(self.rewards),
                 next_states=np.array(self.next_states))
        print(f"Data saved to {filename}")

    def load_from_disk(self, filename):
        data = np.load(filename)
        self.observations = data['observations'].tolist()
        self.actions = data['actions'].tolist()
        self.rewards = data['rewards'].tolist()
        self.next_states = data['next_states'].tolist()
        print(f"Data loaded from {filename}")





class TransitionDataset(Dataset):
    def __init__(self, storage):
        # Load data from TransitionStorage
        observations, actions, rewards, next_states = storage.get_data()
        
        # Combine observations and actions to form input data
        self.inputs = np.hstack([observations, actions])
        self.targets = rewards
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # Return a single sample
        input_data = torch.tensor(self.inputs[idx], dtype=torch.float32)
        target_data = torch.tensor(self.targets[idx], dtype=torch.float32)
        return input_data, target_data