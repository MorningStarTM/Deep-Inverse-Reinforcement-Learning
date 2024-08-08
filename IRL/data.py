import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



def load_and_read_transitions(filename):
    transition_storage = TransitionStorage()
    transition_storage.load_from_disk(filename)

    observations, actions, rewards, next_states = transition_storage.get_data()
    return observations, actions, rewards, next_states


class StateActionDataset(Dataset):

    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device) if y is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx] if self.y is not None else torch.empty((1, 1), dtype=torch.float32)
    

