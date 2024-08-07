import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RewardNet(nn.Module):
    def __init__(self, input_dim, lr):
        self.seq = nn.Sequential(
                            nn.Linear(input_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 1)
                        )
        
    
    def forward(self, state_action:torch.Tensor):
        x = self.seq(state_action)
        return x
    

    