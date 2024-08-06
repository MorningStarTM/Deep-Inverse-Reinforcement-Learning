import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import matplotlib.pyplot as plt
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, dropout = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
    


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, lr):
        super().__init__()
        
        self.actor = MLP(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
        self.critic = MLP(input_dim=input_dim, hidden_dim=64)
        self.optimizer = torch.optim(self.parameters(), lr=lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

        
    def forward(self, state):
        
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        
        return action_pred, value_pred
    
    