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
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state_action:torch.Tensor):
        reward = self.seq(state_action)
        return reward
    

    def learn(self, state_action, reward_target):
        self.train()  # Set model to training mode
        self.optimizer.zero_grad()
        output = self.forward(state_action)
        loss = nn.MSELoss()(output, reward_target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

    def evaluate(self, state_action, reward_target):
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = self.forward(state_action)
            loss = nn.MSELoss()(output, reward_target)
        return loss.item()