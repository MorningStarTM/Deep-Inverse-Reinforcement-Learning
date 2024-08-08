import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error


class NeuralNetwork(nn.Module):
    """
    Simple neural network
    """

    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        logits = self.head(x)

        return logits
    

class RewardNet:
    def __init__(self, input_dim, lr):
        super(RewardNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_accuracy = 0.0

        self.seq = NeuralNetwork(input_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.seq.parameters(), lr=lr)

    def train_loop(dataloader, model, loss_fn, optimizer):
        """
        Optimization
        """
        size = len(dataloader.dataset)
        loss_history = []
        score_history = []
        
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X).squeeze(1)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            
            if batch % 1000 == 0:            
                loss, current = loss.item(), batch * len(X)
                loss_history.append(loss)

                score = mean_squared_error(y.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist(), squared=False)
                score_history.append(score)
            
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
                
        return loss_history, score_history

    

    def evaluate(self, X_test:torch.Tensor, y_test:torch.Tensor, batch_size=32):
        """
        Function for evaluate the model
        Args:
            X_test (tensor) data without label
            y_test (tensor) true label
            batch size (int)

        Return:
            Accuracy (float)
        """
        self.seq.eval()
        total_loss = 0
        with torch.no_grad():
            for i in range(0, X_test.size(0), batch_size):
                batch_X = X_test[i:i+batch_size]
                batch_y = y_test[i:i+batch_size]

                x = torch.tensor(batch_X, device=self.device, dtype=torch.float32)
                y = torch.tensor(batch_y, device=self.device, dtype=torch.float32)
                outputs = self.seq(batch_X)
                
                loss = self.criterion(outputs.squeeze(), y)  
                total_loss += loss.item() * y.size(0)  
        average_loss = total_loss / X_test.size(0)
        return average_loss
    

    def validate(self, dataloader):
        self.seq.eval()  # Set model to evaluation mode
        total_loss = 0
        with torch.no_grad():
            for state_action, reward_target in dataloader:
                state_action = state_action.to(self.device)
                reward_target = reward_target.to(self.device)
                output = self.seq(state_action)
                loss = nn.MSELoss()(output, reward_target)
                total_loss += loss.item()
        return total_loss / len(dataloader)
    

    def save_model(self, filepath):
        torch.save(self.seq.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.seq.load_state_dict(torch.load(filepath))
        self.seq.eval() 