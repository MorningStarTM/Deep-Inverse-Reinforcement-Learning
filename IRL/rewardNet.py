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
    def __init__(self, lr):
        super(RewardNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_accuracy = 0.0

        self.model = NeuralNetwork().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_loop(self, dataloader):
        """
        Optimization
        """
        size = len(dataloader.dataset)
        loss_history = []
        score_history = []
        
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self.model(X).squeeze(1)
            loss = self.criterion(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()
            
            if batch % 1000 == 0:            
                loss, current = loss.item(), batch * len(X)
                loss_history.append(loss)

                score = mean_squared_error(y.cpu().detach().numpy().tolist(), pred.cpu().detach().numpy().tolist(), squared=False)
                score_history.append(score)
            
                print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')
                
        return loss_history, score_history

    

    def validate_loop(self, dataloader):
        """
        Evaluates
        """
        num_batches = len(dataloader)
        validate_loss = 0

        final_predictions = []
        final_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                pred = self.model(X).squeeze(1)
                validate_loss += self.criterion(pred, y).item()

                final_predictions.extend(pred.cpu().detach().numpy().tolist())
                final_targets.extend(y.cpu().detach().numpy().tolist())

        validate_loss /= num_batches
        score = mean_squared_error(final_targets, final_predictions, squared=False)
        print(f"Validate Error: \n RMSE accuracy: {(100 * score):>0.1f}%, Avg loss: {validate_loss:>8f} \n")

    

    def predict(self, dataloader):
        """
        predict
        """
        final_predictions = []

        self.model.eval()
        with torch.no_grad():
            for X, _ in dataloader:
                pred = self.model(X).squeeze(1)

                final_predictions.extend(pred.cpu().detach().numpy().tolist())

        return final_predictions
    

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval() 