import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class RewardNet:
    def __init__(self, input_dim, lr):
        super(RewardNet, self).__init__()
        self.seq = NeuralNetwork(input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Function for training 

        Args: 
            X_train (tensot) data without lable
            y_train (tensor) label
            epochs (int)
            batch_size (int)
        """
        self.model.train()
        for epoch in range(epochs):
            permutation = torch.randperm(X_train.size()[0])
            for i in range(0, X_train.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices].to(self.device), y_train[indices].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

            #  save the  model when loss is reduce
            train_accuracy = self.evaluate(X_train, y_train)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Training Accuracy: {train_accuracy:.4f}')
            if self.best_accuracy < train_accuracy:
                self.best_accuracy = train_accuracy
                self.save_model()
    

    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Function for evaluate the model
        Args:
            X_test (tensor) data without label
            y_test (tensor) true label
            batch size (int)

        Return:
            Accuracy (float)
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, X_test.size(0), batch_size):
                batch_X = X_test[i:i+batch_size].to(self.device)
                batch_y = y_test[i:i+batch_size].to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = correct / total
        return accuracy
    

    def validate(self, dataloader):
        self.eval()  # Set model to evaluation mode
        total_loss = 0
        with torch.no_grad():
            for state_action, reward_target in dataloader:
                state_action = state_action.to(self.device)
                reward_target = reward_target.to(self.device)
                output = self.forward(state_action)
                loss = nn.MSELoss()(output, reward_target)
                total_loss += loss.item()
        return total_loss / len(dataloader)
    

    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))
        self.eval() 