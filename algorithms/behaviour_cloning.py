import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


class BehaviorClonning:
    def __init__(self, input_dim, output_dim, learning_rate=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.best_accuracy = 0.0
        self.path = "artifacts\\model.pth"

    
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