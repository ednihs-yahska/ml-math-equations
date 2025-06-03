import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class LinearEquationDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.data = np.linspace(-10, 10, num_samples)
        self.labels = 2 * self.data + 3


    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.num_samples

class LinearEquationNN(nn.Module):
    def __init__(self):
        super(LinearEquationNN, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

def train_linear_model(model, dataset, num_epochs=1000, learning_rate=0.01):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for x, y in dataloader:
            x = x.view(-1, 1)  # Reshape for linear layer
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.view(-1, 1))
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print("Training complete.")

def evaluate_linear_model(model, dataset):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    with torch.no_grad():
        for x, y in dataloader:
            x = x.view(-1, 1)
            outputs = model(x)
            loss = nn.MSELoss()(outputs, y.view(-1, 1))
            print(f'Evaluation Loss: {loss.item():.4f}')
    return predictions.numpy()
