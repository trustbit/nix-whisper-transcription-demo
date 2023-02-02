import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


def main():
    # Load the Iris dataset
    data = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')

    # Define the features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Convert the labels to binary values
    y = (y == 'setosa').astype(int)

    # Convert numpy arrays to PyTorch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    # Define the neural network
    class BinaryClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(4, 8)
            self.layer2 = nn.Linear(8, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = self.sigmoid(self.layer2(x))
            return x

    # Initialize the model
    model = BinaryClassifier()

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Train the model
    for epoch in range(100):
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred.squeeze(), y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


if __name__ == "__main__":
    main()
