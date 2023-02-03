import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys
from loguru import logger
from .logs import configure_logs

from flask import Flask, request, jsonify, render_template




app = Flask(__name__)

handler = configure_logs()
app.logger.addHandler(handler)

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

    if epoch % 10 == 9:
        logger.info('Epoch {epoch}, Loss: {loss}', epoch=epoch+1, loss=loss.item())

# Initialize the Flask API

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input features from the request
    x = request.json['x']

    # Convert the input to a PyTorch tensor
    x = torch.tensor(x).float()

    # Use the model to make predictions
    with torch.no_grad():
        y_pred = model(x)

    # Convert the predictions to a binary class
    y_pred = (y_pred.squeeze() > 0.5).int().item()

    # Return the predictions as a JSON response
    return jsonify({'y_pred': y_pred})

def serve():
    app.run()

if __name__ == '__main__':
    serve()
