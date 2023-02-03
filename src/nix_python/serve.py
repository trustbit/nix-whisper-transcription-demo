import sys
from loguru import logger
from nix_python.logs import configure_logs
from flask import Flask, request, jsonify, render_template




app = Flask(__name__)

handler = configure_logs()
app.logger.addHandler(handler)
# Train the model
for epoch in range(100):

    if epoch % 10 == 9:
        logger.info('Epoch {epoch}, Loss: {loss}', epoch=epoch+1, loss=0)

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
