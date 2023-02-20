import whisper_timestamped as whisper
import pkgutil
from flask import Flask, request, jsonify, render_template
import os

from nix_python.logs import configure_logs

print('getcwd:      ', os.getcwd())
print('file:    ', __file__)

whisper_checkpoint_path = os.path.join(os.getcwd(), "src", "nix_python", "bashkir-whisper-medium-checkpoint.pt") 
print("file exist :", os.path.isfile(whisper_checkpoint_path))

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = whisper.load_model(whisper_checkpoint_path, device=device)

app = Flask(__name__)
handler = configure_logs()
app.logger.addHandler(handler)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)

        audio = whisper.load_audio(f.filename)
        result = whisper.transcribe(model, audio, language="ba")
        return jsonify(result)

    return jsonify({"status": "can't transcript"})


def serve():
    app.run(debug=True, port=5001)


if __name__ == '__main__':
    serve()
