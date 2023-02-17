import whisper_timestamped as whisper
from flask import Flask, request, jsonify, render_template

from nix_python.logs import configure_logs

whisper_checkpoint_path = './bashkir-whisper-medium-checkpoint.pt'

model = whisper.load_model(whisper_checkpoint_path, device="cuda")

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
