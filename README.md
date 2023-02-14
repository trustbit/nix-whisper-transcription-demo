# Python project scaffolding for Machine Learning

This is how we'd setup our own python projects (data ops, science and ML).

This project trains a neural network (PyTorch) on [Iris flower dataset](https://archive.ics.uci.edu/ml/datasets/iris). The model is then exposed via Flask as an API and Web UI.

This implementation sets a stage to demonstrate dependency management in data science and machine learning projects. 

The project is based on a standard Python project layout (including PEP621 for pyproject.toml). Based on our experience in operating ML models, we have extended this scaffolding with Nix and direnv. Just to make sure all data scientists have same setup as training and serving stages, without worrying about Docker or broken Python setups.  


## Usage

If you have our recommended configuration of [Nix data science VM](https://github.com/trustbit/nix-data-science-vm), then to use this setup:

1. Check out this repository and open the folder
2. `direnv allow` to permit direng operate
3. **Automatic**: direnv and Nix will ensure exact python version, virtual environment and binary dependencies.
4. `pip install --editable .` - install python project for editing. This will also create `serve` command.


Then you can execute `serve` and should get the output like:

```                                                         
2023-02-14 12:10:46.588 | INFO     | nix_python.serve:<module>:67 - Epoch 10, Loss: 0.39774712920188904
2023-02-14 12:10:46.590 | INFO     | nix_python.serve:<module>:67 - Epoch 20, Loss: 0.227406844496727
2023-02-14 12:10:46.592 | INFO     | nix_python.serve:<module>:67 - Epoch 30, Loss: 0.12843188643455505
2023-02-14 12:10:46.595 | INFO     | nix_python.serve:<module>:67 - Epoch 40, Loss: 0.08055561035871506
2023-02-14 12:10:46.597 | INFO     | nix_python.serve:<module>:67 - Epoch 50, Loss: 0.05586405470967293
2023-02-14 12:10:46.599 | INFO     | nix_python.serve:<module>:67 - Epoch 60, Loss: 0.04163944721221924
2023-02-14 12:10:46.600 | INFO     | nix_python.serve:<module>:67 - Epoch 70, Loss: 0.032664697617292404
2023-02-14 12:10:46.602 | INFO     | nix_python.serve:<module>:67 - Epoch 80, Loss: 0.026602912694215775
2023-02-14 12:10:46.604 | INFO     | nix_python.serve:<module>:67 - Epoch 90, Loss: 0.022280622273683548
2023-02-14 12:10:46.605 | INFO     | nix_python.serve:<module>:67 - Epoch 100, Loss: 0.019068924710154533
2023-02-14 12:10:46.611 | INFO     | werkzeug._internal:_log:224 - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
2023-02-14 12:10:46.611 | INFO     | werkzeug._internal:_log:224 - Press CTRL+C to quit
```


## Implementation Details

- Using direnv and nix flakes for automated dependency management
- Using pyproject.toml for project metadata ([PEP621](https://peps.python.org/pep-0621/) and [Project Metadata](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#declaring-project-metadata))
- Setuptools as the default build system
- Modules go into `src` folder
- Dockerfile that support install flakes from Github
