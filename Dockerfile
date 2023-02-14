# Use a base image with NixOs
FROM nixos/nix

# Create a working directory to install a code
RUN mkdir /app
WORKDIR /app

# Need to enable flakes 
RUN mkdir -p ~/.config/nix && echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

# Fetch data about last conservative updates for fixing bugs and security vulnerabilities 
RUN nix-channel --update

# Install demo app as flake from github
RUN nix build github:trustbit/nix-python/0.0.0

# Run the command to start demo app Flask API
CMD /app/result/bin/serve
