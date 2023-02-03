# Use a base image with Python and Flask already installed
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the application code to the container
COPY . /app

# Install the required packages
RUN pip install  --no-cache-dir -r requirements.txt .

# Expose the port for the Flask API
EXPOSE 5000

# Run the command to start the Flask API
CMD ["serve"]
