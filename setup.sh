#!/bin/bash

echo "Starting setup process..."

# Update package lists
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv

# Install system dependencies
sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install --upgrade pip

# Install PyTorch with CUDA support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining requirements
pip install -r requirements.txt

# Generate SSH key if it doesn't exist
if [ ! -f ~/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
    echo "SSH key generated successfully"
    echo "Public key:"
    cat ~/.ssh/id_rsa.pub
fi

# Set proper permissions
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub

# Create necessary directories
mkdir -p logs
mkdir -p data

# Set up environment variables
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "FLASK_APP=app.py" > .env
    echo "FLASK_ENV=production" >> .env
fi

echo "Setup completed successfully!"
echo "To start the application, run: source venv/bin/activate && python app.py" 