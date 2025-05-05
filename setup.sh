#!/bin/bash

echo "Installing required packages..."

pip3 install --upgrade pip
pip3 install torch torchvision
pip3 install timm
pip3 install tqdm
pip3 install matplotlib
pip3 install pandas
pip3 install numpy
pip3 install psutil

echo "All packages installed successfully!"
echo "Starting training..."

python3 card_classifier.py 