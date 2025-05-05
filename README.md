# Playing Cards Classifier

This project implements a deep learning model to classify playing cards using PyTorch and ResNet50.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download and prepare the dataset:
```bash
python download_dataset.py
```

3. Train the model:
```bash
python card_classifier.py
```

## Features

- Uses ResNet50 as the base model
- Implements custom dataset loading for playing cards
- Includes data augmentation and normalization
- Saves the trained model for future use

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- PIL
- tqdm
- numpy
- matplotlib

## Dataset

The dataset contains images of playing cards from different angles and lighting conditions. The model is trained to classify these cards into their respective categories. 