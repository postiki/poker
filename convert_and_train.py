import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path

from data_processing.yolo_to_classification import convert_yolo_to_classification
from data_processing.utils import get_yolo_class_ids, validate_dataset
from models.card_classifier import CardClassifier, load_pretrained_model
from training.trainer import train_model, get_optimal_batch_size, get_optimal_workers
from config.transforms import get_train_transforms, get_val_transforms

def create_class_mapping(include_joker=False):
    suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
    ranks = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']
    class_names = ['No Card']
    for suit in suits:
        for rank in ranks:
            class_names.append(f'{rank} of {suit}')
    if include_joker:
        class_names.append('Joker')
    return class_names

def main():
    yolo_dir = Path('data/data-set-objectDetection')
    output_dir = Path('data_classification')
    
    yolo_class_ids = get_yolo_class_ids(yolo_dir)
    include_joker = 52 in yolo_class_ids
    class_names = create_class_mapping(include_joker=include_joker)
    
    if not output_dir.exists() or not any(output_dir.iterdir()):
        print('Converting YOLO dataset to classification format...')
        convert_yolo_to_classification(yolo_dir, output_dir, class_names, allow_joker=include_joker)
        validate_dataset(output_dir)
    else:
        print('Using existing converted dataset...')
    
    device = torch.device('mps')
    num_classes = len(class_names)
    
    train_dataset = ImageFolder(output_dir / 'train', transform=get_train_transforms())
    val_dataset = ImageFolder(output_dir / 'valid', transform=get_val_transforms())
    
    batch_size = get_optimal_batch_size()
    num_workers = get_optimal_workers()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if os.path.exists('best_card_classifier.pth'):
        print('Loading pretrained model...')
        model = load_pretrained_model('best_card_classifier.pth', num_classes, device=device)
    else:
        model = CardClassifier(num_classes=num_classes)

    model = model.to(device)

    try:
        model = torch.compile(model)
    except:
        print('Warning: torch.compile failed, using uncompiled model')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    print('Starting training...')
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        device=device,
        use_amp=True
    )

if __name__ == '__main__':
    main()