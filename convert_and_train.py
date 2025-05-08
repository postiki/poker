import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pathlib import Path
import argparse

from data_processing.yolo_to_classification import convert_yolo_to_classification
from data_processing.utils import get_yolo_class_ids, validate_dataset
from models.card_classifier import CardClassifier, load_pretrained_model
from training.trainer import train_model
from config.transforms import get_train_transforms, get_val_transforms, get_train_transforms_new, get_val_transforms_new

# --- Class mapping and sorting ---

def create_class_mapping(include_joker=False):
    suits = ['spades', 'hearts', 'diamonds', 'clubs']
    ranks = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven',
             'eight', 'nine', 'ten', 'jack', 'queen', 'king']
    class_names = [f'{rank} of {suit}' for suit in suits for rank in ranks]
    if include_joker:
        class_names.append('joker')
    return sort_card_names(class_names)

def sort_card_names(class_names):
    def card_sort_key(name):
        order_ranks = ['ace', 'two', 'three', 'four', 'five', 'six', 'seven',
                       'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'joker']
        order_suits = ['spades', 'hearts', 'diamonds', 'clubs']
        if name == 'joker':
            return (100, 100)
        rank, _, suit = name.partition(' of ')
        return (order_suits.index(suit), order_ranks.index(rank))
    return sorted(class_names, key=card_sort_key)

# --- Main logic ---

def main():
    parser = argparse.ArgumentParser(description='Train card classifier with different dataset options')
    parser.add_argument('--dataset', type=str, choices=['small-set', 'big-set'], required=True,
                        help='Choose which dataset to use (small-set or big-set)')
    args = parser.parse_args()

    if args.dataset == 'small-set':
        dataset_dir = Path('data/data-set-classification')
        train_dataset = ImageFolder(dataset_dir / 'train', transform=get_train_transforms())
        val_dataset = ImageFolder(dataset_dir / 'valid', transform=get_val_transforms())

        # Удаляем 'joker' из train/val вручную, если он есть
        if 'joker' in train_dataset.class_to_idx:
            joker_idx = train_dataset.class_to_idx['joker']
            train_dataset.samples = [s for s in train_dataset.samples if s[1] != joker_idx]
            val_dataset.samples = [s for s in val_dataset.samples if s[1] != joker_idx]

            # Переопределим class_names вручную, убрав 'joker'
            class_names = [c for c in train_dataset.classes if c != 'joker']
        else:
            class_names = train_dataset.classes

        class_names = sort_card_names(class_names)

    else:
        yolo_dir = Path('data/data-set-objectDetection')
        output_dir = Path('data/data_classification')

        if not output_dir.exists() or not any(output_dir.iterdir()):
            print('Converting YOLO dataset to classification format...')
            yolo_class_ids = get_yolo_class_ids(yolo_dir)
            include_joker = 52 in yolo_class_ids
            class_names = create_class_mapping(include_joker=include_joker)
            convert_yolo_to_classification(yolo_dir, output_dir, class_names, allow_joker=include_joker)
            validate_dataset(output_dir)
        else:
            print('Using existing converted dataset...')
            yolo_class_ids = get_yolo_class_ids(yolo_dir)
            include_joker = 52 in yolo_class_ids
            class_names = create_class_mapping(include_joker=include_joker)

        dataset_dir = output_dir
        train_dataset = ImageFolder(dataset_dir / 'train', transform=get_train_transforms_new())
        val_dataset = ImageFolder(dataset_dir / 'valid', transform=get_val_transforms_new())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    num_classes = len(class_names)
    print(class_names, num_classes)
    batch_size = 128 if device.type == 'cuda' else 32
    num_workers = 16 if device.type == 'cuda' else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    if os.path.exists('best_card_classifier.pth'):
        print('Loading pretrained model...')
        model = load_pretrained_model('best_card_classifier.pth', num_classes, device=device)
    else:
        print('Training from scratch...')
        model = CardClassifier(num_classes=num_classes).to(device)

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
        use_amp=device.type == 'cuda'
    )

if __name__ == '__main__':
    main()