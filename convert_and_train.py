import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader

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

# --- Custom ImageFolder with consistent class_to_idx ---

def get_custom_imagefolder(root, transform, class_names):
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    samples = []
    for cls_name in class_names:
        cls_path = os.path.join(root, cls_name)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            if os.path.isfile(fpath):
                samples.append((fpath, class_to_idx[cls_name]))

    return DatasetWithCustomClasses(samples, class_to_idx, transform)

class DatasetWithCustomClasses(torch.utils.data.Dataset):
    def __init__(self, samples, class_to_idx, transform):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.classes = list(class_to_idx.keys())
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = default_loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.samples)

# --- Main logic ---

def main():
    parser = argparse.ArgumentParser(description='Train card classifier with different dataset options')
    parser.add_argument('--dataset', type=str, choices=['small-set', 'big-set'], required=True,
                        help='Choose which dataset to use (small-set or big-set)')
    args = parser.parse_args()

    if args.dataset == 'small-set':
        dataset_dir = Path('data/data-set-classification')
        existing_classes = sorted(os.listdir(dataset_dir / 'train'))
        include_joker = 'joker' in existing_classes
        class_names = create_class_mapping(include_joker=include_joker)
        if not include_joker:
            class_names = [c for c in class_names if c != 'joker']

        train_dataset = get_custom_imagefolder(dataset_dir / 'train', get_train_transforms(), class_names)
        val_dataset = get_custom_imagefolder(dataset_dir / 'valid', get_val_transforms(), class_names)

    else:
        yolo_dir = Path('data/data-set-objectDetection')
        output_dir = Path('data/data_classification')

        if not output_dir.exists() or not any(output_dir.iterdir()):
            print('📦 Converting YOLO dataset...')
            yolo_class_ids = get_yolo_class_ids(yolo_dir)
            include_joker = 52 in yolo_class_ids
            class_names = create_class_mapping(include_joker=include_joker)
            convert_yolo_to_classification(yolo_dir, output_dir, class_names, allow_joker=include_joker)
            validate_dataset(output_dir)
        else:
            print('✅ Using existing converted dataset...')
            yolo_class_ids = get_yolo_class_ids(yolo_dir)
            include_joker = 52 in yolo_class_ids
            class_names = create_class_mapping(include_joker=include_joker)

        dataset_dir = output_dir
        train_dataset = get_custom_imagefolder(dataset_dir / 'train', get_train_transforms_new(), class_names)
        val_dataset = get_custom_imagefolder(dataset_dir / 'valid', get_val_transforms_new(), class_names)

    # --- Logging and training ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print("🔍 Checking class name consistency...")
    print("Expected class_names:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")

    print("Actual class_to_idx from dataset:")
    for name, idx in train_dataset.class_to_idx.items():
        print(f"  {idx}: {name}")

    if set(class_names) != set(train_dataset.class_to_idx.keys()):
        print("❗ WARNING: class_names and dataset class_to_idx do NOT match.")
    else:
        print("✅ class_names and dataset class_to_idx are consistent.")

    print(device.type)

    batch_size = 128 if device.type == 'cuda' else 32
    num_workers = 16 if device.type == 'cuda' else 4

    print('🧪 Creating DataLoaders...')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    # print('📥 Loading model...')
    # if os.path.exists('best_card_classifier.pth'):
    #     model = load_pretrained_model('best_card_classifier.pth', len(class_names), device=device)
    #     print('✔️ Loaded pretrained model')
    # else:
    model = CardClassifier(num_classes=len(class_names)).to(device)
    print('🛠️ Training from scratch')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    print('🏋️ Starting training loop...')
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        device=device,
        use_amp=(device.type == 'cuda')
    )

if __name__ == '__main__':
    main()
