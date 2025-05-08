import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def is_blank_image(img, threshold=5):
    if img is None or img.size == 0:
        return True
    if np.std(img) < threshold:
        return True
    return False

def convert_yolo_to_classification(yolo_dir, output_dir, class_names, min_size=10, allow_joker=True):
    yolo_dir = Path(yolo_dir)
    output_dir = Path(output_dir)
    
    for split in ['train', 'valid']:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for class_id in range(len(class_names)):
            (split_dir / str(class_id)).mkdir(exist_ok=True)
    
    total_crops = 0
    skipped_crops = 0
    
    for split in ['train', 'valid']:
        images_dir = yolo_dir / split / 'images'
        labels_dir = yolo_dir / split / 'labels'
        
        print(f'Processing {split} split:')
        print(f'Images directory: {images_dir}')
        print(f'Labels directory: {labels_dir}')
        
        if not images_dir.exists() or not labels_dir.exists():
            print(f'Warning: Missing images or labels directory for {split} split')
            continue
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        print(f'Found {len(image_files)} images in {split} split')
        
        for img_path in tqdm(image_files, desc=f'Processing {split} images'):
            txt_path = labels_dir / f'{img_path.stem}.txt'
            img = cv2.imread(str(img_path))
            if img is None:
                print(f'Warning: Could not read image {img_path}')
                continue
                
            height, width = img.shape[:2]
            has_valid_cards = False
            
            if txt_path.exists():
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f'Warning: Invalid annotation format in {txt_path}: {line.strip()}')
                            continue
                            
                        class_id, x_center, y_center, w, h = map(float, parts)
                        class_id = int(class_id)
                        
                        if not allow_joker and class_id == 52:
                            continue

                        padding = 0.25

                        x1 = int((x_center - w / 2 - padding) * width)
                        y1 = int((y_center - h / 2 - padding) * height)
                        x2 = int((x_center + w / 2 + padding) * width)
                        y2 = int((y_center + h / 2 + padding) * height)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)

                        crop_w = x2 - x1
                        crop_h = y2 - y1
                        
                        if crop_w < min_size or crop_h < min_size:
                            skipped_crops += 1
                            continue
                            
                        card_img = img[y1:y2, x1:x2]
                        if is_blank_image(card_img):
                            skipped_crops += 1
                            continue
                            
                        class_dir = output_dir / split / str(class_id)
                        output_path = class_dir / f'{img_path.stem}_{x1}_{y1}.jpg'
                        cv2.imwrite(str(output_path), card_img)
                        total_crops += 1
                        has_valid_cards = True
            
            if not has_valid_cards:
                class_dir = output_dir / split / '0'
                output_path = class_dir / f'{img_path.stem}_full.jpg'
                cv2.imwrite(str(output_path), img)
                total_crops += 1
    
    for split in ['train', 'valid']:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            if not any(class_dir.iterdir()):
                print(f'Removing empty class directory: {class_dir}')
                class_dir.rmdir()
    
    print(f'Dataset conversion complete:')
    print(f'- Total crops saved: {total_crops}')
    print(f'- Crops skipped (too small or blank): {skipped_crops}')
    print(f'- Output directory: {output_dir}') 