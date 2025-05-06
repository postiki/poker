from pathlib import Path

def get_yolo_class_ids(yolo_dir):
    yolo_dir = Path(yolo_dir)
    class_ids = set()
    
    for split in ['train', 'valid']:
        labels_dir = yolo_dir / split / 'labels'
        if not labels_dir.exists():
            continue
            
        for txt_file in labels_dir.glob('*.txt'):
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        class_ids.add(class_id)
    
    return sorted(class_ids)

def validate_dataset(output_dir):
    output_dir = Path(output_dir)
    total_images = 0
    class_counts = {}
    
    for split in ['train', 'valid']:
        split_dir = output_dir / split
        if not split_dir.exists():
            print(f'Warning: {split} directory not found')
            continue
            
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_id = int(class_dir.name)
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            class_counts[class_id] = len(images)
            total_images += len(images)
    
    print(f'Dataset validation results:')
    print(f'- Total images: {total_images}')
    print(f'- Classes found: {len(class_counts)}')
    print(f'- Images per class:')
    for class_id, count in sorted(class_counts.items()):
        print(f'  Class {class_id}: {count} images') 