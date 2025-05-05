import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from tqdm import tqdm

def prepare_dataset():
    if not os.path.exists('data'):
        os.makedirs('data')
    
    print("Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()
    
    print("Downloading dataset...")
    api.dataset_download_files('robikscube/playing-cards-dataset', path='data')
    
    zip_path = 'data/playing-cards-dataset.zip'
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')
    
    os.remove(zip_path)
    print("Dataset preparation completed!")

def check_dataset():
    dataset_path = 'data/playing_cards'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found at {dataset_path}")
        print("Please create the directory and place your card images in the following structure:")
        print("data/playing_cards/")
        print("├── ace_of_hearts/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("├── king_of_spades/")
        print("│   ├── image1.jpg")
        print("│   └── image2.jpg")
        print("└── ... (other card directories)")
        return False
    
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if not classes:
        print("Error: No class directories found in the dataset")
        return False
    
    print(f"Found {len(classes)} card classes:")
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"- {class_name}: {len(images)} images")
    
    return True

if __name__ == "__main__":
    prepare_dataset()
    check_dataset() 