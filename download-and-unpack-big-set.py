import kagglehub
import os
import shutil

target_dir = "./data/data-set-objectDetection"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

path = kagglehub.dataset_download("vdntdesai11/playing-cards")

print("Path to dataset files:", path)

if os.path.exists(path):
    for item in os.listdir(path):
        source = os.path.join(path, item)
        destination = os.path.join(target_dir, item)
        if os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(source, destination)
    print(f"Dataset has been unpacked to {target_dir}")