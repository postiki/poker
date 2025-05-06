import torch
import cv2
import numpy as np
from PIL import Image

def predict_on_crop(model, image_path, bbox, class_names, device, transform):
    x1, y1, x2, y2 = bbox
    img = cv2.imread(str(image_path))
    if img is None:
        return None, 0.0
        
    card_img = img[y1:y2, x1:x2]
    if card_img.size == 0:
        return None, 0.0
        
    card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
    card_img = Image.fromarray(card_img)
    card_img = transform(card_img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        try:
            with torch.autocast(device_type='mps', dtype=torch.float16):
                outputs = model(card_img)
        except:
            outputs = model(card_img)
            
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return class_names[predicted.item()], confidence.item() 