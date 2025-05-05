from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
from card_classifier import CardClassifier

app = Flask(__name__)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = CardClassifier(num_classes=52)  # Changed to 52 by default

try:
    checkpoint = torch.load('best_card_classifier.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded successfully. Best validation accuracy: {checkpoint['val_acc']:.2f}%")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

model.to(device)
model.eval()

# Validation transform (no augmentations)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Card names mapping
card_names = {
    0: 'ace of clubs', 1: 'ace of diamonds', 2: 'ace of hearts', 3: 'ace of spades',
    4: 'eight of clubs', 5: 'eight of diamonds', 6: 'eight of hearts', 7: 'eight of spades',
    8: 'five of clubs', 9: 'five of diamonds', 10: 'five of hearts', 11: 'five of spades',
    12: 'four of clubs', 13: 'four of diamonds', 14: 'four of hearts', 15: 'four of spades',
    16: 'jack of clubs', 17: 'jack of diamonds', 18: 'jack of hearts', 19: 'jack of spades',
    20: 'joker', 21: 'king of clubs', 22: 'king of diamonds', 23: 'king of hearts',
    24: 'king of spades', 25: 'nine of clubs', 26: 'nine of diamonds', 27: 'nine of hearts',
    28: 'nine of spades', 29: 'queen of clubs', 30: 'queen of diamonds', 31: 'queen of hearts',
    32: 'queen of spades', 33: 'seven of clubs', 34: 'seven of diamonds', 35: 'seven of hearts',
    36: 'seven of spades', 37: 'six of clubs', 38: 'six of diamonds', 39: 'six of hearts',
    40: 'six of spades', 41: 'ten of clubs', 42: 'ten of diamonds', 43: 'ten of hearts',
    44: 'ten of spades', 45: 'three of clubs', 46: 'three of diamonds', 47: 'three of hearts',
    48: 'three of spades', 49: 'two of clubs', 50: 'two of diamonds', 51: 'two of hearts',
    52: 'two of spades'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and validate image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Basic image validation
        if image.size[0] < 50 or image.size[1] < 50:
            return jsonify({'error': 'Image too small'}), 400
        
        # Process image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities[0], 3)
            top3_predictions = [
                {
                    'card': card_names[idx.item()],
                    'confidence': prob.item()
                }
                for prob, idx in zip(top3_prob, top3_indices)
            ]
        
        return jsonify({
            'predictions': top3_predictions,
            'best_prediction': {
                'card': card_names[predicted_class],
                'confidence': confidence
            }
        })
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(debug=True)