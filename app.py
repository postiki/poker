from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from models.card_classifier import CardClassifier

app = Flask(__name__)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = CardClassifier(num_classes=52)  # Changed to 53 to match dataset

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
    0: 'ace of spades', 1: 'two of spades', 2: 'three of spades', 3: 'four of spades',
    4: 'five of spades', 5: 'six of spades', 6: 'seven of spades', 7: 'eight of spades',
    8: 'nine of spades', 9: 'ten of spades', 10: 'jack of spades', 11: 'queen of spades',
    12: 'king of spades', 13: 'ace of hearts', 14: 'two of hearts', 15: 'three of hearts',
    16: 'four of hearts', 17: 'five of hearts', 18: 'six of hearts', 19: 'seven of hearts',
    20: 'eight of hearts', 21: 'nine of hearts', 22: 'ten of hearts', 23: 'jack of hearts',
    24: 'queen of hearts', 25: 'king of hearts', 26: 'ace of diamonds', 27: 'two of diamonds',
    28: 'three of diamonds', 29: 'four of diamonds', 30: 'five of diamonds', 31: 'six of diamonds',
    32: 'seven of diamonds', 33: 'eight of diamonds', 34: 'nine of diamonds', 35: 'ten of diamonds',
    36: 'jack of diamonds', 37: 'queen of diamonds', 38: 'king of diamonds', 39: 'ace of clubs',
    40: 'two of clubs', 41: 'three of clubs', 42: 'four of clubs', 43: 'five of clubs',
    44: 'six of clubs', 45: 'seven of clubs', 46: 'eight of clubs', 47: 'nine of clubs',
    48: 'ten of clubs', 49: 'jack of clubs', 50: 'queen of clubs', 51: 'king of clubs'
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