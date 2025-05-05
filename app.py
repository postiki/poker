from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
from card_classifier import models

app = Flask(__name__)

# Загружаем модель
device = torch.device('cpu')
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 53)  # 52 карты + джокер
model.load_state_dict(torch.load('best_card_classifier.pth', map_location=device))
model.eval()

# Трансформации для изображения
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Словарь для преобразования индексов в названия карт
card_names = {
    i: name for i, name in enumerate([
        f"{rank} of {suit}" 
        for rank in ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'jack', 'queen', 'king', 'ace']
        for suit in ['spades', 'hearts', 'diamonds', 'clubs']
    ] + ['joker'])
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Читаем и преобразуем изображение
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    # Получаем предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
    
    card_name = card_names[predicted.item()]
    
    return jsonify({
        'card': card_name,
        'confidence': f"{confidence:.2%}"
    })

if __name__ == '__main__':
    app.run(debug=True)