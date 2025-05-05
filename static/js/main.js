document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const preview = document.getElementById('preview');
    const previewImage = document.getElementById('preview-image');
    const predictionText = document.getElementById('prediction-text');
    const confidenceText = document.getElementById('confidence-text');
    const error = document.getElementById('error');
    const errorMessage = document.getElementById('error-message');

    const cardNames = {
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
    };

    function showError(message) {
        errorMessage.textContent = message;
        error.classList.remove('hidden');
        setTimeout(() => {
            error.classList.add('hidden');
        }, 5000);
    }

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            preview.classList.remove('hidden');
            predictionText.textContent = 'Analyzing...';
            confidenceText.textContent = '';
            preview.classList.add('loading');

            const formData = new FormData();
            formData.append('file', file);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                preview.classList.remove('loading');
                if (data.error) {
                    showError(data.error);
                    predictionText.textContent = 'Error';
                    confidenceText.textContent = '';
                } else {
                    const bestPrediction = data.best_prediction;
                    predictionText.innerHTML = `
                        <div class="space-y-2">
                            <div class="font-medium">${bestPrediction.card}</div>
                            <div class="text-sm text-gray-500">Confidence: ${(bestPrediction.confidence * 100).toFixed(2)}%</div>
                            <div class="mt-2 text-sm">
                                <div class="font-medium text-gray-700">Other possibilities:</div>
                                ${data.predictions.slice(1).map(p => 
                                    `<div class="text-gray-600">${p.card} (${(p.confidence * 100).toFixed(2)}%)</div>`
                                ).join('')}
                            </div>
                        </div>
                    `;
                }
            })
            .catch(error => {
                preview.classList.remove('loading');
                showError('Error processing image');
                predictionText.textContent = 'Error';
                confidenceText.textContent = '';
            });
        };
        reader.readAsDataURL(file);
    }

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
}); 