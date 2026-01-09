from flask import Flask, render_template, request, jsonify
import torch
import pandas as pd
from arnn_model import BitcoinPricePredictor
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initializes predictor
predictor = BitcoinPricePredictor(device=device)
model_trained = False

def initialize_model():
    global model_trained
    try:
        if os.path.exists('crypto_dataset.csv'):
            df = pd.read_csv('crypto_dataset.csv')
            print("Dataset loaded successfully!")
            print(f"Dataset shape: {df.shape}")
            
            # Trains the model with simpler configuration
            predictor.train(df, epochs=10, batch_size=2, learning_rate=2e-5)
            model_trained = True
            print("Model trained successfully!")
        else:
            print("Dataset file not found! Using rule-based predictions.")
            # Marks as trained to use rule-based predictions
            model_trained = True
    except Exception as e:
        print(f"Error during model training: {e}")
        print("Falling back to rule-based predictions.")
        model_trained = True  # allows predictions with rule-based method

# Initializes model when app starts
print("Initializing model...")
initialize_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model_trained:
            return jsonify({
                'success': False,
                'error': 'Model not ready yet. Please wait...'
            })
        
        news_text = request.json['news']
        result = predictor.predict_with_explanation(news_text)
        
        response = {
            'success': True,
            'predicted_direction': result['predicted_direction'],
            'probabilities': result['probabilities'],
            'confidence': result['confidence'],
            'explanation': result['explanation'],
            'reasoning': result['reasoning'],
            'news': news_text
        }
        
    except Exception as e:
        response = {
            'success': False,
            'error': str(e)
        }
    
    return jsonify(response)

@app.route('/status')
def status():
    return jsonify({
        'model_trained': model_trained,
        'device': str(predictor.device)
    })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    print("Starting Flask server...")
    app.run(debug=True, port=5000, host='0.0.0.0')

    print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")