from flask import Flask, render_template, request, jsonify
from typing import Dict, Tuple, Any
import torch
import pandas as pd
from arnn_model import BitcoinPricePredictor
import os
import logging
import warnings

# Suppress only known noisy warnings from dependencies (transformers, torch)
# while allowing legitimate warnings through
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torch')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Initializes predictor
predictor = BitcoinPricePredictor(device=device)
model_trained = False

def initialize_model() -> None:
    """Initialize the model on app startup."""
    global model_trained
    try:
        if os.path.exists('crypto_dataset.csv'):
            df = pd.read_csv('crypto_dataset.csv')
            logger.info("Dataset loaded successfully!")
            logger.info(f"Dataset shape: {df.shape}")

            # Check if forced retrain is requested via env var
            retrain = os.getenv('RETRAIN', '').lower() in ('1', 'true', 'yes')

            # Try loading saved model first (unless RETRAIN is set); only retrain if weights don't exist
            if not retrain and predictor.load_saved_model():
                model_trained = True
                logger.info("Model loaded successfully!")
            else:
                if retrain:
                    logger.info("RETRAIN=1 detected. Training a new model...")
                else:
                    logger.info("No saved model found. Training a new model...")
                predictor.train(df, epochs=10, batch_size=2, learning_rate=2e-5)
                model_trained = True
                logger.info("Model trained successfully!")
        else:
            logger.warning("Dataset file not found! Using rule-based predictions.")
            # Marks as trained to use rule-based predictions
            model_trained = True
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        logger.warning("Falling back to rule-based predictions.")
        model_trained = True  # allows predictions with rule-based method

# Initializes model when app starts
logger.info("Initializing model...")
initialize_model()

@app.route('/')
def home() -> str:
    """Render the main UI page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict() -> Tuple[Dict[str, Any], int]:
    try:
        if not model_trained:
            return jsonify({
                'success': False,
                'error': 'Model not ready yet. Please wait...'
            }), 503

        # Validate input
        data = request.get_json(silent=True)
        if not data or 'news' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: news'
            }), 400

        news_text = data['news']
        if not isinstance(news_text, str) or not news_text.strip():
            return jsonify({
                'success': False,
                'error': 'News text must be a non-empty string'
            }), 400

        # Cap input length to prevent abuse
        MAX_NEWS_LENGTH = 5000
        if len(news_text) > MAX_NEWS_LENGTH:
            return jsonify({
                'success': False,
                'error': f'News text exceeds maximum length of {MAX_NEWS_LENGTH} characters'
            }), 400

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
        return jsonify(response), 200

    except Exception as e:
        logger.exception("Unhandled error in /predict")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/status')
def status() -> Tuple[Dict[str, Any], int]:
    """Return model readiness status."""
    return jsonify({
        'model_trained': model_trained,
        'device': str(predictor.device)
    }), 200

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    debug_mode = os.getenv('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', '5000'))
    logger.info(f"Starting Flask server on {host}:{port} (debug={debug_mode})")
    app.run(debug=debug_mode, port=port, host=host)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")