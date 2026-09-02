[![Tests](https://github.com/Illiason/SignalForge/actions/workflows/tests.yml/badge.svg)](https://github.com/Illiason/SignalForge/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker Ready](https://img.shields.io/badge/docker-ready-brightgreen.svg)](./DOCKER.md)
[![Code style](https://img.shields.io/badge/code%20style-clean-lightgrey.svg)]()

# SignalForge

Transformer-powered sentiment analysis for Bitcoin price prediction. Analyzes cryptocurrency news headlines to predict price movements in real-time.

## ✨ Features

- 📰 **News Sentiment Analysis** — Real-time processing of cryptocurrency news headlines
- 📈 **Multi-Coin Support** — Analyze Bitcoin, Ethereum, Solana, Cardano, Polkadot, XRP (easily extensible)
- 🎯 **Price Direction Prediction** — Classifies price movements (UP/DOWN/FLAT) with confidence scores per coin
- 📊 **Interactive Charts** — Doughnut charts and confidence indicators for easy interpretation
- ⚡ **Fast Model Loading** — Trained model loads in ~1 second on subsequent runs
- 🔧 **Configurable** — Environment variables for flexible configuration
- 🐳 **Docker Ready** — One-command deployment with Docker Compose
- ✅ **Well-Tested** — 30 passing unit and integration tests

🚀 Quick Start
Prerequisites
Python 3.11+

Installation
Clone the repository

git clone https://github.com/yourusername/signalforge.git
cd signalforge
Create a virtual environment (optional but recommended)

python -m venv venv
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
Set up your data

Place your crypto_dataset.csv in the project root

Or use the example dataset structure:

Date,Price,Open,High,Low,Vol,Change %,News
07/01/2025,105694.3,107176.4,107532.3,105289.4,40280.0,-1.38,Bitcoin news headline...
 
🎯 Usage

1. **First Run** — Model Training & Saving
   - On first startup, the app auto-trains if no saved weights exist (2–5 min)
   - Training checkpoints are saved to `model_weights.pth` and `label_encoder.pkl`
   - Next run loads these instantly (~1 sec startup)

2. **Startup** — Model Loading (Fast Path)
   ```bash
   python app.py  # Loads saved model in ~1 second
   ```

3. **Force Retraining**
   ```bash
   RETRAIN=1 python app.py  # Ignores saved weights, trains fresh
   ```
   Or set `RETRAIN=true` or `RETRAIN=yes` — useful for hyperparameter tuning or new data.

4. **Using the App**
   - Open browser: http://127.0.0.1:5000
   - **Select a cryptocurrency** from the dropdown (Bitcoin, Ethereum, Solana, Cardano, Polkadot, XRP)
   - Enter cryptocurrency news or headline
   - Click "Analyze Sentiment" to get direction prediction (UP/DOWN/FLAT)

   Example News Input:
   > Bitcoin ETF Approval Expected This Week As SEC Deadline Approaches
   > Ethereum Shanghai Upgrade Successfully Completes Transition to Proof-of-Stake

5. **Configuration** (Optional)
   - Copy `.env.example` to `.env` and edit for custom settings
   - `RETRAIN=1` — force retrain on startup
   - `FLASK_DEBUG=1` — enable debug mode (dev only)
   - `FLASK_HOST` / `FLASK_PORT` — customize server address

6. **Run Tests**
   ```bash
   pytest test_arnn_model.py test_app.py -v  # 27 tests covering model & API
   ```

🏗️ Project Structure

signalforge/
├── app.py                 # Flask application & API endpoints
├── arnn_model.py          # Model architecture & training logic
├── crypto_dataset.csv     # Training data (your crypto news dataset)
├── model_weights.pth      # Saved model weights (auto-generated)
├── label_encoder.pkl      # Label encoder (auto-generated)
├── templates/
│   └── index.html         # Modern web interface
├── requirements.txt       # Python dependencies
└── README.md             # This file

🤖 Model Architecture
SignalForge uses a hybrid neural network architecture:

DistilBERT (Transformer) → [CLS] Token → 
FC Layers (128→64→3) → Softmax → Prediction
Model Components:
Base Model: DistilBERT (distilbert-base-uncased)

Classification Head: 3 fully-connected layers

Output: 3 classes (UP/DOWN/FLAT)

Training: Cross-entropy loss

Early Stopping: 5 epochs

Performance Metrics:
Validation Accuracy: 74.13%

Training Accuracy: 86.63%

Training Time: approx 2-5 minutes on GPU

🎨 Web Interface Features
Main Dashboard
News Input Panel: Paste or type cryptocurrency news

Recent Analysis: History of previous predictions

Real-time Results:

Sentiment badges (Positive/Negative/Neutral)

Confidence indicators (55-95%)

Estimated percentage change ranges

Interactive probability charts

Visual Elements
Dark Modern Theme: Easy on the eyes during extended use

Responsive Design: Works on desktop and mobile

Animated Transitions: Smooth UI interactions

Color-coded Results: Green for UP, Red for DOWN, Yellow for FLAT

📊 Dataset Format
The model expects a CSV with these columns (example):

File type is csv
Date,Price,Open,High,Low,Vol,Change %,News
07/01/2025,105694.3,107176.4,107532.3,105289.4,40280.0,-1.38,Bitcoin pulled back after record close...
06/30/2025,107171.1,108362.3,108777.0,106743.1,37460.0,-1.1,Highest monthly close ever...

Data Preprocessing:
Price and percentage columns cleaned

News text tokenized (max 128 tokens)

Labels encoded (UP=0, DOWN=1, FLAT=2)

Stratified train/validation split (80/20)

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.


📞 Support

Email: illialysennyi@gmail.com

<div align="center">
Made with ❤️ for the crypto community :)
Star this repo if you found it helpful!
</div>
