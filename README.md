📰 News Sentiment Analysis: Real-time processing of cryptocurrency news headlines

📈 Price Direction Prediction: Classifies Bitcoin price movements (UP/DOWN/FLAT) with confidence scores

🎯 Percentage Change Estimates: Provides estimated percentage change ranges based on confidence levels

⚡ GPU Accelerated: Uses PyTorch with CUDA support for lightning-fast predictions

🌐 Modern Web Interface: Clean, dark-themed UI with real time visualizations

📊 Interactive Charts: Doughnut charts and confidence indicators for easy interpretation

🔍 Historical Analysis: Keeps track of recent predictions for reference

🚀 Quick Start
Prerequisites
Python 3.11+

8GB+ RAM

Installation
Clone the repository

git clone https://github.com/yourusername/signalforge.git
cd signalforge
Create a virtual environment (optional but recommended)

-m venv venv
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
Set up your data

Place your crypto_dataset.csv in the project root

Or use the example dataset structure:

text
Date,Price,Open,High,Low,Vol,Change %,News
07/01/2025,105694.3,107176.4,107532.3,105289.4,40280.0,-1.38,Bitcoin news headline...
 
🎯 Usage
1. Training the Model

# The model will auto-train on first run if no pre-trained model exists

# After training, models are saved automatically
# Next time, they'll load instantly

Open your browser to:

http://127.0.0.1:5000
Enter cryptocurrency news and click "Analyze Sentiment"

Example News Input

Bitcoin ETF Approval Expected This Week As SEC Deadline Approaches

🏗️ Project Structure

signalforge/
├── app.py                 # Flask application & API endpoints
├── arnn_model.py          # AI model architecture & training logic
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

Training Time: ~2-5 minutes on GPU

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

csv
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
