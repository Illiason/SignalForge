✨ Features
📰 News Sentiment Analysis: Real-time processing of cryptocurrency news headlines

📈 Price Direction Prediction: Classifies Bitcoin price movements (UP/DOWN/FLAT) with confidence scores

🎯 Percentage Change Estimates: Provides estimated percentage change ranges based on confidence levels

⚡ GPU Accelerated: Uses PyTorch with CUDA support for lightning-fast predictions

🌐 Modern Web Interface: Clean, dark-themed UI with real-time visualizations

📊 Interactive Charts: Doughnut charts and confidence indicators for easy interpretation

🔍 Historical Analysis: Keeps track of recent predictions for reference

🚀 Quick Start
Prerequisites
Python 3.11+

NVIDIA GPU (optional, but recommended for faster training)

8GB+ RAM

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

text
Date,Price,Open,High,Low,Vol,Change %,News
07/01/2025,105694.3,107176.4,107532.3,105289.4,40280.0,-1.38,Bitcoin news headline...
 
🎯 Usage
1. Training the Model
bash
# The model will auto-train on first run if no pre-trained model exists
python app.py
2. Using Pre-trained Models
bash
# After training, models are saved automatically
# Next time, they'll load instantly
python app.py
3. Access the Web Interface
Start the server:

bash
python app.py
Open your browser to:

text
http://127.0.0.1:5000
Enter cryptocurrency news and click "Analyze Sentiment"

Example News Input
text
Bitcoin ETF Approval Expected This Week As SEC Deadline Approaches
🏗️ Project Structure
text
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

text
DistilBERT (Transformer) → [CLS] Token → 
FC Layers (128→64→3) → Softmax → Prediction
Model Components:
Base Model: DistilBERT (distilbert-base-uncased)

Classification Head: 3 fully-connected layers

Output: 3 classes (UP/DOWN/FLAT)

Training: Cross-entropy loss with AdamW optimizer

Early Stopping: Patience of 5 epochs

Performance Metrics:
Validation Accuracy: 74.13%

Training Accuracy: 86.63%

Training Time: ~2-5 minutes on GPU

Prediction Time: ~50-100ms per news item

🔌 API Endpoints
GET /
Description: Main web interface

Response: HTML page

POST /predict
Description: Analyze news sentiment

Request Body:

json
{
  "news": "Bitcoin ETF approval news..."
}
Response:

json
{
  "success": true,
  "predicted_direction": "UP",
  "confidence": 85.5,
  "estimated_range": "2.2% to 6.2% increase",
  "probabilities": {
    "UP": 85.5,
    "DOWN": 8.2,
    "FLAT": 6.3
  },
  "explanation": "📈 The model predicts Bitcoin price will INCREASE with 85.5% confidence",
  "reasoning": "Positive news sentiment suggests potential buying pressure..."
}
GET /status
Description: Check model status

Response:

json
{
  "model_trained": true,
  "device": "cuda"
}
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

🚀 Performance Optimization
GPU Acceleration
The model automatically uses GPU if available:

python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Model Caching
Pre-trained models are saved as model_weights.pth

Label encoders saved as label_encoder.pkl

Subsequent loads are instantaneous

Batch Processing
Training: Batch size of 2 (adjustable)

Inference: Single news items or batch support

🔧 Customization
Adjust Prediction Thresholds
Modify in arnn_model.py:

python
def categorize_direction(change):
    if change > 0.5:    # Increase threshold
        return 'UP'
    elif change < -0.5: # Decrease threshold  
        return 'DOWN'
    else:
        return 'FLAT'
Change Model Parameters
python
# In train() method:
predictor.train(df, 
    epochs=10,           # Training epochs
    batch_size=2,        # Batch size
    learning_rate=2e-5   # Learning rate
)
🐛 Troubleshooting
Common Issues:
1. CUDA not available

bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA version of PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
2. Model retrains every time

Ensure model_weights.pth and label_encoder.pkl exist

Disable debug mode in app.py:

python
app.run(debug=False, use_reloader=False)
3. Missing dependencies

bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
4. Port already in use

python
# Change port in app.py
app.run(port=5001)  # Use a different port
📈 Model Evaluation
Current Performance:
Direction Accuracy: 74.13% (validation)

Confidence Calibration: Well-calibrated (55-95% range)

Training Stability: Early stopping prevents overfitting

Sample Predictions:
News	Prediction	Confidence	Est. Range
"Bitcoin ETF approved"	UP	92%	2.3-6.6% increase
"Exchange hacked"	DOWN	88%	2.1-6.4% decrease
"Regulatory clarity"	FLAT	65%	-1.1 to +1.1%
🔮 Future Enhancements
Planned Features:
Multi-coin support (ETH, SOL, etc.)

Real-time news feeds integration

Advanced technical indicators

Ensemble models for improved accuracy

Mobile app (React Native)

API rate limiting & authentication

Research Directions:
Larger transformer models (RoBERTa, DeBERTa)

Attention visualization for interpretability

Multi-modal analysis (news + price charts)

Transfer learning from financial news

🤝 Contributing
We welcome contributions! Here's how:

Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

Areas for Contribution:
Model improvements

UI/UX enhancements

Additional data sources

Documentation

Bug fixes

📝 Citation
If you use SignalForge in your research, please cite:

bibtex
@software{signalforge2023,
  title = {SignalForge: Crypto News Sentiment Analyzer},
  author = {Your Name},
  year = {2023},
  url = {https://github.com/yourusername/signalforge}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Hugging Face for Transformers library

PyTorch for deep learning framework

Flask for web framework

Chart.js for data visualization

📞 Support
Issues: GitHub Issues

Discussions: GitHub Discussions

Email: your.email@example.com

<div align="center">
Made with ❤️ for the crypto community

Star this repo if you found it helpful!

</div>
