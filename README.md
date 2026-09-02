[![Tests](https://github.com/Illiason/SignalForge/actions/workflows/tests.yml/badge.svg)](https://github.com/Illiason/SignalForge/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker Ready](https://img.shields.io/badge/docker-ready-brightgreen.svg)](./DOCKER.md)

# SignalForge

Transformer-powered sentiment analysis for Bitcoin price prediction. It reads cryptocurrency news headlines and predicts whether the price is about to go up, down, or stay flat.

## Features

- News sentiment analysis - processes crypto headlines in real time
- Multi-coin support - Bitcoin, Ethereum, Solana, Cardano, Polkadot, XRP out of the box, easy to extend
- Price direction prediction - classifies UP / DOWN / FLAT with a confidence score per coin
- Interactive charts - doughnut chart and confidence bar so results are easy to read at a glance
- Fast model loading - once trained, it loads from disk in about a second
- Configurable through environment variables
- Docker support, one command to deploy
- 33 passing tests covering the model and the API

## Quick Start

### Prerequisites

Python 3.11 or newer.

### Installation

```bash
git clone https://github.com/Illiason/SignalForge.git
cd SignalForge
python -m venv venv
venv\Scripts\activate   # on Windows
pip install -r requirements.txt
```

### Data

Put a `crypto_dataset.csv` in the project root. It needs these columns:

```csv
Date,Price,Open,High,Low,Vol,Change %,News
07/01/2025,105694.3,107176.4,107532.3,105289.4,40280.0,-1.38,Bitcoin news headline...
```

## Usage

**First run - training.** If there's no saved model yet, the app trains one automatically. That takes 2-5 minutes. The result is saved to `model_weights.pth` and `label_encoder.pkl`, so it only has to happen once.

**Every run after that - fast startup.**

```bash
python app.py   # loads the saved model in about a second
```

**Forcing a retrain.**

```bash
RETRAIN=1 python app.py
```

`RETRAIN=true` and `RETRAIN=yes` also work. Useful after changing the dataset or tuning hyperparameters.

**Using it.** Open `http://127.0.0.1:5000`, pick a coin from the dropdown, paste in a headline, and hit Analyze. A couple of headlines to try:

> Bitcoin ETF Approval Expected This Week As SEC Deadline Approaches
> Ethereum Shanghai Upgrade Successfully Completes Transition to Proof-of-Stake

**Configuration.** Copy `.env.example` to `.env` and adjust as needed:

- `RETRAIN=1` - force a retrain on startup
- `FLASK_DEBUG=1` - turn on debug mode (development only, never in production)
- `FLASK_HOST` / `FLASK_PORT` - change where the server listens

**Tests.**

```bash
pytest test_arnn_model.py test_app.py -v   # 33 tests, model and API
```

## Project Structure

```
SignalForge/
├── app.py               Flask app and API endpoints
├── arnn_model.py         Model architecture and training logic
├── crypto_dataset.csv    Training data (your crypto news dataset)
├── model_weights.pth     Saved model weights (generated after first run)
├── label_encoder.pkl     Label encoder (generated after first run)
├── templates/
│   └── index.html        Web interface
├── requirements.txt      Python dependencies
└── README.md
```

## Model Architecture

A DistilBERT transformer with a small classification head on top:

```
DistilBERT -> [CLS] token -> FC layers (128 -> 64 -> 3) -> Softmax -> prediction
```

- Base model: `distilbert-base-uncased`
- Classification head: 3 fully connected layers
- Output: 3 classes (UP / DOWN / FLAT)
- Loss: cross-entropy
- Early stopping after 5 epochs without improvement

On the held-out validation split it lands around 74% accuracy, versus about 87% on training data. That gap is typical for a dataset this size and is something still being worked on. Training takes roughly 2-5 minutes on a GPU.

## Web Interface

The dashboard has a news input box, a list of recent analyses, and a result panel: sentiment badge, confidence percentage, an estimated range for the price move, and a probability chart. Dark theme, responsive layout, colors map to direction - green for up, red for down, yellow for flat.

## Dataset Format

The training CSV needs these columns:

```csv
Date,Price,Open,High,Low,Vol,Change %,News
07/01/2025,105694.3,107176.4,107532.3,105289.4,40280.0,-1.38,Bitcoin pulled back after record close...
06/30/2025,107171.1,108362.3,108777.0,106743.1,37460.0,-1.1,Highest monthly close ever...
```

Before training: price and percentage columns are cleaned, news text is tokenized to a max of 128 tokens, labels are encoded as UP=0 / DOWN=1 / FLAT=2, and the train/validation split is stratified 80/20.

## License

MIT - see the [LICENSE](./LICENSE) file.

## Support

Questions or bugs: illialysennyi@gmail.com

<div align="center">
Made for the crypto community.
If this was useful to you, a star on the repo is appreciated.
</div>
