import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import os

class NewsPriceDataset(Dataset):
    def __init__(self, texts, price_directions, tokenizer, max_length=128):
        self.texts = texts
        self.price_directions = price_directions
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        price_direction = self.price_directions[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(price_direction, dtype=torch.long)
        }

class SimpleNewsModel(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=64, dropout=0.3):
        super(SimpleNewsModel, self).__init__()
        
        # DistilBERT for faster training
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.bert_dim = 768
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token for classification
        cls_output = bert_output.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits

class BitcoinPricePredictor:
    def __init__(self, device='cpu'):
        # Set device properly
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
            
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        print(f"BitcoinPricePredictor initialized on: {self.device}")
        
    def clean_data(self, df):
        # Cleans numerical columns
        if 'Price' in df.columns:
            df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
        if 'Change %' in df.columns:
            df['Change %'] = df['Change %'].astype(str).str.replace('%', '').astype(float)
        return df
    
    def create_price_direction_categories(self, df):
        # More balanced categorization for small dataset
        def categorize_direction(change):
            if change > 0.5:  # More than 0.5% increase
                return 'UP'
            elif change < -0.5:  # More than 0.5% decrease
                return 'DOWN'
            else:  # Between -0.5% and +0.5%
                return 'FLAT'
        
        if 'Change %' in df.columns:
            df['price_direction'] = df['Change %'].apply(categorize_direction)
        else:
            # If no Change % column, create dummy labels
            df['price_direction'] = 'FLAT'
            
        return df
    
    def prepare_data(self, df):
        df = self.clean_data(df)
        df = self.create_price_direction_categories(df)
        
        # Finds the news column
        news_columns = ['News', 'news', 'cleaned_news', 'text', 'Text', 'article']
        news_column = None
        for col in news_columns:
            if col in df.columns:
                news_column = col
                break
        
        if news_column is None:
            # Trys to find any text column
            for col in df.columns:
                if df[col].dtype == 'object' and len(df[col].iloc[0]) > 20:
                    news_column = col
                    break
        
        if news_column is None:
            raise ValueError("No news column found in dataset")
            
        texts = df[news_column].astype(str).tolist()
        labels = df['price_direction'].tolist()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        print(f"Price direction distribution:")
        direction_counts = pd.Series(labels).value_counts()
        print(direction_counts)
        
        return texts, encoded_labels
    
    def train(self, df, epochs=15, batch_size=2, learning_rate=2e-5):
        texts, labels = self.prepare_data(df)
        
        # Initializing tokenizer DistilBERT for speed
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        
        # Creates datasets
        train_dataset = NewsPriceDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = NewsPriceDataset(val_texts, val_labels, self.tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = SimpleNewsModel(num_classes=len(self.label_encoder.classes_))
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        print("Starting training...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                # Move data to GPU
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Move data to GPU
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save model and label encoder separately
                torch.save(self.model.state_dict(), 'model_weights.pth')
                with open('label_encoder.pkl', 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                print(f'  New best model saved with validation accuracy: {val_acc:.2f}%')
            else:
                patience_counter += 1
                print(f'  No improvement for {patience_counter} epochs')
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if os.path.exists('model_weights.pth'):
            # Use weights_only=False
            model_weights = torch.load('model_weights.pth', weights_only=False)
            self.model.load_state_dict(model_weights)
            
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        else:
            print("Training completed but no model was saved.")
    
    def predict(self, news_text):
        if self.model is None or self.tokenizer is None or self.label_encoder is None:
            # If model not trained, using fallback
            return self.rule_based_predict(news_text)
        
        self.model.eval()
        
        # Tokenize input
        encoding = self.tokenizer(
            news_text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to GPU
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)
        
        # Convert to readable format
        class_probabilities = {
            self.label_encoder.inverse_transform([i])[0]: round(prob.item() * 100, 2)
            for i, prob in enumerate(probabilities[0])
        }
        
        predicted_direction = self.label_encoder.inverse_transform(predicted_class.cpu())[0]
        
        return {
            'probabilities': class_probabilities,
            'predicted_direction': predicted_direction,
            'confidence': max(class_probabilities.values())
        }
    
    def rule_based_predict(self, news_text):
        text_lower = news_text.lower()
        
        # Simple keyword based rules
        positive_keywords = ['approval', 'etf', 'adoption', 'buy', 'rally', 'institution', 
                           'positive', 'bullish', 'green light', 'approved', 'partnership']
        negative_keywords = ['hack', 'attack', 'regulatory', 'ban', 'sell', 'drop', 'crash',
                           'negative', 'bearish', 'rejected', 'lawsuit', 'fraud']
        
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if positive_count > negative_count:
            direction = 'UP'
            confidence = min(60 + (positive_count * 5), 85)
        elif negative_count > positive_count:
            direction = 'DOWN'
            confidence = min(60 + (negative_count * 5), 85)
        else:
            direction = 'FLAT'
            confidence = 55
        
        return {
            'probabilities': {'UP': 33.3, 'DOWN': 33.3, 'FLAT': 33.3},
            'predicted_direction': direction,
            'confidence': confidence
        }
    
    def predict_with_explanation(self, news_text):
        result = self.predict(news_text)
        
        # Add explanatory text based on prediction
        direction = result['predicted_direction']
        confidence = result['confidence']
        
        if direction == 'UP':
            explanation = f"📈 The model predicts Bitcoin price will INCREASE with {confidence}% confidence"
            reasoning = "Positive news sentiment suggests potential buying pressure or positive market catalysts."
        elif direction == 'DOWN':
            explanation = f"📉 The model predicts Bitcoin price will DECREASE with {confidence}% confidence"
            reasoning = "Negative news sentiment suggests potential selling pressure or negative market factors."
        else:
            explanation = f"➡️ The model predicts Bitcoin price will REMAIN STABLE with {confidence}% confidence"
            reasoning = "Neutral news sentiment suggests market consolidation or balanced conditions."
        
        result['explanation'] = explanation
        result['reasoning'] = reasoning
        
        return result