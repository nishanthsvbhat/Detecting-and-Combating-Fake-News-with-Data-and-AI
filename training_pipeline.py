"""
Complete Training Pipeline for Fake News Detection
Trains Word2Vec embeddings + Neural models on labeled news data
Reference: ISOT dataset with 12K+ articles per category
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import os
import json

from enhanced_preprocessing import EnhancedPreprocessor, preprocess_full
from word2vec_embedder import Word2VecEmbedder
from neural_models import ANN, CNN1D, BiLSTM, TextDataset, train_model, get_device


class FakeNewsTrainingPipeline:
    """
    Complete training pipeline:
    1. Load and preprocess data
    2. Train Word2Vec embeddings
    3. Train neural models
    4. Evaluate and save models
    """
    
    def __init__(self, output_dir: str = 'model_artifacts'):
        """
        Args:
            output_dir: Directory to save models and artifacts
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.preprocessor = EnhancedPreprocessor()
        self.embedder = Word2VecEmbedder(embedding_dim=100)
        self.device = get_device()
        
        self.train_history = {}
        self.models = {}
    
    def load_dataset(self, true_csv: str, fake_csv: str, 
                    sample_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ISOT dataset (True.csv and Fake.csv)
        
        Args:
            true_csv: Path to True.csv
            fake_csv: Path to Fake.csv
            sample_size: Limit samples per category (None = use all)
        
        Returns:
            (texts, labels) where labels are 1 for real, 0 for fake
        """
        print("Loading dataset...")
        
        # Load real news
        true_df = pd.read_csv(true_csv)
        true_texts = true_df['text'].tolist()
        if sample_size:
            true_texts = true_texts[:sample_size]
        
        # Load fake news
        fake_df = pd.read_csv(fake_csv)
        fake_texts = fake_df['text'].tolist()
        if sample_size:
            fake_texts = fake_texts[:sample_size]
        
        # Combine
        texts = true_texts + fake_texts
        labels = np.array([1] * len(true_texts) + [0] * len(fake_texts))
        
        print(f"✓ Loaded {len(texts)} texts ({len(true_texts)} real, {len(fake_texts)} fake)")
        
        return texts, labels
    
    def preprocess_texts(self, texts: list, verbose: bool = True) -> list:
        """
        Preprocess all texts to tokenized form
        
        Args:
            texts: List of raw text strings
            verbose: Print progress
        
        Returns:
            List of tokenized texts
        """
        if verbose:
            print(f"Preprocessing {len(texts)} texts...")
        
        tokenized_texts = []
        for i, text in enumerate(texts):
            if verbose and (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(texts)}")
            
            tokens = preprocess_full(text, apply_stem=True, apply_lemma=False, aggressive=False)
            tokenized_texts.append(tokens)
        
        if verbose:
            print(f"✓ Preprocessing complete")
        
        return tokenized_texts
    
    def train_embeddings(self, tokenized_texts: list, epochs: int = 5) -> None:
        """
        Train Word2Vec embeddings
        
        Args:
            tokenized_texts: List of tokenized texts
            epochs: Training epochs
        """
        print(f"Training Word2Vec embeddings...")
        self.embedder.train(tokenized_texts, epochs=epochs)
        
        # Save embedder
        embedder_path = os.path.join(self.output_dir, 'word2vec_model')
        self.embedder.save_model(embedder_path)
    
    def create_embeddings(self, tokenized_texts: list) -> np.ndarray:
        """
        Create embeddings from tokenized texts
        
        Args:
            tokenized_texts: List of tokenized texts
        
        Returns:
            Array of embeddings (n_texts, 100)
        """
        print(f"Creating embeddings for {len(tokenized_texts)} texts...")
        embeddings = self.embedder.vectorize_batch(tokenized_texts)
        return embeddings
    
    def train_neural_model(self, embeddings: np.ndarray, labels: np.ndarray,
                          model_name: str = 'ANN', epochs: int = 50,
                          batch_size: int = 32, split_ratio: float = 0.7) -> dict:
        """
        Train a neural model
        
        Args:
            embeddings: Array of embeddings (n_texts, 100)
            labels: Array of binary labels (n_texts,)
            model_name: 'ANN', 'CNN1D', or 'BiLSTM'
            epochs: Training epochs
            batch_size: Batch size
            split_ratio: Train/val split ratio
        
        Returns:
            Training history
        """
        print(f"\nTraining {model_name} model...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings, labels, train_size=split_ratio, random_state=42
        )
        
        # Create datasets
        train_dataset = TextDataset(X_train, y_train)
        val_dataset = TextDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        if model_name == 'ANN':
            model = ANN(input_size=embeddings.shape[1])
        elif model_name == 'CNN1D':
            model = CNN1D(input_size=embeddings.shape[1])
        elif model_name == 'BiLSTM':
            model = BiLSTM(input_size=embeddings.shape[1])
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train
        history = train_model(
            model, train_loader, val_loader,
            epochs=epochs,
            learning_rate=3e-4,
            device=self.device,
            verbose=True
        )
        
        # Save model
        model_path = os.path.join(self.output_dir, f'{model_name}_best_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Store
        self.models[model_name] = model
        self.train_history[model_name] = history
        
        return history
    
    def train_all_models(self, embeddings: np.ndarray, labels: np.ndarray,
                        epochs: int = 50, batch_size: int = 32) -> dict:
        """
        Train all models (ANN, CNN1D, BiLSTM)
        
        Args:
            embeddings: Array of embeddings
            labels: Array of labels
            epochs: Training epochs
            batch_size: Batch size
        
        Returns:
            Dictionary of all training histories
        """
        results = {}
        
        for model_name in ['ANN', 'CNN1D', 'BiLSTM']:
            try:
                history = self.train_neural_model(
                    embeddings, labels,
                    model_name=model_name,
                    epochs=epochs,
                    batch_size=batch_size
                )
                results[model_name] = history
            except Exception as e:
                print(f"✗ Error training {model_name}: {e}")
        
        return results
    
    def evaluate_models(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Evaluate all trained models"""
        results = {}
        
        embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
        labels_tensor = torch.FloatTensor(labels).to(self.device)
        
        for model_name, model in self.models.items():
            model.to(self.device)
            model.eval()
            
            with torch.no_grad():
                outputs = model(embeddings_tensor)
            
            predictions = (outputs > 0.5).float().squeeze().cpu().numpy()
            accuracy = np.mean(predictions == labels)
            
            results[model_name] = {
                'accuracy': float(accuracy),
                'correct': int(np.sum(predictions == labels)),
                'total': len(labels)
            }
            
            print(f"{model_name}: {accuracy*100:.2f}% accuracy")
        
        return results
    
    def save_pipeline(self, config_path: Optional[str] = None) -> None:
        """Save pipeline configuration"""
        if config_path is None:
            config_path = os.path.join(self.output_dir, 'pipeline_config.json')
        
        config = {
            'embedding_dim': self.embedder.embedding_dim,
            'models_trained': list(self.models.keys()),
            'device': str(self.device),
            'vocab_size': self.embedder.vocab_size
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Config saved to {config_path}")


# Convenience function
def run_full_pipeline(true_csv: str, fake_csv: str, 
                      epochs: int = 50, batch_size: int = 32,
                      sample_size: Optional[int] = None) -> FakeNewsTrainingPipeline:
    """
    Run complete training pipeline
    
    Usage:
        pipeline = run_full_pipeline('True.csv', 'Fake.csv', epochs=50)
    """
    pipeline = FakeNewsTrainingPipeline()
    
    # Load data
    texts, labels = pipeline.load_dataset(true_csv, fake_csv, sample_size)
    
    # Preprocess
    tokenized_texts = pipeline.preprocess_texts(texts)
    
    # Train embeddings
    pipeline.train_embeddings(tokenized_texts, epochs=5)
    
    # Create embeddings
    embeddings = pipeline.create_embeddings(tokenized_texts)
    
    # Train models
    pipeline.train_all_models(embeddings, labels, epochs=epochs, batch_size=batch_size)
    
    # Evaluate
    pipeline.evaluate_models(embeddings, labels)
    
    # Save
    pipeline.save_pipeline()
    
    return pipeline


if __name__ == "__main__":
    # Example usage
    pipeline = run_full_pipeline(
        'True.csv', 'Fake.csv',
        epochs=50, batch_size=32,
        sample_size=1000  # Limit for testing
    )
