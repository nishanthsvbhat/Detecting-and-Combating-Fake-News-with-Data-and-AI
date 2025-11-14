"""
Word2Vec Embeddings Module
Generates 100-dimensional word embeddings for text vectorization
Inspired by reference repo approach using gensim
"""

import numpy as np
from typing import List, Tuple, Optional
import os
import pickle

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False


class Word2VecEmbedder:
    """
    Word2Vec embedding generator and manager
    Creates and manages word embeddings for text data
    Reference: 100-dimensional vectors trained on news corpus
    """
    
    def __init__(self, embedding_dim: int = 100, window: int = 5, 
                 min_count: int = 1, sg: int = 1):
        """
        Args:
            embedding_dim: Dimensionality of embeddings (default: 100)
            window: Context window size (default: 5)
            min_count: Minimum word frequency (default: 1)
            sg: Skip-gram (1) or CBOW (0) (default: 1 for better quality)
        """
        if not GENSIM_AVAILABLE:
            raise ImportError("gensim is required for Word2Vec. Install via: pip install gensim")
        
        self.embedding_dim = embedding_dim
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.model = None
        self.vocab_size = 0
    
    def train(self, tokenized_texts: List[List[str]], epochs: int = 5, workers: int = 4) -> None:
        """
        Train Word2Vec model on tokenized texts
        
        Args:
            tokenized_texts: List of token lists (one list per document)
            epochs: Number of training epochs
            workers: Number of parallel processes
        """
        print(f"Training Word2Vec model on {len(tokenized_texts)} documents...")
        
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=epochs,
            workers=workers,
            seed=42
        )
        
        self.vocab_size = len(self.model.wv)
        print(f"✓ Model trained with vocabulary size: {self.vocab_size}")
    
    def vectorize_text(self, tokens: List[str]) -> Optional[np.ndarray]:
        """
        Convert tokenized text to embedding vector
        Uses mean pooling of word vectors
        
        Args:
            tokens: List of tokens
        
        Returns:
            Embedding vector (1D numpy array) or None if no valid words
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get vectors for words in vocabulary
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])
        
        if not vectors:
            # Return zero vector if no words found
            return np.zeros(self.embedding_dim)
        
        # Mean pooling
        return np.mean(vectors, axis=0)
    
    def vectorize_batch(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        Vectorize batch of tokenized texts
        
        Args:
            tokenized_texts: List of token lists
        
        Returns:
            Array of shape (n_texts, embedding_dim)
        """
        embeddings = []
        for tokens in tokenized_texts:
            embedding = self.vectorize_text(tokens)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.model.save(filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        self.model = Word2Vec.load(filepath)
        self.vocab_size = len(self.model.wv)
        self.embedding_dim = self.model.vector_size
        print(f"✓ Model loaded from {filepath} (vocab size: {self.vocab_size})")
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get vector for a single word"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        if word in self.model.wv:
            return self.model.wv[word]
        else:
            return None
    
    def most_similar(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        try:
            return self.model.wv.most_similar(word, topn=topn)
        except KeyError:
            return []
    
    def similarity(self, word1: str, word2: str) -> Optional[float]:
        """Calculate similarity between two words"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        try:
            return self.model.wv.similarity(word1, word2)
        except KeyError:
            return None


# Convenience functions
def create_embedder(embedding_dim: int = 100) -> Word2VecEmbedder:
    """Create new Word2Vec embedder"""
    return Word2VecEmbedder(embedding_dim=embedding_dim)


def load_embedder(filepath: str) -> Word2VecEmbedder:
    """Load Word2Vec embedder from disk"""
    embedder = Word2VecEmbedder()
    embedder.load_model(filepath)
    return embedder
