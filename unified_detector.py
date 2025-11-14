"""
Unified Inference System for Fake News Detection
Supports multiple ML backends: PassiveAggressive + Neural Models (ANN/CNN1D/BiLSTM)
Includes ensemble voting for higher accuracy
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import os

from enhanced_preprocessing import preprocess_full
from neural_models import ANN, CNN1D, BiLSTM, get_device


class UnifiedFakeNewsDetector:
    """
    Multi-model inference engine supporting:
    - PassiveAggressive (baseline, fast)
    - ANN (dense neural network)
    - CNN1D (convolutional features)
    - BiLSTM (sequence modeling)
    - Ensemble voting (combined predictions)
    """
    
    def __init__(self, embedder=None, use_neural: bool = True, 
                 model_dir: str = 'model_artifacts'):
        """
        Args:
            embedder: Word2VecEmbedder instance (required for neural models)
            use_neural: Use neural models if available
            model_dir: Directory containing saved models
        """
        self.embedder = embedder
        self.use_neural = use_neural
        self.model_dir = model_dir
        self.device = get_device()
        
        self.preprocessor = None
        self.pa_model = None
        self.pa_vectorizer = None
        
        self.neural_models = {}
        self.model_weights = {
            'ANN': 0.4,
            'CNN1D': 0.3,
            'BiLSTM': 0.3
        }
        
        self._load_neural_models()
    
    def _load_neural_models(self) -> None:
        """Load trained neural models from disk"""
        if not self.use_neural:
            return
        
        for model_name in ['ANN', 'CNN1D', 'BiLSTM']:
            model_path = os.path.join(self.model_dir, f'{model_name}_best_model.pth')
            
            if not os.path.exists(model_path):
                print(f"⚠ {model_name} model not found at {model_path}")
                continue
            
            try:
                if model_name == 'ANN':
                    model = ANN(input_size=100)
                elif model_name == 'CNN1D':
                    model = CNN1D(input_size=100)
                elif model_name == 'BiLSTM':
                    model = BiLSTM(input_size=100)
                
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                
                self.neural_models[model_name] = model
                print(f"✓ Loaded {model_name} from {model_path}")
            except Exception as e:
                print(f"✗ Error loading {model_name}: {e}")
    
    def set_pa_model(self, vectorizer, classifier) -> None:
        """Set PassiveAggressive model"""
        self.pa_vectorizer = vectorizer
        self.pa_model = classifier
    
    def preprocess_text(self, text: str) -> Tuple[list, str]:
        """
        Preprocess text to tokens and cleaned string
        
        Returns:
            (tokens, preprocessed_string)
        """
        tokens = preprocess_full(text, apply_stem=True, apply_lemma=False, aggressive=False)
        preprocessed_str = ' '.join(tokens)
        return tokens, preprocessed_str
    
    def predict_pa(self, text: str) -> Dict:
        """
        Predict using PassiveAggressive classifier
        
        Returns:
            {
                'model': 'PassiveAggressive',
                'prediction': 'REAL'/'FAKE',
                'confidence': 0-100,
                'score': raw decision score
            }
        """
        if self.pa_model is None or self.pa_vectorizer is None:
            return None
        
        try:
            _, preprocessed = self.preprocess_text(text)
            vector = self.pa_vectorizer.transform([preprocessed])
            
            prediction = self.pa_model.predict(vector)[0]
            score = self.pa_model.decision_function(vector)[0]
            confidence = min(100, max(10, int(abs(score) * 30)))
            
            return {
                'model': 'PassiveAggressive',
                'prediction': prediction,
                'confidence': confidence,
                'score': float(score)
            }
        except Exception as e:
            print(f"PA prediction error: {e}")
            return None
    
    def predict_neural(self, text: str) -> Dict[str, Dict]:
        """
        Predict using neural models
        
        Returns:
            {
                'ANN': {'prediction': ..., 'confidence': ...},
                'CNN1D': {...},
                'BiLSTM': {...}
            }
        """
        if self.embedder is None:
            print("Embedder not available for neural predictions")
            return {}
        
        try:
            tokens, _ = self.preprocess_text(text)
            embedding = self.embedder.vectorize_text(tokens)
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
            
            results = {}
            
            for model_name, model in self.neural_models.items():
                with torch.no_grad():
                    output = model(embedding_tensor)
                
                confidence = float(output.squeeze().cpu().numpy())
                prediction = 'REAL' if confidence > 0.5 else 'FAKE'
                
                results[model_name] = {
                    'prediction': prediction,
                    'confidence': int(confidence * 100),
                    'raw_score': confidence
                }
            
            return results
        except Exception as e:
            print(f"Neural prediction error: {e}")
            return {}
    
    def predict_ensemble(self, text: str, use_pa: bool = True, 
                        use_neural: bool = True) -> Dict:
        """
        Ensemble prediction combining all available models
        Weighted voting with confidence aggregation
        
        Args:
            text: News text to classify
            use_pa: Use PassiveAggressive model
            use_neural: Use neural models
        
        Returns:
            {
                'final_verdict': 'REAL'/'FAKE',
                'confidence': 0-100,
                'models_used': [...],
                'model_predictions': {...},
                'reasoning': '...'
            }
        """
        predictions = []
        weights = []
        model_details = {}
        
        # PA prediction
        if use_pa:
            pa_pred = self.predict_pa(text)
            if pa_pred:
                pred_label = 1 if pa_pred['prediction'] == 'REAL' else 0
                predictions.append(pred_label)
                weights.append(0.2)
                model_details['PassiveAggressive'] = pa_pred
        
        # Neural predictions
        if use_neural:
            neural_preds = self.predict_neural(text)
            for model_name, pred in neural_preds.items():
                pred_label = 1 if pred['prediction'] == 'REAL' else 0
                predictions.append(pred_label)
                weights.append(self.model_weights.get(model_name, 0.3))
                model_details[model_name] = pred
        
        if not predictions:
            return {
                'final_verdict': 'UNVERIFIABLE',
                'confidence': 0,
                'models_used': [],
                'model_predictions': {},
                'reasoning': 'No models available for prediction'
            }
        
        # Weighted voting
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        weighted_score = np.sum(predictions * weights)
        final_pred = 'REAL' if weighted_score > 0.5 else 'FAKE'
        confidence = int(abs(weighted_score - 0.5) * 2 * 100)
        
        return {
            'final_verdict': final_pred,
            'confidence': confidence,
            'weighted_score': float(weighted_score),
            'models_used': list(model_details.keys()),
            'model_predictions': model_details,
            'reasoning': f'Ensemble vote: {list(model_details.keys())} agree on {final_pred}'
        }
    
    def predict_with_confidence(self, text: str) -> Dict:
        """
        High-level prediction interface
        
        Returns comprehensive analysis with all available methods
        """
        return self.predict_ensemble(text, use_pa=True, use_neural=True)


# Convenience function
def create_detector(embedder=None, use_neural: bool = True,
                   model_dir: str = 'model_artifacts') -> UnifiedFakeNewsDetector:
    """Create a unified detector instance"""
    return UnifiedFakeNewsDetector(embedder=embedder, use_neural=use_neural, 
                                   model_dir=model_dir)
