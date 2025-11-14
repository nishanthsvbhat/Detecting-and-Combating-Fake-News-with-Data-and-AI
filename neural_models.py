"""
Robust Deep Learning Models for Fake News Detection
Inspired by reference repo: ANN, CNN1D, BiLSTM architectures
Models trained on word embeddings (Word2Vec, 100D vectors)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


# ============ NEURAL NETWORK MODELS ============

class ANN(nn.Module):
    """
    Artificial Neural Network for classification
    Architecture: Input → Dense layers with LeakyReLU → Dropout → Output (Sigmoid)
    Reference: Achieves ~97% accuracy on ISOT dataset
    """
    def __init__(self, input_size: int = 100, hidden_dims: list = None, dropout_rate: float = 0.25):
        super(ANN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]
        
        # Build sequential layers
        layers = []
        prev_dim = input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network"""
        return self.network(x)
    
    def compute_l1_loss(self, w: torch.Tensor) -> torch.Tensor:
        """L1 regularization"""
        return torch.abs(w).sum()
    
    def compute_l2_loss(self, w: torch.Tensor) -> torch.Tensor:
        """L2 regularization"""
        return torch.pow(w, 2).sum()


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for sequence feature extraction
    Uses convolutional layers for temporal pattern detection
    """
    def __init__(self, input_size: int = 100, num_filters: int = 64, kernel_sizes: list = None):
        super(CNN1D, self).__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.pool = nn.MaxPool1d(2)
        
        # MLP head for classification
        mlp_input_size = num_filters * len(kernel_sizes) * (input_size // 2)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Input shape: (batch_size, seq_len) or (batch_size, 1, seq_len)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Apply conv layers
        conv_outputs = []
        for conv in self.conv_layers:
            out = self.activation(conv(x))
            out = self.pool(out)
            conv_outputs.append(out)
        
        # Concatenate conv outputs
        x = torch.cat(conv_outputs, dim=1)
        x = x.view(x.size(0), -1)  # Flatten
        
        # MLP classification
        return self.mlp(x)
    
    def compute_l1_loss(self, w: torch.Tensor) -> torch.Tensor:
        """L1 regularization"""
        return torch.abs(w).sum()
    
    def compute_l2_loss(self, w: torch.Tensor) -> torch.Tensor:
        """L2 regularization"""
        return torch.pow(w, 2).sum()


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence modeling
    Captures context from both directions for better representation
    Reference: Achieves ~96% accuracy on ISOT dataset
    """
    def __init__(self, input_size: int = 100, hidden_size: int = 64, 
                 num_layers: int = 2, bidirectional: bool = True):
        super(BiLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.25 if num_layers > 1 else 0
        )
        
        # Output layers
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, 1)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Input shape: (batch_size, seq_len, input_size)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension if missing
        
        # Initialize hidden states
        device = x.device
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(device)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take last output
        last_out = lstm_out[:, -1, :]
        
        # Classification
        out = self.fc(last_out)
        out = self.activation(out)
        out = self.output_activation(out)
        
        return out


# ============ DATASET UTILITIES ============

class TextDataset(Dataset):
    """Custom dataset for text embeddings and labels"""
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            labels: numpy array of shape (n_samples,) with binary labels (0 or 1)
        """
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)  # (n_samples, 1)
    
    def __len__(self) -> int:
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


# ============ TRAINING UTILITIES ============

def get_device() -> torch.device:
    """Get device (GPU if available, else CPU)"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate binary classification accuracy"""
    y_pred_binary = (y_pred > 0.5).float()
    return (y_true == y_pred_binary).sum().item() / len(y_true)


def train_epoch(model: nn.Module, train_loader: DataLoader, 
                optimizer: optim.Optimizer, loss_fn: nn.Module, 
                device: torch.device) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        output = model(batch_x)
        loss = loss_fn(output, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += accuracy(batch_y, output)
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    
    return avg_loss, avg_acc


def validate_epoch(model: nn.Module, val_loader: DataLoader, 
                   loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            
            total_loss += loss.item()
            total_acc += accuracy(batch_y, output)
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = total_acc / len(val_loader)
    
    return avg_loss, avg_acc


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int = 100, learning_rate: float = 3e-4, 
                device: Optional[torch.device] = None, verbose: bool = True) -> dict:
    """
    Complete training loop
    Reference: Uses Adam optimizer with lr=3e-4, BCELoss, 300 epochs
    """
    if device is None:
        device = get_device()
    
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc*100:.2f}%")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


# ============ INFERENCE UTILITIES ============

def predict(model: nn.Module, embeddings: torch.Tensor, 
            device: Optional[torch.device] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict on batches of embeddings
    Returns: (predictions, confidences)
    """
    if device is None:
        device = get_device()
    
    model.to(device)
    model.eval()
    
    embeddings = embeddings.to(device) if isinstance(embeddings, torch.Tensor) else torch.FloatTensor(embeddings).to(device)
    
    with torch.no_grad():
        outputs = model(embeddings)
    
    confidences = outputs.cpu().numpy().flatten()
    predictions = (confidences > 0.5).astype(int)
    
    return predictions, confidences
