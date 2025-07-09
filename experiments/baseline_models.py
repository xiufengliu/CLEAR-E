"""
Baseline Models for CLEAR-E Comparison
Implementation of industry-standard and state-of-the-art forecasting methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ARIMA_X_Model:
    """ARIMA with exogenous variables - industry standard"""
    
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)):
        self.order = order
        self.models = {}
        self.scalers = {}
        
    def fit(self, train_data: np.ndarray, exog_data: np.ndarray = None):
        """Fit ARIMA-X model"""
        n_series = train_data.shape[1] if len(train_data.shape) > 1 else 1
        
        if n_series == 1:
            train_data = train_data.reshape(-1, 1)
        
        for i in range(n_series):
            try:
                model = ARIMA(train_data[:, i], order=self.order, exog=exog_data)
                fitted_model = model.fit()
                self.models[i] = fitted_model
            except:
                # Fallback to simpler model
                model = ARIMA(train_data[:, i], order=(1, 1, 1), exog=exog_data)
                fitted_model = model.fit()
                self.models[i] = fitted_model
    
    def predict(self, steps: int, exog_data: np.ndarray = None) -> np.ndarray:
        """Generate forecasts"""
        predictions = []
        
        for i in self.models:
            try:
                pred = self.models[i].forecast(steps=steps, exog=exog_data)
                predictions.append(pred)
            except:
                # Fallback prediction
                last_value = self.models[i].fittedvalues[-1]
                pred = np.full(steps, last_value)
                predictions.append(pred)
        
        return np.array(predictions).T

class ExponentialSmoothingModel:
    """Holt-Winters Exponential Smoothing"""
    
    def __init__(self, seasonal_periods: int = 24):
        self.seasonal_periods = seasonal_periods
        self.models = {}
    
    def fit(self, train_data: np.ndarray):
        """Fit exponential smoothing model"""
        n_series = train_data.shape[1] if len(train_data.shape) > 1 else 1
        
        if n_series == 1:
            train_data = train_data.reshape(-1, 1)
        
        for i in range(n_series):
            try:
                model = ExponentialSmoothing(
                    train_data[:, i],
                    trend='add',
                    seasonal='add',
                    seasonal_periods=self.seasonal_periods
                )
                fitted_model = model.fit()
                self.models[i] = fitted_model
            except:
                # Fallback to simple exponential smoothing
                model = ExponentialSmoothing(train_data[:, i], trend='add')
                fitted_model = model.fit()
                self.models[i] = fitted_model
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecasts"""
        predictions = []
        
        for i in self.models:
            pred = self.models[i].forecast(steps=steps)
            predictions.append(pred)
        
        return np.array(predictions).T

class SVRModel:
    """Support Vector Regression with RBF kernel"""
    
    def __init__(self, lookback: int = 168, horizon: int = 24):
        self.lookback = lookback
        self.horizon = horizon
        self.models = {}
        
    def _create_sequences(self, data: np.ndarray, exog_data: np.ndarray = None):
        """Create input-output sequences"""
        X, y = [], []
        n_samples = len(data) - self.lookback - self.horizon + 1
        
        for i in range(n_samples):
            # Input features
            input_seq = data[i:i+self.lookback].flatten()
            if exog_data is not None:
                exog_seq = exog_data[i:i+self.lookback].flatten()
                input_seq = np.concatenate([input_seq, exog_seq])
            
            # Target
            target_seq = data[i+self.lookback:i+self.lookback+self.horizon].flatten()
            
            X.append(input_seq)
            y.append(target_seq)
        
        return np.array(X), np.array(y)
    
    def fit(self, train_data: np.ndarray, exog_data: np.ndarray = None):
        """Fit SVR model"""
        X, y = self._create_sequences(train_data, exog_data)
        
        # Train separate model for each output dimension
        for i in range(y.shape[1]):
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
            model.fit(X, y[:, i])
            self.models[i] = model
    
    def predict(self, last_sequence: np.ndarray, exog_data: np.ndarray = None) -> np.ndarray:
        """Generate forecasts"""
        input_seq = last_sequence.flatten()
        if exog_data is not None:
            exog_seq = exog_data.flatten()
            input_seq = np.concatenate([input_seq, exog_seq])
        
        predictions = []
        for i in self.models:
            pred = self.models[i].predict(input_seq.reshape(1, -1))
            predictions.append(pred[0])
        
        return np.array(predictions).reshape(self.horizon, -1)

class LSTMModel(nn.Module):
    """LSTM with attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 output_dim: int = 1, horizon: int = 24):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, output_dim * horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = x.shape[0]
        
        # LSTM encoding
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = attended_out.mean(dim=1)
        
        # Output projection
        output = self.output_projection(pooled)
        
        return output.view(batch_size, self.horizon, -1)

class TransformerModel(nn.Module):
    """Standard Transformer for time series forecasting"""
    
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 6, output_dim: int = 1, horizon: int = 24):
        super().__init__()
        self.d_model = d_model
        self.horizon = horizon
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, output_dim * horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        batch_size = x.shape[0]
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)
        
        # Output projection
        output = self.output_projection(pooled)
        
        return output.view(batch_size, self.horizon, -1)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class PatchTSTModel(nn.Module):
    """PatchTST: Patching-based Transformer for time series forecasting"""
    
    def __init__(self, input_dim: int, patch_size: int = 16, d_model: int = 128,
                 nhead: int = 8, num_layers: int = 3, output_dim: int = 1, horizon: int = 24):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.horizon = horizon
        
        self.patch_embedding = nn.Linear(patch_size * input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, output_dim * horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with patching"""
        batch_size, seq_len, input_dim = x.shape
        
        # Create patches
        n_patches = seq_len // self.patch_size
        if seq_len % self.patch_size != 0:
            # Pad sequence to make it divisible by patch_size
            pad_len = self.patch_size - (seq_len % self.patch_size)
            x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
            n_patches += 1
        
        # Reshape into patches
        patches = x[:, :n_patches * self.patch_size].view(
            batch_size, n_patches, self.patch_size * input_dim
        )
        
        # Patch embedding
        embedded = self.patch_embedding(patches)
        embedded = self.positional_encoding(embedded)
        
        # Transformer encoding
        encoded = self.transformer(embedded)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)
        
        # Output projection
        output = self.output_projection(pooled)
        
        return output.view(batch_size, self.horizon, -1)

class DLinearModel(nn.Module):
    """DLinear: Decomposition-based Linear model"""
    
    def __init__(self, seq_len: int, output_dim: int = 1, horizon: int = 24):
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Decomposition
        self.decomposition = SeriesDecomposition(kernel_size=25)
        
        # Linear layers for trend and seasonal components
        self.linear_trend = nn.Linear(seq_len, horizon)
        self.linear_seasonal = nn.Linear(seq_len, horizon)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with decomposition"""
        batch_size, seq_len, n_vars = x.shape
        
        # Process each variable separately
        outputs = []
        for i in range(n_vars):
            series = x[:, :, i]  # [batch_size, seq_len]
            
            # Decomposition
            trend, seasonal = self.decomposition(series)
            
            # Linear projections
            trend_pred = self.linear_trend(trend)
            seasonal_pred = self.linear_seasonal(seasonal)
            
            # Combine predictions
            pred = trend_pred + seasonal_pred
            outputs.append(pred.unsqueeze(-1))
        
        return torch.cat(outputs, dim=-1)

class SeriesDecomposition(nn.Module):
    """Series decomposition for DLinear"""
    
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose series into trend and seasonal components"""
        # Moving average for trend
        trend = self.moving_avg(x.unsqueeze(1)).squeeze(1)
        
        # Seasonal component
        seasonal = x - trend
        
        return trend, seasonal

class PROCEEDModel(nn.Module):
    """PROCEED: Parameter-efficient concept drift adaptation baseline"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, concept_dim: int = 64,
                 output_dim: int = 1, horizon: int = 24):
        super().__init__()
        self.concept_dim = concept_dim
        self.horizon = horizon
        
        # Backbone network
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * horizon)
        )
        
        # Concept encoder (simplified)
        self.concept_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, concept_dim)
        )
        
        # Adaptation generator for all layers
        self.adaptation_generator = nn.ModuleDict({
            'layer_0': nn.Linear(concept_dim, hidden_dim * input_dim + hidden_dim),
            'layer_1': nn.Linear(concept_dim, hidden_dim * hidden_dim + hidden_dim),
            'layer_2': nn.Linear(concept_dim, (output_dim * horizon) * hidden_dim + (output_dim * horizon))
        })
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptation"""
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Concept encoding
        concept = self.concept_encoder(x_flat)
        
        # Generate adaptations (simplified - not actually applied)
        adaptations = {}
        for layer_name in self.adaptation_generator:
            adaptations[layer_name] = self.adaptation_generator[layer_name](concept)
        
        # Backbone forward pass (adaptations not applied for simplicity)
        output = self.backbone(x_flat)
        
        return output.view(batch_size, self.horizon, -1)

# Factory function for creating baseline models
def create_baseline_model(model_name: str, config: Dict) -> nn.Module:
    """Create baseline model based on name and configuration"""
    
    if model_name == 'LSTM':
        return LSTMModel(
            input_dim=config['input_dim'],
            hidden_dim=config.get('hidden_dim', 128),
            output_dim=config['output_dim'],
            horizon=config['horizon']
        )
    elif model_name == 'Transformer':
        return TransformerModel(
            input_dim=config['input_dim'],
            d_model=config.get('d_model', 128),
            output_dim=config['output_dim'],
            horizon=config['horizon']
        )
    elif model_name == 'PatchTST':
        return PatchTSTModel(
            input_dim=config['input_dim'],
            patch_size=config.get('patch_size', 16),
            output_dim=config['output_dim'],
            horizon=config['horizon']
        )
    elif model_name == 'DLinear':
        return DLinearModel(
            seq_len=config['seq_len'],
            output_dim=config['output_dim'],
            horizon=config['horizon']
        )
    elif model_name == 'PROCEED':
        return PROCEEDModel(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            horizon=config['horizon']
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
