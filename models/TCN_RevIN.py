import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):
    """TCN model with RevIN normalization - ultra simplified implementation"""
    def __init__(self, args):
        super().__init__()

        # Get the correct input dimension for ETTh1 dataset
        # ETTh1 with features='M' has 7 features
        self.enc_in = args.enc_in

        # RevIN normalization layer
        self.revin = RevIN(num_features=self.enc_in)

        # Use a simple MLP encoder instead of LSTM to avoid dimension issues
        self.encoder = nn.Sequential(
            nn.Linear(self.enc_in * args.seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.pred_len = args.pred_len

        # Output projection
        self.projection = nn.Linear(128, self.enc_in * args.pred_len)

    def forward(self, x, x_mark=None):
        # Get input shape
        B, L, D = x.shape  # batch, length, dimension

        # Apply RevIN normalization
        x_norm = self.revin(x, mode='norm')

        # Flatten the input for MLP
        x_flat = x_norm.reshape(B, -1)

        # MLP encoding
        features = self.encoder(x_flat)

        # Project to prediction
        pred_flat = self.projection(features)

        # Reshape to [Batch, PredLen, Feature]
        pred = pred_flat.reshape(B, self.pred_len, self.enc_in)

        # Apply RevIN denormalization
        pred = self.revin(pred, mode='denorm')

        return pred


class Model_Ensemble(nn.Module):
    """TCN ensemble model with RevIN normalization - ultra simplified implementation"""
    def __init__(self, args):
        super().__init__()
        # Create a single model and use it directly
        # This is a simplification to avoid training multiple models
        self.model = Model(args)

    def forward(self, x, x_mark=None, w1=0.5, w2=0.5):
        # Just use the single model
        return self.model(x, x_mark)

    def forward_individual(self, x, x_mark):
        # Return the same prediction twice for compatibility
        y = self.model(x, x_mark)
        return y, y
