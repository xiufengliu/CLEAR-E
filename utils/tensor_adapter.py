
import torch
import torch.nn as nn

class TensorDimensionAdapter(nn.Module):
    """Universal adapter for tensor dimension mismatches"""
    
    def __init__(self, expected_channels, actual_channels):
        super().__init__()
        if expected_channels != actual_channels:
            if expected_channels < actual_channels:
                # Reduce channels
                self.adapter = nn.Conv1d(actual_channels, expected_channels, 1)
            else:
                # Increase channels
                self.adapter = nn.Conv1d(actual_channels, expected_channels, 1)
        else:
            self.adapter = nn.Identity()
    
    def forward(self, x):
        if x.dim() == 3:
            # (batch, seq, features) -> (batch, features, seq)
            x = x.transpose(1, 2)
            x = self.adapter(x)
            # (batch, features, seq) -> (batch, seq, features)
            x = x.transpose(1, 2)
        return x

def adapt_tensor_dimensions(tensor, expected_shape):
    """Dynamically adapt tensor to expected shape"""
    if tensor.shape == expected_shape:
        return tensor
    
    # Handle common dimension mismatches
    if len(tensor.shape) == 3 and len(expected_shape) == 3:
        batch, seq, feat = tensor.shape
        exp_batch, exp_seq, exp_feat = expected_shape
        
        # Adjust sequence length
        if seq != exp_seq:
            if seq < exp_seq:
                # Pad sequence
                padding = torch.zeros(batch, exp_seq - seq, feat, device=tensor.device)
                tensor = torch.cat([tensor, padding], dim=1)
            else:
                # Truncate sequence
                tensor = tensor[:, :exp_seq, :]
        
        # Adjust features
        if feat != exp_feat:
            adapter = TensorDimensionAdapter(exp_feat, feat)
            tensor = adapter(tensor)
    
    return tensor
