"""
CLEAR-E Model Implementation
Energy-specific concept-aware lightweight adaptation for smart grid forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

class EnergySpecificConceptEncoder(nn.Module):
    """Energy-specific concept encoder integrating temporal and metadata information"""
    
    def __init__(self, input_dim: int, metadata_dim: int, concept_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.metadata_dim = metadata_dim
        self.concept_dim = concept_dim
        
        # Temporal pattern encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, concept_dim // 2)
        )
        
        # Feature importance learning
        self.feature_importance = nn.Parameter(torch.randn(metadata_dim))
        
        # Metadata encoder with attention
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, concept_dim // 2)
        )
        
        # Multi-head attention for metadata
        self.attention = nn.MultiheadAttention(
            embed_dim=metadata_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        # Concept fusion
        self.fusion = nn.Linear(concept_dim, concept_dim)
        
    def forward(self, temporal_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of concept encoder
        
        Args:
            temporal_input: [batch_size, seq_len, input_dim]
            metadata_input: [batch_size, seq_len, metadata_dim]
        
        Returns:
            concept_vector: [batch_size, concept_dim]
        """
        batch_size, seq_len, _ = temporal_input.shape
        
        # Temporal encoding
        temporal_flat = temporal_input.view(batch_size, -1)
        h_temporal = self.temporal_encoder(temporal_flat)
        
        # Feature importance weighting
        importance_weights = F.softmax(self.feature_importance, dim=0)
        weighted_metadata = metadata_input * importance_weights.unsqueeze(0).unsqueeze(0)
        
        # Attention mechanism for metadata
        attended_metadata, _ = self.attention(
            weighted_metadata, weighted_metadata, weighted_metadata
        )
        
        # Metadata encoding
        metadata_pooled = attended_metadata.mean(dim=1)  # Global average pooling
        h_metadata = self.metadata_encoder(metadata_pooled)
        
        # Concept fusion
        concept_vector = torch.cat([h_temporal, h_metadata], dim=1)
        concept_vector = self.fusion(concept_vector)
        
        return concept_vector

class LightweightAdaptationGenerator(nn.Module):
    """Lightweight adaptation generator using bottleneck architecture"""
    
    def __init__(self, concept_dim: int, target_layers: List[Tuple[str, int]], 
                 bottleneck_dim: int = 32):
        super().__init__()
        self.concept_dim = concept_dim
        self.target_layers = target_layers
        self.bottleneck_dim = bottleneck_dim
        
        # Adaptation generators for each target layer
        self.adaptation_generators = nn.ModuleDict()
        
        for layer_name, layer_dim in target_layers:
            self.adaptation_generators[layer_name] = nn.Sequential(
                nn.Linear(concept_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, layer_dim),
                nn.Tanh()  # Bounded adaptation
            )
    
    def forward(self, drift_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate adaptation parameters for target layers
        
        Args:
            drift_vector: [batch_size, concept_dim]
        
        Returns:
            adaptations: Dictionary of adaptation parameters
        """
        adaptations = {}
        
        for layer_name in self.adaptation_generators:
            adaptations[layer_name] = self.adaptation_generators[layer_name](drift_vector)
        
        return adaptations

class EnhancedDriftMemoryModule(nn.Module):
    """Enhanced drift memory module with statistical drift detection"""
    
    def __init__(self, concept_dim: int, memory_size: int = 10, momentum: float = 0.9):
        super().__init__()
        self.concept_dim = concept_dim
        self.memory_size = memory_size
        self.momentum = momentum
        
        # Exponential moving average
        self.register_buffer('ema_concept', torch.zeros(concept_dim))
        
        # Drift memory buffer
        self.register_buffer('drift_buffer', torch.zeros(memory_size, concept_dim))
        self.register_buffer('buffer_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('buffer_full', torch.zeros(1, dtype=torch.bool))
        
        # Smoothness regularization weight
        self.register_buffer('lambda_s_base', torch.tensor(0.02))
        self.register_buffer('gamma', torch.tensor(0.5))
        
    def forward(self, concept_vector: torch.Tensor) -> Tuple[torch.Tensor, bool, float]:
        """
        Process concept vector and detect drift
        
        Args:
            concept_vector: [batch_size, concept_dim]
        
        Returns:
            drift_vector: [batch_size, concept_dim]
            drift_detected: bool
            lambda_s: float (adaptive regularization weight)
        """
        batch_size = concept_vector.shape[0]
        
        # Update EMA
        if self.training:
            concept_mean = concept_vector.mean(dim=0)
            self.ema_concept = self.momentum * self.ema_concept + (1 - self.momentum) * concept_mean
        
        # Compute drift vector
        drift_vector = concept_vector - self.ema_concept.unsqueeze(0)
        
        # Update drift buffer
        if self.training:
            drift_mean = drift_vector.mean(dim=0)
            ptr = int(self.buffer_ptr)
            self.drift_buffer[ptr] = drift_mean
            self.buffer_ptr[0] = (ptr + 1) % self.memory_size
            if ptr == self.memory_size - 1:
                self.buffer_full[0] = True
        
        # Drift detection using simple magnitude threshold
        drift_detected = False
        lambda_s = float(self.lambda_s_base)
        
        if self.buffer_full:
            # Compute statistics
            valid_drifts = self.drift_buffer
            drift_mean = valid_drifts.mean(dim=0)
            drift_std = valid_drifts.std(dim=0).mean()
            
            # Simple threshold-based detection
            current_drift_magnitude = drift_vector.norm(dim=1).mean()
            threshold = drift_mean.norm() + 2 * drift_std
            
            if current_drift_magnitude > threshold:
                drift_detected = True
                lambda_s = float(self.gamma * self.lambda_s_base)
        
        return drift_vector, drift_detected, lambda_s

class EnergyAwareLoss(nn.Module):
    """Energy-aware asymmetric loss function"""
    
    def __init__(self, base_weight: float = 1.0, penalty_weight: float = 1.4):
        super().__init__()
        self.base_weight = base_weight
        self.penalty_weight = penalty_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute energy-aware loss
        
        Args:
            predictions: [batch_size, horizon, n_loads]
            targets: [batch_size, horizon, n_loads]
        
        Returns:
            loss: scalar tensor
        """
        # Base MSE loss
        mse_loss = F.mse_loss(predictions, targets)
        
        # Adaptive threshold (mean + std)
        target_mean = targets.mean()
        target_std = targets.std()
        threshold = target_mean + target_std
        
        # Asymmetric penalty for underestimation during high demand
        high_demand_mask = targets > threshold
        underestimation_mask = predictions < targets
        penalty_mask = high_demand_mask & underestimation_mask
        
        if penalty_mask.any():
            penalty_loss = F.mse_loss(
                predictions[penalty_mask], 
                targets[penalty_mask]
            )
            total_loss = self.base_weight * mse_loss + self.penalty_weight * penalty_loss
        else:
            total_loss = self.base_weight * mse_loss
        
        return total_loss

class CLEAR_E(nn.Module):
    """Complete CLEAR-E model for smart grid load forecasting"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.input_dim = config['input_dim']
        self.metadata_dim = config['metadata_dim']
        self.hidden_dim = config.get('hidden_dim', 128)
        self.concept_dim = config.get('concept_dim', 64)
        self.output_dim = config['output_dim']
        self.horizon = config['horizon']
        
        # Backbone network (frozen during adaptation)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        
        # Final prediction layers (adaptation targets)
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim * self.horizon)
        )
        
        # CLEAR-E components
        self.concept_encoder = EnergySpecificConceptEncoder(
            input_dim=self.input_dim,
            metadata_dim=self.metadata_dim,
            concept_dim=self.concept_dim
        )
        
        # Target layers for adaptation
        target_layers = [
            ('prediction_head.0', 64 * self.hidden_dim + 64),  # weight + bias
            ('prediction_head.2', (self.output_dim * self.horizon) * 64 + (self.output_dim * self.horizon))
        ]
        
        self.adaptation_generator = LightweightAdaptationGenerator(
            concept_dim=self.concept_dim,
            target_layers=target_layers,
            bottleneck_dim=config.get('bottleneck_dim', 32)
        )
        
        self.drift_memory = EnhancedDriftMemoryModule(
            concept_dim=self.concept_dim,
            memory_size=config.get('memory_size', 10),
            momentum=config.get('momentum', 0.9)
        )
        
        self.energy_loss = EnergyAwareLoss(
            penalty_weight=config.get('penalty_weight', 1.4)
        )
        
        # Training phase management
        self.frozen_phase = True
        self.phase_counter = 0
        self.frozen_phase_length = config.get('frozen_phase_length', 100)
        self.unfrozen_phase_length = config.get('unfrozen_phase_length', 20)
    
    def forward(self, temporal_input: torch.Tensor, metadata_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CLEAR-E
        
        Args:
            temporal_input: [batch_size, seq_len, input_dim]
            metadata_input: [batch_size, seq_len, metadata_dim]
        
        Returns:
            outputs: Dictionary containing predictions and intermediate results
        """
        batch_size = temporal_input.shape[0]
        
        # Concept encoding
        concept_vector = self.concept_encoder(temporal_input, metadata_input)
        
        # Drift computation and detection
        drift_vector, drift_detected, lambda_s = self.drift_memory(concept_vector)
        
        # Generate adaptations
        adaptations = self.adaptation_generator(drift_vector)
        
        # Backbone forward pass
        temporal_flat = temporal_input.view(batch_size, -1)
        backbone_output = self.backbone(temporal_flat)
        
        # Apply adaptations to prediction head (simplified)
        adapted_output = self.prediction_head(backbone_output)
        
        # Reshape to [batch_size, horizon, output_dim]
        predictions = adapted_output.view(batch_size, self.horizon, self.output_dim)
        
        return {
            'predictions': predictions,
            'concept_vector': concept_vector,
            'drift_vector': drift_vector,
            'drift_detected': drift_detected,
            'lambda_s': lambda_s,
            'adaptations': adaptations
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss including energy-aware and smoothness terms
        
        Args:
            outputs: Model outputs dictionary
            targets: [batch_size, horizon, output_dim]
        
        Returns:
            total_loss: scalar tensor
        """
        predictions = outputs['predictions']
        lambda_s = outputs['lambda_s']
        
        # Energy-aware loss
        energy_loss = self.energy_loss(predictions, targets)
        
        # Smoothness regularization (simplified)
        smoothness_loss = 0.0
        if hasattr(self.drift_memory, 'drift_buffer') and self.drift_memory.buffer_full:
            drift_diffs = torch.diff(self.drift_memory.drift_buffer, dim=0)
            smoothness_loss = lambda_s * torch.mean(drift_diffs.norm(dim=1))
        
        total_loss = energy_loss + smoothness_loss
        
        return total_loss
    
    def update_phase(self):
        """Update training phase (frozen/unfrozen)"""
        self.phase_counter += 1
        
        if self.frozen_phase and self.phase_counter >= self.frozen_phase_length:
            self.frozen_phase = False
            self.phase_counter = 0
            # Unfreeze final layers
            for param in self.prediction_head.parameters():
                param.requires_grad = True
        elif not self.frozen_phase and self.phase_counter >= self.unfrozen_phase_length:
            self.frozen_phase = True
            self.phase_counter = 0
            # Freeze backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def get_feature_importance(self) -> torch.Tensor:
        """Get learned feature importance weights"""
        return F.softmax(self.concept_encoder.feature_importance, dim=0)

# Example usage and configuration
def create_clear_e_model(dataset_config: Dict) -> CLEAR_E:
    """Create CLEAR-E model with appropriate configuration"""
    
    model_config = {
        'input_dim': dataset_config['lookback'] * dataset_config['n_loads'],
        'metadata_dim': dataset_config['n_metadata_features'],
        'output_dim': dataset_config['n_loads'],
        'horizon': dataset_config['horizon'],
        'hidden_dim': 128,
        'concept_dim': 64,
        'bottleneck_dim': 32,
        'memory_size': 10,
        'momentum': 0.9,
        'penalty_weight': 1.4,
        'frozen_phase_length': 100,
        'unfrozen_phase_length': 20
    }
    
    return CLEAR_E(model_config)
