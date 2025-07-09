import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
from collections import deque
from typing import Optional, Dict, Any

from adapter.proceed import Proceed, add_adapters_
from adapter.module.generator import AdaptGenerator, Bottleneck
from adapter.module.down_up import Down_Up
from adapter.module.base import Adaptation
import transformers


class Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class EnergyMetadataEncoder(nn.Module):
    """
    Enhanced metadata encoder with attention and feature importance learning
    """
    def __init__(self, metadata_dim: int, hidden_dim: int, output_dim: int,
                 use_attention: bool = True, feature_groups: list = None):
        super().__init__()
        self.metadata_dim = metadata_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        self.feature_groups = feature_groups or ['weather', 'temporal', 'calendar']

        # Feature importance learning
        self.feature_importance = nn.Parameter(torch.ones(metadata_dim))

        # Main encoder
        self.encoder = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        # Attention mechanism for temporal metadata
        if use_attention:
            # Ensure num_heads divides embed_dim evenly
            possible_heads = [1, 2, 4, 5, 8, 10]
            num_heads = 1
            for h in possible_heads:
                if metadata_dim % h == 0 and h <= metadata_dim:
                    num_heads = h

            self.attention = nn.MultiheadAttention(
                embed_dim=metadata_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(metadata_dim)

        # Feature group encoders for specialized processing
        self.group_encoders = nn.ModuleDict()
        self.group_projection = None

        if len(self.feature_groups) > 1:
            group_size = metadata_dim // len(self.feature_groups)
            total_group_output = 0

            for i, group in enumerate(self.feature_groups):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < len(self.feature_groups) - 1 else metadata_dim
                group_dim = end_idx - start_idx
                group_output_dim = output_dim // len(self.feature_groups)
                total_group_output += group_output_dim

                self.group_encoders[group] = nn.Sequential(
                    nn.Linear(group_dim, hidden_dim // len(self.feature_groups)),
                    nn.GELU(),
                    nn.Linear(hidden_dim // len(self.feature_groups), group_output_dim)
                )

            # Create projection layer to ensure exact output dimension
            if total_group_output != output_dim:
                self.group_projection = nn.Linear(total_group_output, output_dim)

    def forward(self, metadata):
        """
        Enhanced forward pass with attention and feature importance
        """
        original_shape = metadata.shape

        # Apply feature importance weighting
        importance_weights = torch.softmax(self.feature_importance, dim=0)

        if metadata.dim() == 3:
            # [batch_size, seq_len, metadata_dim]
            batch_size, seq_len, _ = metadata.shape

            # Apply feature importance
            metadata_weighted = metadata * importance_weights.unsqueeze(0).unsqueeze(0)

            if self.use_attention and seq_len > 1:
                # Apply self-attention over temporal dimension
                attended, _ = self.attention(metadata_weighted, metadata_weighted, metadata_weighted)
                attended = self.attention_norm(attended + metadata_weighted)
                # Aggregate over time
                metadata_agg = attended.mean(dim=1)
            else:
                metadata_agg = metadata_weighted.mean(dim=1)
        else:
            # [batch_size, metadata_dim]
            metadata_agg = metadata * importance_weights.unsqueeze(0)

        # Group-based encoding if multiple groups
        if len(self.group_encoders) > 1:
            group_outputs = []
            group_size = self.metadata_dim // len(self.feature_groups)

            for i, (group_name, encoder) in enumerate(self.group_encoders.items()):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < len(self.feature_groups) - 1 else self.metadata_dim
                group_features = metadata_agg[:, start_idx:end_idx]
                group_output = encoder(group_features)
                group_outputs.append(group_output)

            # Combine group outputs and ensure correct output dimension
            combined_output = torch.cat(group_outputs, dim=1)

            # Apply projection if needed to match exact output dimension
            if self.group_projection is not None:
                return self.group_projection(combined_output)
            else:
                return combined_output
        else:
            # Standard encoding
            return self.encoder(metadata_agg)

    def get_feature_importance(self):
        """Get current feature importance weights"""
        return torch.softmax(self.feature_importance, dim=0)


class LightweightAdaptGenerator(nn.Module):
    """
    Lightweight adaptation generator that only affects final output layers
    """
    def __init__(self, backbone: nn.Module, concept_features: int,
                 activation=nn.LeakyReLU, mid_dim: int = None, target_layers: list = None):
        super().__init__()
        self.concept_features = concept_features
        self.target_layers = target_layers or ['projection', 'head', 'output', 'fc']

        # Use same structure as PROCEED's AdaptGenerator but filter for target layers
        self.dim_name_dict = collections.defaultdict(list)
        self.bottlenecks = nn.ModuleDict()

        # Find target layers and build dim_name_dict like PROCEED
        for name, module in backbone.named_modules():
            if isinstance(module, Adaptation):
                layer_name = name.split('.')[-1].lower()
                # Only include target layers
                if any(target in layer_name for target in self.target_layers):
                    _weight = module.affine_weight if hasattr(module, 'affine_weight') else module.weight
                    out_features = _weight.shape[1 if isinstance(module, transformers.Conv1D) else 0]
                    if _weight.dim() == 1:
                        in_features = 1
                    else:
                        in_features = _weight.shape[0 if isinstance(module, transformers.Conv1D) else 1]
                    out_dim = out_features
                    if module.bias is not None:
                        out_dim += out_features
                    if isinstance(module, Down_Up):
                        out_dim += in_features
                    # Use same key format as PROCEED
                    key = name.split('.')[-1] + '_' + str(out_dim)
                    self.dim_name_dict[key].append(name)

        # Create bottlenecks for target layers only
        for key, names in self.dim_name_dict.items():
            out_dim = int(key.split('_')[-1])
            _hid_dims = min(mid_dim or 64, out_dim // 4)
            self.bottlenecks[key] = Bottleneck(
                concept_features, out_dim, len(names), _hid_dims, activation,
                shared=True, need_bias=True
            )

    def forward(self, concept_vector):
        """Generate adaptations only for target layers"""
        adaptations = {k: bottleneck(concept_vector) for k, bottleneck in self.bottlenecks.items()}
        return adaptations


class DriftMemoryModule(nn.Module):
    """
    Enhanced drift memory module with adaptive features
    """
    def __init__(self, concept_dim: int, memory_size: int = 10, reg_weight: float = 0.1,
                 adaptive_memory: bool = True, drift_threshold: float = 0.5):
        super().__init__()
        self.concept_dim = concept_dim
        self.memory_size = memory_size
        self.reg_weight = reg_weight
        self.adaptive_memory = adaptive_memory
        self.drift_threshold = drift_threshold

        # Memory buffers
        self.register_buffer('memory_tensor', torch.zeros(memory_size, concept_dim))
        self.register_buffer('memory_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('memory_full', torch.tensor(False, dtype=torch.bool))

        # Drift detection
        self.register_buffer('drift_magnitude_history', torch.zeros(memory_size))
        self.register_buffer('drift_detected', torch.tensor(False, dtype=torch.bool))

        # Adaptive weights
        if adaptive_memory:
            self.weight_predictor = nn.Sequential(
                nn.Linear(concept_dim, concept_dim // 2),
                nn.ReLU(),
                nn.Linear(concept_dim // 2, 1),
                nn.Sigmoid()
            )

    def update_memory(self, drift_vector):
        """Add new drift vector to memory with drift detection"""
        if drift_vector.dim() > 1:
            drift_vector = drift_vector.mean(0)

        # Calculate drift magnitude
        drift_magnitude = torch.norm(drift_vector, p=2)

        # Update memory
        ptr = self.memory_ptr.item()
        self.memory_tensor[ptr] = drift_vector.detach()
        self.drift_magnitude_history[ptr] = drift_magnitude.detach()
        self.memory_ptr = (self.memory_ptr + 1) % self.memory_size

        if ptr == self.memory_size - 1:
            self.memory_full = torch.tensor(True, dtype=torch.bool)

        # Detect significant drift
        if self.memory_full or self.memory_ptr > 3:
            valid_magnitudes = self.drift_magnitude_history[:self.memory_ptr] if not self.memory_full else self.drift_magnitude_history
            avg_magnitude = valid_magnitudes.mean()
            self.drift_detected = drift_magnitude > avg_magnitude + self.drift_threshold

    def get_smoothness_loss(self, current_drift):
        """Enhanced smoothness loss with adaptive weighting"""
        if not self.memory_full and self.memory_ptr == 0:
            return torch.tensor(0.0, device=current_drift.device)

        if current_drift.dim() > 1:
            current_drift = current_drift.mean(0)

        # Get valid memory entries
        if self.memory_full:
            valid_memory = self.memory_tensor
        else:
            valid_memory = self.memory_tensor[:self.memory_ptr]

        if len(valid_memory) == 0:
            return torch.tensor(0.0, device=current_drift.device)

        # Compute smoothness loss
        recent_drift = valid_memory[-1]
        diff = current_drift - recent_drift
        smoothness_loss = torch.norm(diff, p=2)

        # Adaptive weighting based on drift detection
        if self.adaptive_memory and hasattr(self, 'weight_predictor'):
            weight = self.weight_predictor(current_drift.unsqueeze(0)).squeeze()
            # Reduce regularization during significant drift
            if self.drift_detected:
                weight = weight * 0.5
        else:
            weight = torch.tensor(1.0, device=current_drift.device)

        return self.reg_weight * weight * smoothness_loss

    def get_drift_statistics(self):
        """Get drift statistics for monitoring"""
        if not self.memory_full and self.memory_ptr == 0:
            return {}

        valid_magnitudes = self.drift_magnitude_history[:self.memory_ptr] if not self.memory_full else self.drift_magnitude_history

        return {
            'avg_drift_magnitude': valid_magnitudes.mean().item(),
            'max_drift_magnitude': valid_magnitudes.max().item(),
            'drift_detected': self.drift_detected.item(),
            'memory_utilization': (self.memory_ptr.item() if not self.memory_full else self.memory_size) / self.memory_size
        }


class EnergyAwareLoss(nn.Module):
    """
    Enhanced asymmetric loss with adaptive penalties and load-aware weighting
    """
    def __init__(self, base_criterion=nn.MSELoss(), high_load_threshold: float = 0.8,
                 underestimate_penalty: float = 1.5, adaptive_penalty: bool = True,
                 load_aware_weighting: bool = True):
        super().__init__()
        self.base_criterion = base_criterion
        self.high_load_threshold = high_load_threshold
        self.underestimate_penalty = underestimate_penalty
        self.adaptive_penalty = adaptive_penalty
        self.load_aware_weighting = load_aware_weighting

        # Running statistics for adaptive penalty
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_std', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))
        self.momentum = 0.1

    def _update_statistics(self, targets):
        """Update running statistics for adaptive thresholding"""
        batch_mean = targets.mean()
        batch_std = targets.std()

        if self.update_count == 0:
            self.running_mean = batch_mean
            self.running_std = batch_std
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_std = (1 - self.momentum) * self.running_std + self.momentum * batch_std

        self.update_count += 1

    def forward(self, predictions, targets):
        """
        Enhanced forward pass with adaptive penalties
        """
        base_loss = self.base_criterion(predictions, targets)

        if self.training:
            self._update_statistics(targets)

        # Adaptive high-load threshold based on running statistics
        if self.adaptive_penalty:
            adaptive_threshold = self.running_mean + self.running_std * 0.5
            high_load_mask = targets > adaptive_threshold
        else:
            # Use percentile-based threshold
            target_flat = targets.view(-1)
            threshold_value = torch.quantile(target_flat, self.high_load_threshold)
            high_load_mask = targets > threshold_value

        # Identify underestimation
        underestimate_mask = predictions < targets

        # Apply penalty for underestimation during high-load periods
        penalty_mask = high_load_mask & underestimate_mask

        if penalty_mask.any():
            # Calculate penalty with load-aware weighting
            if self.load_aware_weighting:
                # Weight penalty by how much load exceeds threshold
                if self.adaptive_penalty:
                    excess_load = torch.clamp(targets[penalty_mask] - adaptive_threshold, min=0)
                    load_weights = 1 + excess_load / (self.running_std + 1e-8)
                else:
                    threshold_value = torch.quantile(targets.view(-1), self.high_load_threshold)
                    excess_load = torch.clamp(targets[penalty_mask] - threshold_value, min=0)
                    load_weights = 1 + excess_load / (targets.std() + 1e-8)

                penalty_loss = F.mse_loss(predictions[penalty_mask], targets[penalty_mask], reduction='none')
                weighted_penalty = (penalty_loss * load_weights).mean()
            else:
                weighted_penalty = F.mse_loss(predictions[penalty_mask], targets[penalty_mask])

            # Moderate penalty to avoid overwhelming the base loss
            penalty_weight = min(self.underestimate_penalty - 1.0, 0.5)
            total_loss = base_loss + penalty_weight * weighted_penalty
        else:
            total_loss = base_loss

        return total_loss


class ClearE(nn.Module):
    """
    CLEAR-E: Concept-aware Lightweight Energy Adaptation for Robust Forecasting

    Key differences from PROCEED:
    1. Energy-specific concept encoder with metadata integration
    2. Lightweight adaptation (only final layers)
    3. Drift memory module with regularization
    4. Energy-aware loss function
    """
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args

        # Freeze backbone if specified
        if args.freeze:
            backbone.requires_grad_(False)

        # Add adapters to backbone (but we'll only use lightweight adaptation)
        self.backbone = add_adapters_(backbone, args)
        self.more_bias = not args.freeze

        # Enhanced energy-specific metadata encoder
        metadata_dim = getattr(args, 'metadata_dim', 10)
        metadata_hidden = getattr(args, 'metadata_hidden_dim', 32)
        use_attention = getattr(args, 'use_metadata_attention', True)
        feature_groups = getattr(args, 'metadata_feature_groups', ['weather', 'temporal', 'calendar'])

        self.metadata_encoder = EnergyMetadataEncoder(
            metadata_dim=metadata_dim,
            hidden_dim=metadata_hidden,
            output_dim=args.concept_dim,
            use_attention=use_attention,
            feature_groups=feature_groups
        )

        # Lightweight adaptation generator (only final layers)
        target_layers = getattr(args, 'target_layers', ['projection', 'head', 'output', 'fc'])
        self.lightweight_generator = LightweightAdaptGenerator(
            backbone=backbone,
            concept_features=args.concept_dim,
            activation=nn.Sigmoid if args.act == 'sigmoid' else nn.Identity,
            mid_dim=args.bottleneck_dim,
            target_layers=target_layers
        )

        # Enhanced drift memory module
        memory_size = getattr(args, 'drift_memory_size', 10)
        reg_weight = getattr(args, 'drift_reg_weight', 0.1)
        adaptive_memory = getattr(args, 'adaptive_memory', True)
        drift_threshold = getattr(args, 'drift_threshold', 0.5)

        self.drift_memory = DriftMemoryModule(
            concept_dim=args.concept_dim,
            memory_size=memory_size,
            reg_weight=reg_weight,
            adaptive_memory=adaptive_memory,
            drift_threshold=drift_threshold
        )

        # Concept encoders for time series data (similar to PROCEED)
        self.mlp1 = nn.Sequential(
            Transpose(-1, -2),
            nn.Linear(args.seq_len, args.concept_dim),
            nn.GELU(),
            nn.Linear(args.concept_dim, args.concept_dim)
        )
        self.mlp2 = nn.Sequential(
            Transpose(-1, -2),
            nn.Linear(args.seq_len + args.pred_len, args.concept_dim),
            nn.GELU(),
            nn.Linear(args.concept_dim, args.concept_dim)
        )

        # Recent batch buffer
        self.register_buffer(
            'recent_batch',
            torch.zeros(1, args.seq_len + args.pred_len, args.enc_in),
            persistent=False
        )

        # EMA for concept tracking
        self.ema = args.ema
        if args.ema > 0:
            self.register_buffer('recent_concept', None, persistent=True)

        # Flags for different modes
        self.flag_online_learning = False
        self.flag_update = False
        self.flag_current = False
        self.flag_basic = False

    def generate_adaptation(self, x, metadata=None):
        """
        Generate adaptations using energy-aware concept encoding

        Args:
            x: Time series input [batch_size, seq_len, features]
            metadata: Energy metadata [batch_size, metadata_dim] or [batch_size, seq_len, metadata_dim]
        """
        # Time series concept encoding (similar to PROCEED)
        ts_concept = self.mlp1(x).mean(-2)  # [batch_size, concept_dim]

        # Recent concept from buffer
        recent_concept = self.mlp2(self.recent_batch).mean(-2).mean(
            list(range(0, self.recent_batch.dim() - 2))
        )

        # Apply EMA if enabled
        if self.ema > 0:
            if self.recent_concept is not None:
                recent_concept = self.recent_concept * self.ema + recent_concept * (1 - self.ema)
            if self.flag_update or self.flag_online_learning and not self.flag_current:
                self.recent_concept = recent_concept.detach()

        # Energy-specific concept enhancement
        if metadata is not None:
            metadata_concept = self.metadata_encoder(metadata)  # [batch_size, concept_dim]
            # Combine time series and metadata concepts
            enhanced_concept = ts_concept + 0.5 * metadata_concept  # Weighted combination
        else:
            enhanced_concept = ts_concept

        # Compute drift
        drift = enhanced_concept - recent_concept

        # Update drift memory and get regularization loss
        if self.training:
            self.drift_memory.update_memory(drift)

        # Generate lightweight adaptations (only for target layers)
        adaptations = self.lightweight_generator(drift)

        return adaptations, drift

    def get_drift_regularization_loss(self, current_drift):
        """Get smoothness regularization loss from drift memory"""
        return self.drift_memory.get_smoothness_loss(current_drift)

    def forward(self, *x, metadata=None):
        """
        Forward pass with optional metadata

        Args:
            x: Time series inputs
            metadata: Optional energy metadata
        """
        if self.flag_basic:
            # Basic mode - use default adaptations like PROCEED
            adaptations = {}
            for i, (k, adapter) in enumerate(self.lightweight_generator.bottlenecks.items()):
                adaptations[k] = adapter.biases[-1] if adapter.need_bias else [None] * len(self.lightweight_generator.dim_name_dict[k])
            drift = None
        else:
            # Generate energy-aware adaptations
            adaptations, drift = self.generate_adaptation(x[0], metadata)
            # Store drift for regularization loss computation
            self._last_drift = drift

        # Apply adaptations using PROCEED's logic but only for target layers
        for out_dim, adaptation in adaptations.items():
            for i in range(len(adaptation)):
                name = self.lightweight_generator.dim_name_dict[out_dim][i]
                self.backbone.get_submodule(name).assign_adaptation(adaptation[i])

        # Forward through backbone
        if self.args.do_predict:
            return self.backbone(*x)
        else:
            return self.backbone(*x)

    def freeze_adapter(self, freeze: bool):
        """Freeze/unfreeze adapter parameters"""
        for param in self.lightweight_generator.parameters():
            param.requires_grad = not freeze
        for param in self.metadata_encoder.parameters():
            param.requires_grad = not freeze
        for param in self.drift_memory.parameters():
            param.requires_grad = not freeze

    def freeze_bias(self, freeze: bool):
        """Freeze/unfreeze bias parameters"""
        for name, module in self.backbone.named_modules():
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.requires_grad = not freeze

    def get_adaptation_statistics(self):
        """Get comprehensive adaptation statistics for monitoring"""
        stats = {}

        # Drift memory statistics
        if hasattr(self.drift_memory, 'get_drift_statistics'):
            stats['drift_memory'] = self.drift_memory.get_drift_statistics()

        # Feature importance from metadata encoder
        if hasattr(self.metadata_encoder, 'get_feature_importance'):
            stats['feature_importance'] = self.metadata_encoder.get_feature_importance().detach().cpu().numpy()

        # Adaptation target statistics
        stats['adaptation_targets'] = {
            'num_target_groups': len(self.lightweight_generator.dim_name_dict),
            'total_target_modules': sum(len(names) for names in self.lightweight_generator.dim_name_dict.values()),
            'target_groups': list(self.lightweight_generator.dim_name_dict.keys())
        }

        # Model parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        adapter_params = sum(p.numel() for p in self.lightweight_generator.parameters())
        metadata_params = sum(p.numel() for p in self.metadata_encoder.parameters())

        stats['parameters'] = {
            'total': total_params,
            'adapter': adapter_params,
            'metadata_encoder': metadata_params,
            'adapter_ratio': adapter_params / total_params,
            'metadata_ratio': metadata_params / total_params
        }

        return stats

    def reset_adaptation_state(self):
        """Reset adaptation state for new sequence"""
        # Reset drift memory
        self.drift_memory.memory_ptr.zero_()
        self.drift_memory.memory_full.fill_(False)
        self.drift_memory.drift_detected.fill_(False)

        # Reset EMA if used
        if self.ema > 0 and hasattr(self, 'recent_concept'):
            self.recent_concept = None
