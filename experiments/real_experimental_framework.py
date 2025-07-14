"""
Real Experimental Framework for CLEAR-E
Implements actual model training and evaluation on real datasets
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter.clear_e import ClearE
from experiments.clear_e_model import CLEAR_E, create_clear_e_model
from models.PatchTST import Model as PatchTST
from models.DLinear import Model as DLinear
from models.Transformer import Model as Transformer

class ModelWrapper(nn.Module):
    """Wrapper to ensure correct output shape"""
    def __init__(self, model, n_loads, horizon):
        super().__init__()
        self.model = model
        self.n_loads = n_loads
        self.horizon = horizon

    def forward(self, x):
        output = self.model(x)

        # Handle different output shapes
        if len(output.shape) == 2:
            # [batch_size, horizon * n_features] -> [batch_size, horizon, n_loads]
            batch_size = output.shape[0]
            n_features = output.shape[1] // self.horizon

            # Reshape and select only load columns
            output = output.view(batch_size, self.horizon, n_features)
            output = output[:, :, :self.n_loads]

        elif len(output.shape) == 3:
            # [batch_size, horizon, n_features] -> [batch_size, horizon, n_loads]
            if output.shape[2] > self.n_loads:
                output = output[:, :, :self.n_loads]

        return output

class RealExperimentalFramework:
    """Real experimental framework with actual model training"""
    
    def __init__(self, config_path: str = None):
        """Initialize the experimental framework"""
        self.config = self._load_config(config_path)
        self.results = {}
        self.datasets = {}
        self.scalers = {}
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load experimental configuration"""
        default_config = {
            'datasets': ['ETTh1', 'ETTh2', 'southern_china'],
            'lookback': 96,  # 4 days for hourly data
            'horizon': 24,   # 1 day prediction
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 50,
            'patience': 10,
            'n_runs': 3,
            'train_ratio': 0.7,
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'baseline_models': ['DLinear', 'PatchTST', 'Transformer'],
            'clear_e_config': {
                'concept_dim': 64,
                'hidden_dim': 128,
                'bottleneck_dim': 32,
                'memory_size': 10,
                'momentum': 0.9,
                'penalty_weight': 1.4,
                'drift_reg_weight': 0.1,
                'adaptive_memory': True,
                'drift_threshold': 0.5
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
            
        return default_config
    
    def load_datasets(self):
        """Load and preprocess real datasets"""
        print("Loading real datasets...")
        
        dataset_dir = "../dataset/processed"
        
        for dataset_name in self.config['datasets']:
            print(f"  Loading {dataset_name}...")
            
            if dataset_name.startswith('ETT'):
                file_path = os.path.join(dataset_dir, f"{dataset_name}_processed.csv")
            else:
                file_path = os.path.join(dataset_dir, f"{dataset_name}_processed.csv")
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Remove rows with all NaN values
                df = df.dropna(how='all')
                
                # Forward fill and backward fill remaining NaN values
                df = df.fillna(method='ffill').fillna(method='bfill')
                
                # Ensure we have enough data
                if len(df) < self.config['lookback'] + self.config['horizon']:
                    print(f"    Warning: {dataset_name} has insufficient data, skipping...")
                    continue
                
                self.datasets[dataset_name] = df
                print(f"    Loaded {dataset_name}: {df.shape}")
            else:
                print(f"    Warning: {file_path} not found, skipping {dataset_name}")
    
    def _prepare_dataset(self, dataset: pd.DataFrame, dataset_name: str) -> Dict:
        """Prepare dataset for training"""
        # Identify load columns and metadata columns
        if dataset_name.startswith('ETT'):
            # ETT datasets: OT is target, others are features
            load_columns = ['OT']
            feature_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        elif dataset_name == 'southern_china':
            # Southern China: transformer load columns
            load_columns = [col for col in dataset.columns if 
                          (col.count('-') == 2 and col.replace('-', '').replace('_', '').isdigit())][:5]  # Limit to 5 transformers
            feature_columns = ['TEMP', 'DEWP', 'RH', 'WDSP', 'PRCP', 'MAX', 'MIN']
        else:
            # Default: try to identify load columns
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            calendar_features = ['hour', 'day_of_week', 'day_of_year', 'month', 'quarter', 'year',
                               'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                               'is_weekend', 'season', 'is_holiday']
            load_columns = [col for col in numeric_cols if col not in calendar_features][:5]
            feature_columns = calendar_features
        
        # Ensure we have valid columns
        load_columns = [col for col in load_columns if col in dataset.columns]
        feature_columns = [col for col in feature_columns if col in dataset.columns]
        
        if not load_columns:
            raise ValueError(f"No valid load columns found for {dataset_name}")
        
        # Calendar features
        calendar_features = ['hour', 'day_of_week', 'month', 'is_weekend']
        calendar_features = [col for col in calendar_features if col in dataset.columns]
        
        # Combine all features
        all_features = load_columns + feature_columns + calendar_features
        all_features = list(dict.fromkeys(all_features))  # Remove duplicates while preserving order
        
        # Select data
        data = dataset[all_features].values
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - self.config['lookback'] - self.config['horizon'] + 1):
            X.append(data[i:i + self.config['lookback']])
            # For y, only take the load columns
            y_seq = data[i + self.config['lookback']:i + self.config['lookback'] + self.config['horizon'], :len(load_columns)]
            y.append(y_seq)

        X = np.array(X)  # Shape: [n_samples, lookback, n_features]
        y = np.array(y)  # Shape: [n_samples, horizon, n_loads]

        print(f"    Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        n_samples = len(X)
        train_end = int(n_samples * self.config['train_ratio'])
        val_end = int(n_samples * (self.config['train_ratio'] + self.config['val_ratio']))
        
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Scale data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        # Fit scalers on training data
        X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
        
        # Transform validation and test data
        X_val_scaled = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, y_val.shape[-1])).reshape(y_val.shape)
        X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
        
        # Store scalers
        self.scalers[dataset_name] = {'X': scaler_X, 'y': scaler_y}
        
        return {
            'train': (X_train_scaled, y_train_scaled),
            'val': (X_val_scaled, y_val_scaled),
            'test': (X_test_scaled, y_test_scaled),
            'load_columns': load_columns,
            'feature_columns': feature_columns,
            'calendar_features': calendar_features,
            'n_features': len(all_features),
            'n_loads': len(load_columns)
        }

    def _create_data_loaders(self, data_dict: Dict) -> Dict:
        """Create PyTorch data loaders"""
        loaders = {}

        for split in ['train', 'val', 'test']:
            X, y = data_dict[split]
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)

            dataset = TensorDataset(X_tensor, y_tensor)
            shuffle = (split == 'train')

            loaders[split] = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=shuffle,
                drop_last=False
            )

        return loaders

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics"""
        # Flatten arrays for metric calculation
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        # Remove any NaN values
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]

        if len(y_true_clean) == 0:
            return {'rmse': float('inf'), 'mae': float('inf'), 'mape': float('inf')}

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)

        # MAPE calculation with handling for zero values
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + epsilon))) * 100

        # Peak load error (error during high load periods)
        high_load_threshold = np.percentile(y_true_clean, 80)
        high_load_mask = y_true_clean > high_load_threshold
        if np.any(high_load_mask):
            peak_load_error = np.mean(np.abs((y_true_clean[high_load_mask] - y_pred_clean[high_load_mask]) /
                                           (y_true_clean[high_load_mask] + epsilon))) * 100
        else:
            peak_load_error = mape

        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'peak_load_error': peak_load_error
        }

    def _train_model(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                    dataset_name: str) -> Dict:
        """Train a model and return training metrics"""
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}

        start_time = time.time()

        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass - handle different model types
                try:
                    if hasattr(model, 'forward'):
                        outputs = model(batch_X)
                    else:
                        outputs = model(batch_X)

                    # The ModelWrapper should handle shape matching
                    if outputs.shape != batch_y.shape:
                        print(f"      Warning: Shape mismatch after wrapper - Output: {outputs.shape}, Target: {batch_y.shape}")
                        # Try to fix any remaining shape issues
                        if outputs.numel() == batch_y.numel():
                            outputs = outputs.reshape(batch_y.shape)
                        else:
                            raise RuntimeError(f"Cannot match shapes: {outputs.shape} vs {batch_y.shape}")

                except Exception as e:
                    print(f"      Forward pass error: {e}")
                    print(f"      Input shape: {batch_X.shape}, Target shape: {batch_y.shape}")
                    if hasattr(model, '__class__'):
                        print(f"      Model type: {model.__class__.__name__}")
                    raise e

                loss = criterion(outputs, batch_y)

                # Add regularization for CLEAR-E models
                if hasattr(model, 'get_drift_regularization_loss'):
                    if hasattr(model, '_last_drift') and model._last_drift is not None:
                        reg_loss = model.get_drift_regularization_loss(model._last_drift)
                        loss = loss + reg_loss

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                    outputs = model(batch_X)
                    if outputs.shape != batch_y.shape:
                        outputs = outputs.view(batch_y.shape)

                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"    Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch + 1}/{self.config['epochs']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Restore best model
        model.load_state_dict(best_model_state)

        training_time = time.time() - start_time

        return {
            'training_time': training_time,
            'best_val_loss': best_val_loss,
            'training_history': training_history,
            'final_epoch': epoch + 1
        }

    def _evaluate_model(self, model: nn.Module, test_loader: DataLoader, dataset_name: str) -> Dict:
        """Evaluate model on test set"""
        model.eval()
        model = model.to(self.device)

        all_predictions = []
        all_targets = []

        start_time = time.time()

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                outputs = model(batch_X)

                # The ModelWrapper should handle shape matching
                if outputs.shape != batch_y.shape:
                    if outputs.numel() == batch_y.numel():
                        outputs = outputs.reshape(batch_y.shape)
                    else:
                        raise RuntimeError(f"Cannot match shapes: {outputs.shape} vs {batch_y.shape}")

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        inference_time = (time.time() - start_time) / len(test_loader) * 1000  # ms per batch

        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Inverse transform to original scale
        scaler_y = self.scalers[dataset_name]['y']
        predictions_orig = scaler_y.inverse_transform(predictions.reshape(-1, predictions.shape[-1])).reshape(predictions.shape)
        targets_orig = scaler_y.inverse_transform(targets.reshape(-1, targets.shape[-1])).reshape(targets.shape)

        # Calculate metrics
        metrics = self._calculate_metrics(targets_orig, predictions_orig)
        metrics['inference_time'] = inference_time

        # Calculate model size
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metrics['n_parameters'] = n_parameters

        # Estimate memory usage (rough approximation)
        memory_usage = n_parameters * 4 / (1024 * 1024)  # MB (assuming float32)
        metrics['memory_usage'] = memory_usage

        return metrics

    def _create_clear_e_model(self, data_config: Dict) -> nn.Module:
        """Create CLEAR-E model"""
        # Create a simple args object for compatibility
        class Args:
            def __init__(self, config, data_config):
                self.seq_len = config['lookback']
                self.pred_len = config['horizon']
                self.enc_in = data_config['n_features']
                self.dec_in = data_config['n_features']
                self.c_out = data_config['n_loads']
                self.d_model = config['clear_e_config']['hidden_dim']
                self.concept_dim = config['clear_e_config']['concept_dim']
                self.freeze = False
                self.do_predict = False
                self.individual = False

                # Additional attributes needed by baseline models
                self.n_heads = 8
                self.e_layers = 2
                self.d_layers = 1
                self.d_ff = 2048
                self.dropout = 0.1
                self.activation = 'gelu'
                self.output_attention = False
                self.label_len = 48
                self.factor = 1
                self.embed = 'timeF'
                self.freq = 'h'
                self.moving_avg = 25
                self.distil = True
                self.mix = True
                self.des = 'Exp'
                self.itr = 1

                # CLEAR-E specific parameters
                for key, value in config['clear_e_config'].items():
                    setattr(self, key, value)

                # Additional CLEAR-E specific attributes
                self.tune_mode = 'linear_probing'
                self.ema = 0.9
                self.flag_update = True
                self.flag_online_learning = False
                self.flag_current = False
                self.flag_basic = False
                self.act = 'gelu'
                self.norm = 'LayerNorm'
                self.use_norm = True

        args = Args(self.config, data_config)

        # Create backbone model (using DLinear as backbone)
        backbone = DLinear(args)

        # Wrap with CLEAR-E adapter
        model = ClearE(backbone, args)

        # Wrap with shape fixer
        wrapped_model = ModelWrapper(model, data_config['n_loads'], self.config['horizon'])

        return wrapped_model

    def _create_baseline_model(self, model_name: str, data_config: Dict) -> nn.Module:
        """Create baseline model"""
        class Args:
            def __init__(self, config, data_config):
                self.seq_len = config['lookback']
                self.pred_len = config['horizon']
                self.enc_in = data_config['n_features']
                self.dec_in = data_config['n_features']
                self.c_out = data_config['n_loads']
                self.d_model = 512
                self.n_heads = 8
                self.e_layers = 2
                self.d_layers = 1
                self.d_ff = 2048
                self.dropout = 0.1
                self.activation = 'gelu'
                self.output_attention = False
                self.individual = False

                # PatchTST specific
                self.patch_len = 16
                self.stride = 8
                self.fc_dropout = 0.1
                self.head_dropout = 0.0
                self.padding_patch = 'end'
                self.revin = 1
                self.affine = 0
                self.subtract_last = 0
                self.decomposition = 0
                self.kernel_size = 25

                # Additional common attributes
                self.label_len = 48
                self.factor = 1
                self.embed = 'timeF'
                self.freq = 'h'
                self.moving_avg = 25
                self.distil = True
                self.mix = True
                self.des = 'Exp'
                self.itr = 1

        args = Args(self.config, data_config)

        if model_name == 'DLinear':
            model = DLinear(args)
        elif model_name == 'PatchTST':
            model = PatchTST(args)
        elif model_name == 'Transformer':
            model = Transformer(args)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Wrap with shape fixer
        wrapped_model = ModelWrapper(model, data_config['n_loads'], self.config['horizon'])
        return wrapped_model

    def run_experiments(self):
        """Run complete experimental evaluation"""
        print("Starting real experimental evaluation...")

        # Load datasets
        self.load_datasets()

        if not self.datasets:
            print("No datasets loaded. Exiting.")
            return

        results = {}

        for dataset_name, dataset in self.datasets.items():
            print(f"\nEvaluating on {dataset_name}...")

            # Prepare dataset
            try:
                data_config = self._prepare_dataset(dataset, dataset_name)
                data_loaders = self._create_data_loaders(data_config)
            except Exception as e:
                print(f"  Error preparing {dataset_name}: {e}")
                continue

            results[dataset_name] = {}

            # Multiple runs for statistical validation
            for run in range(self.config['n_runs']):
                print(f"  Run {run + 1}/{self.config['n_runs']}")

                run_results = {}

                # Evaluate CLEAR-E
                print("    Training CLEAR-E...")
                try:
                    clear_e_model = self._create_clear_e_model(data_config)
                    train_metrics = self._train_model(clear_e_model, data_loaders['train'], data_loaders['val'], dataset_name)
                    test_metrics = self._evaluate_model(clear_e_model, data_loaders['test'], dataset_name)

                    run_results['CLEAR-E'] = {**train_metrics, **test_metrics}
                except Exception as e:
                    print(f"      Error with CLEAR-E: {e}")
                    run_results['CLEAR-E'] = None

                # Evaluate baselines
                for model_name in self.config['baseline_models']:
                    print(f"    Training {model_name}...")
                    try:
                        baseline_model = self._create_baseline_model(model_name, data_config)
                        train_metrics = self._train_model(baseline_model, data_loaders['train'], data_loaders['val'], dataset_name)
                        test_metrics = self._evaluate_model(baseline_model, data_loaders['test'], dataset_name)

                        run_results[model_name] = {**train_metrics, **test_metrics}
                    except Exception as e:
                        print(f"      Error with {model_name}: {e}")
                        run_results[model_name] = None

                # Store run results
                if run == 0:
                    for model_name in run_results:
                        results[dataset_name][model_name] = []

                for model_name, metrics in run_results.items():
                    if metrics is not None:
                        results[dataset_name][model_name].append(metrics)

        self.results = results
        return results

    def analyze_results(self) -> Dict:
        """Analyze experimental results and compute statistics"""
        if not self.results:
            print("No results to analyze. Run experiments first.")
            return {}

        analysis = {}

        for dataset_name, dataset_results in self.results.items():
            print(f"\nAnalyzing results for {dataset_name}:")

            analysis[dataset_name] = {}

            for model_name, runs in dataset_results.items():
                if not runs:
                    continue

                # Calculate statistics across runs
                metrics = ['rmse', 'mae', 'mape', 'peak_load_error', 'training_time', 'inference_time', 'n_parameters', 'memory_usage']

                stats = {}
                for metric in metrics:
                    values = [run[metric] for run in runs if metric in run and not np.isnan(run[metric])]
                    if values:
                        stats[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }

                analysis[dataset_name][model_name] = stats

                # Print summary
                print(f"  {model_name}:")
                if 'rmse' in stats:
                    print(f"    RMSE: {stats['rmse']['mean']:.4f} ± {stats['rmse']['std']:.4f}")
                if 'mape' in stats:
                    print(f"    MAPE: {stats['mape']['mean']:.2f}% ± {stats['mape']['std']:.2f}%")
                if 'training_time' in stats:
                    print(f"    Training Time: {stats['training_time']['mean']:.1f}s ± {stats['training_time']['std']:.1f}s")

        return analysis

    def save_results(self, output_dir: str = "results"):
        """Save experimental results"""
        os.makedirs(output_dir, exist_ok=True)

        # Save raw results
        with open(os.path.join(output_dir, "raw_results.json"), 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj

            json.dump(convert_numpy(self.results), f, indent=2)

        # Save analysis
        analysis = self.analyze_results()
        with open(os.path.join(output_dir, "analysis.json"), 'w') as f:
            json.dump(convert_numpy(analysis), f, indent=2)

        # Create summary table
        self._create_summary_table(analysis, output_dir)

        print(f"\nResults saved to {output_dir}/")

    def _create_summary_table(self, analysis: Dict, output_dir: str):
        """Create summary table for paper"""
        summary_data = []

        for dataset_name, dataset_analysis in analysis.items():
            for model_name, stats in dataset_analysis.items():
                row = {
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'RMSE': f"{stats.get('rmse', {}).get('mean', 0):.4f} ± {stats.get('rmse', {}).get('std', 0):.4f}",
                    'MAE': f"{stats.get('mae', {}).get('mean', 0):.4f} ± {stats.get('mae', {}).get('std', 0):.4f}",
                    'MAPE (%)': f"{stats.get('mape', {}).get('mean', 0):.2f} ± {stats.get('mape', {}).get('std', 0):.2f}",
                    'Peak Load Error (%)': f"{stats.get('peak_load_error', {}).get('mean', 0):.2f} ± {stats.get('peak_load_error', {}).get('std', 0):.2f}",
                    'Training Time (s)': f"{stats.get('training_time', {}).get('mean', 0):.1f} ± {stats.get('training_time', {}).get('std', 0):.1f}",
                    'Parameters': f"{stats.get('n_parameters', {}).get('mean', 0):.0f}",
                    'Memory (MB)': f"{stats.get('memory_usage', {}).get('mean', 0):.1f}"
                }
                summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, "summary_table.csv"), index=False)

        # Create LaTeX table
        latex_table = summary_df.to_latex(index=False, escape=False)
        with open(os.path.join(output_dir, "summary_table.tex"), 'w') as f:
            f.write(latex_table)

        print("Summary table saved as CSV and LaTeX")

def main():
    """Main function to run experiments"""
    framework = RealExperimentalFramework()

    # Run experiments
    results = framework.run_experiments()

    # Analyze and save results
    framework.save_results()

    print("\nExperimental evaluation completed!")

if __name__ == "__main__":
    main()
