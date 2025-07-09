import copy
import numpy as np
import torch
from tqdm import tqdm

from adapter import clear_e
from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent
from exp import Exp_Online


class Exp_ClearE(Exp_Online):
    """
    Experiment class for CLEAR-E method
    Extends Exp_Online to handle energy-specific adaptations and metadata
    """
    def __init__(self, args):
        args = copy.deepcopy(args)
        args.merge_weights = 1
        super(Exp_ClearE, self).__init__(args)
        self.online_phases = ['val', 'test', 'online']
        self.mean_dim = args.seq_len
        
        # Energy-aware loss function
        if getattr(args, 'use_energy_loss', False):
            from adapter.clear_e import EnergyAwareLoss
            self.energy_criterion = EnergyAwareLoss(
                base_criterion=self._select_criterion(),
                high_load_threshold=getattr(args, 'high_load_threshold', 0.8),
                underestimate_penalty=getattr(args, 'underestimate_penalty', 2.0)
            )
        else:
            self.energy_criterion = None

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        """Online learning phase with CLEAR-E adaptations"""
        if phase == 'val' and self.args.val_online_lr:
            lr = self.model_optim.param_groups[0]['lr']
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = self.args.online_learning_rate
        
        self.model_optim.zero_grad()
        self._model.freeze_adapter(True)
        ret = super().online(online_data, target_variate, phase, show_progress)
        self._model.freeze_adapter(False)
        
        if phase == 'val' and self.args.val_online_lr:
            for j in range(len(self.model_optim.param_groups)):
                self.model_optim.param_groups[j]['lr'] = lr
        return ret

    def update_valid(self, valid_data=None, valid_dataloader=None):
        """Update validation with CLEAR-E specific handling"""
        self.phase = 'online'
        if valid_data is None:
            valid_data = get_dataset(self.args, 'val', self.device,
                                     wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                                     **self.wrap_data_kwargs, take_post=self.args.pred_len - 1)
        valid_loader = get_dataloader(valid_data, self.args, 'online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None
        self.model.train()
        predictions = []
        
        if not self.args.joint_update_valid:
            for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
                self._model.freeze_bias(False)
                self._model.freeze_adapter(True)
                self._update_online_clear_e(recent_batch, criterion, model_optim, scaler, flag_current=False)
                self._model.freeze_bias(True)
                if not self.args.freeze:
                    self._model.backbone.requires_grad_(False)
                self._model.freeze_adapter(False)
                _, outputs = self._update_online_clear_e(current_batch, criterion, model_optim, scaler, flag_current=True)
                if not self.args.freeze:
                    self._model.backbone.requires_grad_(True)
                if self.args.do_predict:
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
            self._model.freeze_bias(False)
        else:
            for i, (recent_batch, current_batch) in enumerate(tqdm(valid_loader, mininterval=10)):
                self._update_online_clear_e(recent_batch, criterion, model_optim, scaler, flag_current=True)
                if self.args.do_predict:
                    self.model.eval()
                    with torch.no_grad():
                        outputs = self.forward(current_batch)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    predictions.append(outputs.detach().cpu().numpy())
                    self.model.train()
        
        self.model_optim.zero_grad()
        self._model.freeze_adapter(True)
        trainable_params = sum([param.nelement() if param.requires_grad else 0 for param in self._model.parameters()])
        print(f'Trainable Params: {trainable_params}', '({:.1f}%)'.format(trainable_params / self.model_params * 100))
        return predictions

    def _update_online_clear_e(self, batch_data, criterion, model_optim, scaler, flag_current=False):
        """
        Update online with CLEAR-E specific features including metadata and drift regularization
        """
        # Extract metadata if available
        metadata = None
        if len(batch_data) > 4:  # Assuming metadata is the 5th element
            batch_x, batch_y, batch_x_mark, batch_y_mark, metadata = batch_data
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch_data
            
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        
        if metadata is not None:
            metadata = metadata.float().to(self.device)

        # Set flags for CLEAR-E
        self._model.flag_current = flag_current
        self._model.flag_update = True
        self._model.flag_online_learning = True

        # Update recent batch buffer
        if hasattr(self._model, 'recent_batch'):
            recent_data = torch.cat([batch_x, batch_y], dim=1)
            self._model.recent_batch = recent_data.mean(0, keepdim=True)

        # Forward pass with metadata
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if metadata is not None:
                    outputs = self._model(batch_x, batch_x_mark, metadata=metadata)
                else:
                    outputs = self._model(batch_x, batch_x_mark)
                
                if self.args.output_attention:
                    outputs = outputs[0]
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # Use energy-aware loss if enabled
                if self.energy_criterion is not None:
                    loss = self.energy_criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs, batch_y)
                
                # Add drift regularization loss
                if hasattr(self._model, 'get_drift_regularization_loss'):
                    # Get current drift from the last forward pass
                    if hasattr(self._model, '_last_drift'):
                        drift_reg_loss = self._model.get_drift_regularization_loss(self._model._last_drift)
                        loss = loss + drift_reg_loss

            scaler.scale(loss).backward()
            scaler.step(model_optim)
            scaler.update()
        else:
            if metadata is not None:
                outputs = self._model(batch_x, batch_x_mark, metadata=metadata)
            else:
                outputs = self._model(batch_x, batch_x_mark)
                
            if self.args.output_attention:
                outputs = outputs[0]
            
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            
            # Use energy-aware loss if enabled
            if self.energy_criterion is not None:
                loss = self.energy_criterion(outputs, batch_y)
            else:
                loss = criterion(outputs, batch_y)
            
            # Add drift regularization loss
            if hasattr(self._model, 'get_drift_regularization_loss'):
                # Get current drift from the last forward pass
                if hasattr(self._model, '_last_drift'):
                    drift_reg_loss = self._model.get_drift_regularization_loss(self._model._last_drift)
                    loss = loss + drift_reg_loss

            loss.backward()
            model_optim.step()

        model_optim.zero_grad()
        return loss.item(), outputs

    def _build_model(self, model=None, framework_class=None):
        """Build CLEAR-E model"""
        model = super()._build_model(model, framework_class=clear_e.ClearE)
        print(model)
        return model

    def _select_criterion(self):
        """Select criterion with energy-aware option"""
        if getattr(self.args, 'use_energy_loss', False):
            from adapter.clear_e import EnergyAwareLoss
            return EnergyAwareLoss(
                base_criterion=super()._select_criterion(),
                high_load_threshold=getattr(self.args, 'high_load_threshold', 0.8),
                underestimate_penalty=getattr(self.args, 'underestimate_penalty', 2.0)
            )
        else:
            return super()._select_criterion()
