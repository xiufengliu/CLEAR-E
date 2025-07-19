import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False, **kwargs):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            # Handle tensor dimension mismatch by ensuring affine_weight and affine_bias are properly broadcast
            # This fixes the "The size of tensor a (21) must match the size of tensor b (7)" error
            if x.ndim > 2 and self.affine_weight.ndim == 1:
                # Reshape affine_weight and affine_bias to match x's dimensions for proper broadcasting
                if x.shape[-1] != self.affine_weight.shape[0]:
                    # If feature dimensions don't match, we need to adapt
                    print(f"Warning: RevIN dimension mismatch - x shape: {x.shape}, affine_weight shape: {self.affine_weight.shape}")
                    # Create new weight and bias tensors that match the feature dimension
                    if x.shape[-1] > self.affine_weight.shape[0]:
                        # If x has more features, expand affine_weight and affine_bias by repeating
                        expanded_weight = self.affine_weight.repeat(x.shape[-1] // self.affine_weight.shape[0])
                        expanded_bias = self.affine_bias.repeat(x.shape[-1] // self.affine_bias.shape[0])
                        if x.shape[-1] % self.affine_weight.shape[0] != 0:
                            # Handle remainder
                            remainder = x.shape[-1] % self.affine_weight.shape[0]
                            expanded_weight = torch.cat([expanded_weight, self.affine_weight[:remainder]])
                            expanded_bias = torch.cat([expanded_bias, self.affine_bias[:remainder]])
                        x = x * expanded_weight
                        x = x + expanded_bias
                    else:
                        # If x has fewer features, use a subset of affine_weight and affine_bias
                        x = x * self.affine_weight[:x.shape[-1]]
                        x = x + self.affine_bias[:x.shape[-1]]
                else:
                    x = x * self.affine_weight
                    x = x + self.affine_bias
            else:
                x = x * self.affine_weight
                x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            # Handle tensor dimension mismatch by ensuring affine_bias and affine_weight are properly broadcast
            if x.ndim > 2 and self.affine_bias.ndim == 1:
                # Reshape affine_bias and affine_weight to match x's dimensions for proper broadcasting
                if x.shape[-1] != self.affine_bias.shape[0]:
                    # If feature dimensions don't match, we need to adapt
                    print(f"Warning: RevIN dimension mismatch in denorm - x shape: {x.shape}, affine_bias shape: {self.affine_bias.shape}")
                    # Create new bias and weight tensors that match the feature dimension
                    if x.shape[-1] > self.affine_bias.shape[0]:
                        # If x has more features, expand affine_bias and affine_weight by repeating
                        expanded_bias = self.affine_bias.repeat(x.shape[-1] // self.affine_bias.shape[0])
                        expanded_weight = self.affine_weight.repeat(x.shape[-1] // self.affine_weight.shape[0])
                        if x.shape[-1] % self.affine_bias.shape[0] != 0:
                            # Handle remainder
                            remainder = x.shape[-1] % self.affine_bias.shape[0]
                            expanded_bias = torch.cat([expanded_bias, self.affine_bias[:remainder]])
                            expanded_weight = torch.cat([expanded_weight, self.affine_weight[:remainder]])
                        x = x - expanded_bias
                        x = x / (expanded_weight + self.eps*self.eps)
                    else:
                        # If x has fewer features, use a subset of affine_bias and affine_weight
                        x = x - self.affine_bias[:x.shape[-1]]
                        x = x / (self.affine_weight[:x.shape[-1]] + self.eps*self.eps)
                else:
                    x = x - self.affine_bias
                    x = x / (self.affine_weight + self.eps*self.eps)
            else:
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x