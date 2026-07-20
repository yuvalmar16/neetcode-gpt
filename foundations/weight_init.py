import torch
import torch.nn as nn
import math
from typing import List

class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = (2 / (fan_in + fan_out)) ** 0.5
        matrix = torch.randn(fan_out, fan_in) * std
        return torch.round(matrix, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = (2 / fan_in) ** 0.5 
        matrix = torch.randn(fan_out, fan_in) * std
        return torch.round(matrix, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        # 1. Initialize and scale all layer weight matrices first to match the expected seed consumption sequence
        weights = []
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            fan_out = hidden_dim 
            
            if init_type == 'xavier':
                std = (2.0 / (fan_in + fan_out)) ** 0.5
            elif init_type == 'kaiming':    
                std = (2.0 / fan_in) ** 0.5
            elif init_type == 'random':
                std = 1.0
            else:
                raise ValueError("Unknown initialization type")
                
            W = torch.randn(fan_out, fan_in) * std
            weights.append(W)
            
        # 2. Generate the random input vector AFTER creating the weights
        current_activation = torch.randn(1, input_dim)
        stds = []
        
        # 3. Perform the forward pass
        for W in weights:
            z = current_activation @ W.t()    
            current_activation = torch.clamp(z, min=0) # ReLU
            
            layer_std = torch.std(current_activation, unbiased=True).item()
            stds.append(round(layer_std, 2))

        return stds