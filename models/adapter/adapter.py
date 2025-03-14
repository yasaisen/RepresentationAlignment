"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2503141354
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU激活函數，提供比GELU更好的梯度流
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super(SwiGLU, self).__init__()
        self.w1 = nn.Linear(input_dim, hidden_dim)
        self.w2 = nn.Linear(input_dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class ConditionalLayerNorm(nn.Module):
    """
    條件層規範化，允許根據輸入調整規範化參數
    """
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super(ConditionalLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        
        # 條件調整參數
        self.scale_mapper = nn.Linear(hidden_size, hidden_size)
        self.bias_mapper = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor = None):
        mean = x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.variance_epsilon)
        
        if condition is not None:
            # 根據條件調整權重和偏置
            scale = self.scale_mapper(condition).unsqueeze(1)
            bias = self.bias_mapper(condition).unsqueeze(1)
            return normalized * (self.weight + scale) + (self.bias + bias)
        else:
            return normalized * self.weight + self.bias

class AdvancedAdapterLayer(nn.Module):
    def __init__( # ConditionalLayerNorm, SwiGLU, adapter_scaling
        self, 
        input_dim: int, 
        bottleneck_dim: int, 
        output_dim: int, 
        dropout_rate: float = 0.1,
        adapter_scaling: float = 1,
    ):
        super(AdvancedAdapterLayer, self).__init__()
        self.input_dim = input_dim
        self.adapter_scaling = adapter_scaling
        
        self.layer_norm = ConditionalLayerNorm(input_dim)
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.activation = SwiGLU(bottleneck_dim, bottleneck_dim * 2)
        self.up_project = nn.Linear(bottleneck_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        residual = x
        x_norm = self.layer_norm(x, condition)
        
        adapter_output = self.down_project(x_norm)
        adapter_output = self.activation(adapter_output)
        adapter_output = self.up_project(adapter_output)
        adapter_output = self.dropout(adapter_output)
        
        output = residual + self.adapter_scaling * adapter_output
        return output
    











    