"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2502141330
"""

import torch
import torch.nn as nn
from typing import Dict, Type, Union
import math
from typing import Tuple, Dict, Any
from datetime import datetime

class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(
            torch.zeros((rank, linear.in_features))
        )
        self.lora_B = nn.Parameter(
            torch.zeros((linear.out_features, rank))
        )
        self.dropout_layer = nn.Dropout(p=dropout)
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.merged = False
        if merge_weights:
            self.merge_weights()
    
    def merge_weights(self) -> None:
        if self.merged:
            return
        
        with torch.no_grad():
            self.linear.weight.data += (
                self.lora_B @ self.lora_A
            ) * self.scaling
            self.merged = True
    
    def unmerge_weights(self) -> None:
        if not self.merged:
            return
            
        with torch.no_grad():
            self.linear.weight.data -= (
                self.lora_B @ self.lora_A
            ) * self.scaling
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)
            
        lora_output = (
            self.dropout_layer(x) @ self.lora_A.t() @ self.lora_B.t()
        ) * self.scaling
        return self.linear(x) + lora_output

def get_attr(model: nn.Module, name: str) -> nn.Module:
    """遞迴獲取模型中的子模組"""
    attr = model
    for n in name.split('.'):
        attr = getattr(attr, n)
    return attr

def add_lora_layers(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: Union[Type[nn.Module], tuple[Type[nn.Module], ...]] = nn.Linear,
    safe_mode: bool = True
) -> Dict[str, LoRALinear]:
    """
    安全地將LoRA添加到模型中的所有目標層。
    
    Args:
        model: 要添加LoRA的基礎模型
        rank: LoRA的秩
        alpha: LoRA縮放因子
        dropout: LoRA dropout率
        target_modules: 要添加LoRA的層的類型
        safe_mode: 是否在替換前保存原始權重
        
    Returns:
        包含所有LoRA層的字典，鍵為層的名稱
    """
    lora_layers = {}
    original_weights = {}
    
    for name, module in model.named_modules():
        if isinstance(module, target_modules):
            if safe_mode:
                # 保存原始權重
                original_weights[name] = {
                    'weight': module.weight.data.clone(),
                    'bias': module.bias.data.clone() if module.bias is not None else None
                }
            
            # 替換為LoRA層
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model if not parent_name else get_attr(model, parent_name)
            
            lora_layer = LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
            
            # 確保權重正確複製
            with torch.no_grad():
                lora_layer.linear.weight.data.copy_(original_weights[name]['weight'])
                if original_weights[name]['bias'] is not None:
                    lora_layer.linear.bias.data.copy_(original_weights[name]['bias'])
            
            setattr(parent, child_name, lora_layer)
            lora_layers[name] = lora_layer
            
    return lora_layers, original_weights

class ModelWithLoRA:
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.lora_layers = None
        self.original_weights = None
    
    def add_lora(self, rank: int = 8, alpha: float = 1.0, dropout: float = 0.0):
        """添加LoRA層"""
        self.lora_layers, self.original_weights = add_lora_layers(
            self.base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            safe_mode=True
        )
    
    def restore_original_weights(self):
        """恢復原始權重"""
        if self.original_weights is None:
            return
            
        for name, weights in self.original_weights.items():
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = self.base_model if not parent_name else get_attr(self.base_model, parent_name)
            
            module = nn.Linear(
                weights['weight'].size(1),
                weights['weight'].size(0),
                bias=weights['bias'] is not None
            )
            
            with torch.no_grad():
                module.weight.data.copy_(weights['weight'])
                if weights['bias'] is not None:
                    module.bias.data.copy_(weights['bias'])
                    
            setattr(parent, child_name, module)
            
        self.lora_layers = None
        self.original_weights = None

def lora_save(model_with_lora: ModelWithLoRA, save_path: str, adddatetime: bool = True):
    lora_state_dict = {
        name: {
            'lora_A': layer.lora_A.data.clone(),
            'lora_B': layer.lora_B.data.clone()
        }
        for name, layer in model_with_lora.lora_layers.items()
    }
    if adddatetime:
        filename = datetime.now().strftime("%y%m%d%H%M_lora_weights_") + save_path + '.pt'
    else:
        filename = save_path
    torch.save(lora_state_dict, filename)

def lora_load(model_with_lora: ModelWithLoRA, save_path: str):
    loaded_state_dict = torch.load(save_path)
    for name, layer in model_with_lora.lora_layers.items():
        if name in loaded_state_dict:
            with torch.no_grad():
                layer.lora_A.data.copy_(loaded_state_dict[name]['lora_A'])
                layer.lora_B.data.copy_(loaded_state_dict[name]['lora_B'])












