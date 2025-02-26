"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2502251532
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common.utils import log_print, get_trainable_params
from ..lora.lora import ModelWithLoRA


class ImageEncoder(nn.Module):
    def __init__(self, 
        img_encoder_path, 
        encoder_dim, 
        dim_proj=True,
        embed_dim=512, 
        rank = 8, 
        alpha = 1.0, 
        dropout = 0.0,
        device='cuda',
        dtype=torch.float32, 
        output_dtype=torch.float32
    ):
        super().__init__()
        self.state_name = 'ImageEncoder'
        self.device = device
        self.dtype = dtype
        print()
        log_print(self.state_name, "Building...")
        self.img_encoder = torch.load(img_encoder_path)

        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.img_encoder)}")
        for name, param in self.img_encoder.named_parameters():
            param.requires_grad = False
        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.img_encoder)}")

        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.dim_proj = dim_proj
        self.output_dtype = output_dtype

        log_print(self.state_name, "Adding LoRA...")
        self.img_encoder_lora = ModelWithLoRA(self.img_encoder)
        self.img_encoder_lora.add_lora(rank=self.rank, alpha=self.alpha, dropout=self.dropout)
        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.img_encoder_lora.base_model)}")

        self.img_encoder_lora.base_model.to(self.device) # 5365MiB

        if self.dim_proj:
            self.embed_proj = nn.Linear(self.encoder_dim , self.embed_dim).to(self.device)#.to(self.dtype)
            log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.img_encoder_lora.base_model) + get_trainable_params(self.embed_proj)}")
        
        self.normalize = lambda x: F.normalize(x, p=2, dim=1)
        log_print(self.state_name, "...Done\n")
        
    def forward(self, x):
        x = self.img_encoder_lora.base_model(x)
        if self.dim_proj:
            x = self.embed_proj(x)

        normalized_features = self.normalize(x)#.to(self.output_dtype)
        return normalized_features












