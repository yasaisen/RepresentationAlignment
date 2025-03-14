"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2503141354
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common.utils import log_print, get_trainable_params
from ..lora.lora import ModelWithLoRA
from ..adapter.adapter import AdvancedAdapterLayer


class ImageEncoder(nn.Module):
    def __init__(self, 
        img_encoder_path, 
        encoder_dim, 
        use_logits=False, 
        dim_proj=True,
        embed_dim=512, 
        lora_config=None, 
        adapter_config=None,
        device='cuda',
        dtype=torch.float32, 
        output_dtype=torch.float32
    ):
        super().__init__()
        self.state_name = 'ImageEncoder'
        self.device = device
        self.dtype = dtype
        self.use_logits = use_logits
        self.use_lora = lora_config is not None
        self.use_adapter = adapter_config is not None
        print()
        log_print(self.state_name, f"use_logits: {self.use_logits}")
        log_print(self.state_name, f"use_lora: {self.use_lora}")
        log_print(self.state_name, f"use_adapter: {self.use_adapter}")
        log_print(self.state_name, "Building...")
        if not self.use_logits:
            self.img_encoder = torch.load(img_encoder_path)

            log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.img_encoder)}")
            for name, param in self.img_encoder.named_parameters():
                param.requires_grad = False
            log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.img_encoder)}")

            if self.use_lora:
                log_print(self.state_name, "Adding LoRA...")
                self.lora_rank = lora_config['rank']
                self.lora_alpha = lora_config['alpha']
                self.lora_dropout = lora_config['dropout']

                self.img_encoder_lora = ModelWithLoRA(self.img_encoder)
                self.img_encoder_lora.add_lora(
                    rank=self.lora_rank, 
                    alpha=self.lora_alpha, 
                    dropout=self.lora_dropout, 
                )
                log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.img_encoder_lora.base_model)}")

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.dim_proj = dim_proj
        self.output_dtype = output_dtype
        
        if self.use_adapter:
            log_print(self.state_name, "Adding Adapter...")
            self.adapter_bottleneck_dim = adapter_config['bottleneck_dim']
            self.adapter_dropout_rate = adapter_config['dropout']
            self.adapter_adapter_scaling = adapter_config['scaling']

            self.adapter = AdvancedAdapterLayer(
                input_dim=self.encoder_dim,
                bottleneck_dim=self.adapter_bottleneck_dim, 
                output_dim=self.embed_dim, 
                dropout_rate=self.adapter_dropout_rate, 
                adapter_scaling=self.adapter_adapter_scaling, 
            )
            log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.adapter)}")
        else:
            if self.dim_proj:
                self.embed_proj = nn.Linear(self.encoder_dim, self.embed_dim).to(self.device)#.to(self.dtype)
        
        self.normalize = lambda x: F.normalize(x, p=2, dim=1)
        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self)}")
        if self.use_lora:
            self.img_encoder_lora.base_model.to(self.device) # 5365MiB
        elif self.use_adapter:
            self.adapter.to(self.device)
        else:
            self.img_encoder.to(self.device)
        log_print(self.state_name, "...Done\n")
        
    def forward(self, x):
        if not self.use_logits:
            if self.use_lora:
                x = self.img_encoder_lora.base_model(x)

        if self.use_adapter:
            x = self.adapter(x)
        else:
            if self.dim_proj:
                x = self.embed_proj(x)

        normalized_features = self.normalize(x)#.to(self.output_dtype)
        return normalized_features












