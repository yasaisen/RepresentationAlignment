"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2502251532
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from ...common.utils import log_print, get_trainable_params


class TextEncoder(nn.Module):
    def __init__(self, 
        text_encoder_path, 
        tokenizer_path,
        encoder_dim, 
        dim_proj=False,
        embed_dim=512, 
        device='cuda',
        dtype=torch.float16, 
        output_dtype=torch.float16
    ):
        super().__init__()
        self.state_name = 'TextEncoder'
        self.device = device
        self.dtype = dtype
        print()
        log_print(self.state_name, "Building...")
        self.text_encoder = torch.load(text_encoder_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.text_encoder)}")
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.text_encoder)}")

        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.dim_proj = dim_proj
        self.output_dtype = output_dtype

        self.text_encoder.to(self.device) # 16361MiB

        if self.dim_proj:
            self.embed_proj = nn.Linear(self.encoder_dim , self.embed_dim).to(self.device).to(self.dtype)
            log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.text_encoder) + get_trainable_params(self.embed_proj)}")
        
        self.normalize = lambda x: F.normalize(x, p=2, dim=1)
        log_print(self.state_name, "...Done\n")
        
    def forward(self, input_ids):
        input_ids = torch.cat([input_ids, torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)], dim=-1)
        
        x = self.text_encoder(input_ids, output_hidden_states=True)
        x = x.hidden_states[-1][:, -1, :]
        if self.dim_proj:
            x = self.embed_proj(x)
        
        normalized_features = self.normalize(x).to(self.output_dtype)
        return normalized_features
    











