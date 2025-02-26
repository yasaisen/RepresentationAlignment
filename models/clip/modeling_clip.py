"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2502261412
"""

import torch
import torch.nn as nn
import numpy as np
import os

from ...common.utils import log_print, get_trainable_params
from ..img_encoder.modeling_img_encoder import ImageEncoder
from ..text_encoder.modeling_text_encoder import TextEncoder


class CLIPModel(nn.Module):
    def __init__(self, 
        text_model_path, 
        tokenizer_path, 
        img_model_path, 
        temperature=0.07, 
        text_encoder_dim=4096, 
        img_encoder_dim=1536, 
        text_dim_proj=False, 
        img_dim_proj=True, 
        img_lora_rank=8, 
        img_lora_alpha=1.0, 
        img_lora_dropout=0.1,
        device='cuda', 
        use_logits=False, 
    ):
        super().__init__()
        self.state_name = 'CLIPModel'
        self.device = device
        print()
        log_print(self.state_name, "Building...")

        self.use_logits = use_logits
        log_print(self.state_name, f"using_logits: {self.use_logits}")
        
        if not self.use_logits:
            self.text_encoder = TextEncoder(
                text_model_path, 
                tokenizer_path, 
                text_encoder_dim, 
                dim_proj=text_dim_proj, 
                output_dtype=torch.float32, 
                device=self.device
            )
        self.img_encoder = ImageEncoder(
            img_model_path, 
            img_encoder_dim, 
            embed_dim=text_encoder_dim, 
            dim_proj=img_dim_proj, 
            rank=img_lora_rank, 
            alpha=img_lora_alpha, 
            dropout=img_lora_dropout,
            device=self.device
        )

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

        if not self.use_logits:
            log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.text_encoder) + get_trainable_params(self.img_encoder)}")
        else:
            log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self.img_encoder)}")
        log_print(self.state_name, "...Done\n")
        
    def forward(self, batch):
        if self.use_logits:
            text_features = batch['text_logits']
        else:
            text_features = self.text_encoder(batch['caption_ids'])
        image_features = self.img_encoder(batch['image'])
        
        logits_per_image = torch.matmul(image_features, text_features.t()) * self.temperature.exp()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

    @classmethod
    def from_config(cls, cfg):
        root_path = cfg['task'].get("root_path")
        device = str(cfg['task'].get("device"))

        clip_cfg = cfg['model']['clip_model']
        temperature = float(clip_cfg.get("temperature"))
        use_logits = bool(clip_cfg.get("use_logits"))

        if cfg['model']['text_encoder'] is not None and not use_logits:
            text_cfg = cfg['model']['text_encoder']
            text_model_path = os.path.join(root_path, text_cfg.get("text_model_path"))
            tokenizer_path = os.path.join(root_path, text_cfg.get("tokenizer_path"))
            text_encoder_dim = int(text_cfg.get("text_encoder_dim"))
            text_dim_proj = bool(text_cfg.get("text_dim_proj"))

        if cfg['model']['img_encoder'] is not None:
            img_cfg = cfg['model']['img_encoder']
            img_model_path = os.path.join(root_path, img_cfg.get("img_model_path"))
            img_encoder_dim = int(img_cfg.get("img_encoder_dim"))
            img_dim_proj = bool(img_cfg.get("img_dim_proj"))
            img_lora_rank = int(img_cfg.get("rank"))
            img_lora_alpha = float(img_cfg.get("alpha"))
            img_lora_dropout = float(img_cfg.get("dropout"))

        model = cls(
        text_model_path=text_model_path, 
        tokenizer_path=tokenizer_path, 
        img_model_path=img_model_path, 
        temperature=temperature, 
        text_encoder_dim=text_encoder_dim, 
        img_encoder_dim=img_encoder_dim, 
        text_dim_proj=text_dim_proj, 
        img_dim_proj=img_dim_proj, 
        img_lora_rank=img_lora_rank, 
        img_lora_alpha=img_lora_alpha, 
        img_lora_dropout=img_lora_dropout,
        device=device, 
        use_logits=use_logits, 
        )

        return model










