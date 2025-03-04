"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2503041417
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from ...common.utils import log_print, get_trainable_params
from ..img_encoder.modeling_img_encoder import ImageEncoder
from ..text_encoder.modeling_text_encoder import TextEncoder


class CLIPModel(nn.Module):
    def __init__(self, 
        text_encoder, 
        img_encoder, 
        batch_size, 
        temperature=0.07, 
        use_hard_pairs=True, 
        use_focal=True, 
        gamma=2.0, 
        class_weights=None,
        device='cuda', 
        use_logits=False, 
    ):
        super().__init__()
        self.state_name = 'CLIPModel'
        self.device = device
        print()
        log_print(self.state_name, "Building...")

        self.use_logits = use_logits
        self.text_encoder = text_encoder
        self.img_encoder = img_encoder

        self.batch_size = batch_size
        self.temperature = temperature
        self.use_hard_pairs = use_hard_pairs
        self.use_focal = use_focal
        self.gamma = gamma
        self.class_weights = class_weights
        
        # 建立對角GT矩陣
        self.labels = torch.arange(self.batch_size).long()
        
        # 雜訊對比估計器相關參數 (InfoNCE)
        self.register_buffer("mask", (~torch.eye(self.batch_size, self.batch_size, dtype=bool)).float())


        log_print(self.state_name, f"using_logits: {self.use_logits}")
        log_print(self.state_name, f"basemodel trainable params: {get_trainable_params(self)}")
        log_print(self.state_name, "...Done\n")
        
    def forward(self, image, text, output_features=False):
        text_features = self.text_encoder(text)
        image_features = self.img_encoder(image)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 計算相似度矩陣
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # 獲取正例和負例
        positive_logits = torch.diag(logits)
        
        # 硬負例挖掘 (Hard Negative Mining)
        if self.use_hard_pairs:
            with torch.no_grad():
                # 找到每個樣本最容易混淆的負例
                mask = self.mask.to(logits.device)
                max_image_text_sim = (logits * mask).max(dim=1)[0]
                max_text_image_sim = (logits.t() * mask).max(dim=1)[0]
                
            # 權重計算：更關注難以區分的負例
            image_weights = F.softmax(max_image_text_sim, dim=0)
            text_weights = F.softmax(max_text_image_sim, dim=0)
            
            # Weighted Image-Text Contrastive Loss
            image_to_text_loss = -torch.sum(image_weights * torch.log(
                torch.exp(positive_logits) / 
                torch.sum(torch.exp(logits) * mask, dim=1)
            ))
            
            # Weighted Text-Image Contrastive Loss
            text_to_image_loss = -torch.sum(text_weights * torch.log(
                torch.exp(positive_logits) / 
                torch.sum(torch.exp(logits.t()) * mask, dim=1)
            ))
            
            loss = (image_to_text_loss + text_to_image_loss) / 2
            
        else:
            # 標準InfoNCE損失
            labels = self.labels.to(logits.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
            loss = (loss_i2t + loss_t2i) / 2
        
        # 焦點損失變體 (Focal Loss Variant)
        if self.use_focal:
            i2t_probs = F.softmax(logits, dim=1)
            t2i_probs = F.softmax(logits.t(), dim=1)
            
            # 對角線上的元素是正例概率
            pos_i2t_probs = torch.diag(i2t_probs)
            pos_t2i_probs = torch.diag(t2i_probs)
            
            # 計算焦點損失權重
            focal_weights_i2t = (1 - pos_i2t_probs) ** self.gamma
            focal_weights_t2i = (1 - pos_t2i_probs) ** self.gamma
            
            # 應用焦點損失
            focal_loss_i2t = -torch.mean(focal_weights_i2t * torch.log(pos_i2t_probs + 1e-8))
            focal_loss_t2i = -torch.mean(focal_weights_t2i * torch.log(pos_t2i_probs + 1e-8))
            
            focal_loss = (focal_loss_i2t + focal_loss_t2i) / 2
            
            # 結合標準損失和焦點損失
            loss = 0.5 * loss + 0.5 * focal_loss
        
        if output_features:
            return loss, image_features, text_features
        else:
            return loss

    @classmethod
    def from_config(cls, cfg):
        root_path = cfg['task'].get("root_path")
        device = str(cfg['task'].get("device"))
        batch_size = int(cfg['task'].get("batch_size"))

        clip_cfg = cfg['model']['clip_model']
        temperature = float(clip_cfg.get("temperature"))
        use_logits = bool(clip_cfg.get("use_logits"))
        embed_dim = int(clip_cfg.get("embed_dim"))
        use_hard_pairs = bool(clip_cfg.get("use_hard_pairs"))
        use_focal_loss = bool(clip_cfg.get("use_focal_loss"))
        gamma = float(clip_cfg.get("gamma"))

        if cfg['model'].get('text_encoder') is not None:
            text_cfg = cfg['model']['text_encoder']
            text_model_path = os.path.join(root_path, text_cfg.get("text_model_path"))
            tokenizer_path = os.path.join(root_path, text_cfg.get("tokenizer_path"))
            text_encoder_dim = int(text_cfg.get("text_encoder_dim"))
            text_dim_proj = bool(text_cfg.get("text_dim_proj"))

            text_encoder = TextEncoder(
                text_model_path, 
                tokenizer_path, 
                text_encoder_dim, 
                use_logits=use_logits, 
                embed_dim=embed_dim, 
                dim_proj=text_dim_proj, 
                dtype=torch.float32, 
                output_dtype=torch.float32, 
                device=device
            )

        if cfg['model'].get('img_encoder') is not None:
            img_cfg = cfg['model']['img_encoder']
            img_model_path = os.path.join(root_path, img_cfg.get("img_model_path"))
            img_encoder_dim = int(img_cfg.get("img_encoder_dim"))
            img_dim_proj = bool(img_cfg.get("img_dim_proj"))
            img_lora_rank = int(img_cfg.get("rank"))
            img_lora_alpha = float(img_cfg.get("alpha"))
            img_lora_dropout = float(img_cfg.get("dropout"))

            img_encoder = ImageEncoder(
                img_model_path, 
                img_encoder_dim, 
                embed_dim=embed_dim, 
                dim_proj=img_dim_proj, 
                rank=img_lora_rank, 
                alpha=img_lora_alpha, 
                dropout=img_lora_dropout,
                device=device
            )

        if cfg['model'].get('voice_encoder') is not None:
            pass

        model = cls(
            text_encoder=text_encoder, 
            img_encoder=img_encoder, 
            batch_size=batch_size, 
            temperature=temperature, 
            use_hard_pairs=use_hard_pairs, 
            use_focal=use_focal_loss, 
            gamma=gamma, 
            class_weights=None,
            device=device, 
            use_logits=use_logits, 
        )
        return model
    











    