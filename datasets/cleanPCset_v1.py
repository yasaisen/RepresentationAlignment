"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2503141556
"""

import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..common.utils import log_print


class cleanPCset_v1_Dataset(Dataset):
    def __init__(self, 
        meta_list, 
        data_path, 
        split,
        img_processor=None, 
        text_processor=None, 
        img_logits_path=None,
        text_logits_path=None,
        device='cuda',
    ):
        self.state_name = 'cleanPCset_v1_Dataset'
        self.device = device
        print()
        log_print(self.state_name, "Building...")

        self.data_path = data_path
        self.img_processor = img_processor
        self.text_processor = text_processor
        self.img_logits_path = img_logits_path
        self.text_logits_path = text_logits_path
        self.split = split

        self.meta_list = []
        for single_data_dict in tqdm(meta_list):
            if single_data_dict['split'] == split:
                self.meta_list += [single_data_dict]

        self.use_img_logits = self.img_logits_path is not None
        self.use_text_logits = self.text_logits_path is not None
        log_print(self.state_name, f"using img_logits: {self.use_img_logits}")
        log_print(self.state_name, f"using text_logits: {self.use_text_logits}")
        log_print(self.state_name, f"using size: {self.img_processor.size}")
        log_print(self.state_name, f"using split: {self.split}")
        log_print(self.state_name, f"data len: {len(self.meta_list)}")
        log_print(self.state_name, "...Done\n")
        
    def __len__(self):
        return len(self.meta_list)
    
    def __getitem__(self, idx):

        global_idx = self.meta_list[idx]['global_idx']

        if self.use_text_logits:
            text_logit_file_name = 'text_logits_' + str(global_idx).zfill(7) + '.pt'
            text_logit_file_path = os.path.join(self.data_path, self.text_logits_path, text_logit_file_name)
            text_logits = torch.load(text_logit_file_path).squeeze(0).to(torch.float32).to(self.device)
        else:
            input_caption = self.meta_list[idx]['caption']
            input_caption_ids = self.text_processor(input_caption).to(self.device)

        if self.use_img_logits:
            img_logit_file_name = 'img_logits_' + str(global_idx).zfill(7) + '.pt'
            img_logit_file_path = os.path.join(self.data_path, self.img_logits_path, img_logit_file_name)
            image_logits = torch.load(img_logit_file_path).squeeze(0).to(torch.float32).to(self.device)
        else:
            image_path = os.path.join(self.data_path, self.meta_list[idx]['img_filepath'])
            input_image = self.img_processor(image_path).to(self.device)

        if self.use_img_logits and self.use_text_logits:
            return {
                'image_logits': image_logits,
                'text_logits': text_logits,
                'idx': global_idx
            }
        elif self.use_img_logits:
            return {
                'image_logits': image_logits,
                'caption_ids': input_caption_ids,
                'idx': global_idx
            }
        elif self.use_text_logits:
            return {
                'image': input_image,
                'text_logits': text_logits,
                'idx': global_idx
            }
        else:
            return {
                'image': input_image,
                'caption_ids': input_caption_ids,
                'idx': global_idx
            }

        











