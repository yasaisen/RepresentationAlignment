"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2502251532
"""

import torch


class TextProcessor():
    def __init__(self, tokenizer, device='cuda'):
        self.tokenizer = tokenizer
        self.device = device

        self.IMAGE_TOKEN_INDEX = -200
        self.DEFAULT_IMAGE_TOKEN = "<image>"

    def tokenizer_image_token(self, prompt, tokenizer, return_tensors=None):
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(self.DEFAULT_IMAGE_TOKEN)]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [self.IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    def __call__(self, prompt):
        input_ids = (
            self.tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
            # .unsqueeze(0)
            .to(self.device)
        )
        return input_ids












