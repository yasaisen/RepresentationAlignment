"""
 Copyright (c) 2025, yasaisen.
 All rights reserved.

 last modified in 2502251532
"""

import cv2
import numpy as np
import random
import albumentations as A
from torchvision import transforms
import torch.nn.functional as F


class ImgProcessor():
    def __init__(self, size=None, device='cuda'):
        self.device = device
        self.size = size

        self.augmentation_pipeline = A.Compose([
            # 空間變換：翻轉與旋轉（旋轉角度限制在±15度內）
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.5),

            # 亮度、對比與色彩調整：輕微調整，避免過強變化
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.3),

            # 噪聲模擬：加入輕微高斯噪聲
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.1), 

            # 輕微的彈性變形：僅作輕微形變，避免破壞關鍵結構
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.1), 

            # 仿真染色偽影
            A.CoarseDropout(max_holes=3, max_height=20, max_width=20, fill_value=random.randint(100, 200), p=0.1)
        ])
        self.pre_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=(0.707223, 0.578729, 0.703617), 
            #     std=(0.211883, 0.230117, 0.177517)
            # ),
        ])
        
    @staticmethod
    def macenko_normalization(I, Io=240, alpha=1, beta=0.15, target_max=None):
        """
        使用 Macenko 方法進行染色歸一化
        參數:
        I: 輸入 RGB 影像 (uint8, 範圍 [0, 255])
        Io: 背景光強度 (例如 240 或 255)
        alpha: 百分位數，用於估計 stain 向量的極值 (建議 1)
        beta: OD 閾值，用於過濾背景 (建議 0.15)
        target_max: 目標 stain 濃度 (2元素陣列)，若未提供則以輸入影像本身為準
        返回:
        標準化後的影像 (uint8)
        """
        # 1. 將影像轉為 float32 並轉換到光密度 (OD) 空間
        I = I.astype(np.float32)
        # 加 1 是為避免 log(0) 問題
        OD = -np.log((I + 1) / Io)
        
        # 2. 過濾掉背景區域：選取所有通道值均低於 Io 的像素
        mask = (I < Io).all(axis=2)
        # 將符合條件的 pixel 拉成 (N,3) 的陣列
        OD_hat = OD[mask].reshape(-1, 3)
        # 只保留最大 OD 超過 beta 的像素
        OD_hat = OD_hat[np.max(OD_hat, axis=1) > beta]
        
        # 若沒有足夠的有效 pixel，則直接返回原影像
        if OD_hat.shape[0] == 0:
            return I.astype(np.uint8)
        
        # 3. 使用 SVD 來估計主要 stain 向量
        U, S, V = np.linalg.svd(OD_hat, full_matrices=False)
        # 取前兩個主成分（每個主成分代表一個 stain）
        V = V[:2, :].T  # 形狀 (3, 2)
        
        # 4. 將 OD_hat 投影到 2 維平面上，計算每個像素的角度
        That = np.dot(OD_hat, V)
        phi = np.arctan2(That[:, 1], That[:, 0])
        
        # 取 alpha 與 (100-alpha) 百分位數作為極值
        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)
        
        # 取得對應 stain 向量，將極值角度映射回原空間
        v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
        v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        
        # 組成 stain 矩陣，每一欄為一個 stain 向量，並正規化成單位向量
        stain_matrix = np.array([v1, v2]).T
        stain_matrix /= np.linalg.norm(stain_matrix, axis=0)
        
        # 5. 對整張影像（所有 pixel）計算 stain 濃度
        # 將整張影像的 OD 轉換成 (3, N) 形式，其中 N = H*W
        OD_flat = OD.reshape((-1, 3)).T
        # 解方程式 OD = stain_matrix * concentrations
        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, OD_flat, rcond=None)
        
        # 6. 使用 99th 百分位數估計濃度上限，若未提供 target_max 則使用本影像數值
        maxC = np.percentile(concentrations, 99, axis=1)
        if target_max is None:
            target_max = maxC
        
        # 7. 正規化濃度，並重構標準化後的 OD
        norm_concentrations = concentrations * (target_max[:, None] / maxC[:, None])
        OD_normalized = np.dot(stain_matrix, norm_concentrations)
        
        # 8. 將標準化後的 OD 轉回 RGB 空間
        I_normalized = Io * np.exp(-OD_normalized)
        I_normalized = I_normalized.T.reshape(I.shape)
        I_normalized = np.clip(I_normalized, 0, 255).astype(np.uint8)
        
        return I_normalized
    
    def __call__(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # augmentation
        aug_image = self.macenko_normalization(image)
        aug_image = self.augmentation_pipeline(image=aug_image)
        aug_image = transforms.ToPILImage()(aug_image['image'])

        # to tensor
        input_image = self.pre_transform(aug_image).to(self.device) # .unsqueeze(0)
        if self.size is not None:
            input_image = F.interpolate(input_image.unsqueeze(0), size=self.size, mode='bicubic', align_corners=False).squeeze(0)

        return input_image
    











