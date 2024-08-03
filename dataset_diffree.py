from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from tqdm import tqdm
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        self.annotation_path = os.path.join(self.path, "annotations.json")
        if not os.path.exists(self.annotation_path):
            raise FileNotFoundError(f"Annotation file not found at {self.annotation_path}")
        with open(self.annotation_path) as f:
            annotations = json.load(f)
        
        original_dir_path = os.path.join(self.path, "original_images")
        inpainted_dir_path = os.path.join(self.path, "inpainted_images")
        mask_dir_path = os.path.join(self.path, "mask_images")
        self.dataset = []
        for annotation in tqdm(annotations):
            original_image_path = os.path.join(original_dir_path, f'{annotation["image_id"]}.jpg')
            inpainted_image_path = os.path.join(inpainted_dir_path, annotation["image_id"], f'{annotation["mask_id"]}.jpg')
            mask_image_path = os.path.join(mask_dir_path, annotation["image_id"], f'{annotation["mask_id"]}.png')
            category_name = annotation["category_name"]
            
            self.dataset.append((original_image_path, inpainted_image_path, mask_image_path, category_name))
        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.dataset))
        idx_1 = math.floor(split_1 * len(self.dataset))
        self.dataset = self.dataset[idx_0:idx_1]
        
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int) -> dict[str, Any]:
        original_image_path, inpainted_image_path, mask_image_path, category_name = self.dataset[i]
        
        prompt = category_name
        
        inpainted_image = Image.open(inpainted_image_path).convert("RGB")
        original_image = Image.open(original_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("L")

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        inpainted_image = inpainted_image.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        original_image = original_image.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        mask_image = mask_image.resize((reize_res, reize_res), Image.Resampling.NEAREST)
        
        inpainted_image = rearrange(2 * torch.tensor(np.array(inpainted_image)).float() / 255 - 1, "h w c -> c h w")
        original_image = rearrange(2 * torch.tensor(np.array(original_image)).float() / 255 - 1, "h w c -> c h w")
        
        mask_image = torch.tensor(np.array(mask_image) / 255).int().unsqueeze(0)
        mask_image = mask_image.repeat(3, 1, 1)
        
        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        inpainted_image, original_image, mask_image = flip(crop(torch.cat((inpainted_image, original_image, mask_image)))).chunk(3)

        mask_image = mask_image[0].unsqueeze(0)
    
        return dict(edited=original_image, mask=mask_image, edit=dict(c_concat=inpainted_image, c_crossattn=prompt))

