"""
Step 4.3: PyTorch Dataset & DataLoader
Provides WeedMapDataset class for paired RGB images and semantic masks,
with verified label integrity and DataLoader construction.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from src.augmentation import SynchronizedAugmentation

class WeedMapDataset(Dataset):
    """
    PyTorch Dataset for WeedMap UAV crop-weed semantic segmentation.
    Loads paired RGB images and 16-bit iMap masks with 3 semantic classes:
      0: Background / Soil
      1: Crop (Sugar Beet)
      2: Weed
    """
    def __init__(
        self,
        split_csv: str,
        output_size=(256, 256),
        is_train=True,
        transform=None
    ):
        super().__init__()
        if not os.path.exists(split_csv):
            raise FileNotFoundError(f"Split CSV not found: {split_csv}")
            
        self.df = pd.read_csv(split_csv)
        self.output_size = output_size
        self.is_train = is_train
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = SynchronizedAugmentation(
                output_size=output_size,
                is_train=is_train
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        rgb_path = row["rgb_path"]
        imap_path = row["imap_path"]
        
        # 1. Load RGB Image (3 channels)
        image = Image.open(rgb_path).convert("RGB")
        
        # 2. Load 16-bit iMap label
        raw_imap = np.array(Image.open(imap_path))
        
        # 3. Clean labels:
        # 10000 -> 1 (Crop boundaries/instances)
        # > 2   -> 0 (Background)
        cleaned_mask = np.where(raw_imap == 10000, 1, raw_imap)
        cleaned_mask = np.where(cleaned_mask > 2, 0, cleaned_mask).astype(np.uint8)
        mask = Image.fromarray(cleaned_mask)
        
        # 4. Apply synchronized spatial & photometric transforms
        image_tensor, mask_tensor = self.transform(image, mask)
        
        metadata = {
            "dataset_id": row["dataset_id"],
            "frame_id": row["frame_id"],
            "rgb_path": rgb_path,
            "valid_ratio": float(row["valid_ratio"])
        }
        
        return image_tensor, mask_tensor, metadata

def get_dataloaders(
    splits_dir="data/splits",
    batch_size=8,
    image_size=(256, 256),
    num_workers=0
):
    """
    Factory function to construct Train, Val, and Test DataLoaders.
    """
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")
    
    train_dataset = WeedMapDataset(train_csv, output_size=image_size, is_train=True)
    val_dataset = WeedMapDataset(val_csv, output_size=image_size, is_train=False)
    test_dataset = WeedMapDataset(test_csv, output_size=image_size, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader
