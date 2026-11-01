"""
Step 4.2: Data Augmentation Pipeline
Applies synchronized spatial transformations to RGB image and mask pairs.
Applies radiometric transformations (brightness, contrast) strictly to images.
Ensures nearest-neighbor interpolation for integer segmentation masks.
"""

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

class SynchronizedAugmentation:
    """
    Synchronized augmentation pipeline for segmentation pairs.
    Applies identical spatial transformations to image and mask,
    with nearest-neighbor interpolation for the mask.
    """
    def __init__(
        self,
        output_size=(256, 256),
        is_train=True,
        hflip_prob=0.5,
        vflip_prob=0.5,
        rotate_prob=0.5,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ):
        self.output_size = output_size
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_prob = rotate_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image, mask: Image.Image):
        """
        Args:
            image (PIL.Image): RGB image (3 channels)
            mask (PIL.Image): Single-channel integer label mask (dtype uint8)
        Returns:
            image_tensor (torch.FloatTensor): Normalized tensor of shape (3, H, W)
            mask_tensor (torch.LongTensor): Integer tensor of shape (H, W) with classes in {0, 1, 2}
        """
        # Ensure correct formats
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image.astype(np.uint8))
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask.astype(np.uint8))

        if self.is_train:
            # 1. Random Crop or Resize
            # WeedMap tiles are 480x360. For training, extract a random 256x256 crop
            w, h = image.size
            out_h, out_w = self.output_size
            
            if w >= out_w and h >= out_h:
                # Random crop
                i = random.randint(0, h - out_h)
                j = random.randint(0, w - out_w)
                image = TF.crop(image, i, j, out_h, out_w)
                mask = TF.crop(mask, i, j, out_h, out_w)
            else:
                image = TF.resize(image, self.output_size, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.resize(mask, self.output_size, interpolation=TF.InterpolationMode.NEAREST)

            # 2. Horizontal Flip
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # 3. Vertical Flip
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # 4. Random 90-degree Rotations (orthogonal to maintain pixel alignment)
            if random.random() < self.rotate_prob:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            # 5. Photometric augmentations (applied ONLY to image, NEVER to mask)
            if self.brightness_range:
                factor = random.uniform(*self.brightness_range)
                image = TF.adjust_brightness(image, factor)
                
            if self.contrast_range:
                factor = random.uniform(*self.contrast_range)
                image = TF.adjust_contrast(image, factor)

        else:
            # Validation / Testing: Deterministic Center Crop or Resize
            w, h = image.size
            out_h, out_w = self.output_size
            if w >= out_w and h >= out_h:
                image = TF.center_crop(image, (out_h, out_w))
                mask = TF.center_crop(mask, (out_h, out_w))
            else:
                image = TF.resize(image, self.output_size, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.resize(mask, self.output_size, interpolation=TF.InterpolationMode.NEAREST)

        # Convert to Tensors
        image_tensor = TF.to_tensor(image) # (3, H, W) in [0, 1]
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)
        
        # Mask: convert directly to LongTensor, preserving exact integer labels
        mask_np = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_np).long()

        return image_tensor, mask_tensor

