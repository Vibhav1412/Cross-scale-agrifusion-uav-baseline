"""
Step 9.1: Explainable AI (XAI) for UAV Segmentation
Implements Class-Specific Grad-CAM for Semantic Segmentation.
Generates explanation heatmaps highlighting feature attribution for Crop and Weed predictions.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

def find_target_layer(model):
    """
    Intelligently identifies the optimal feature representation layer for Grad-CAM
    across SegFormer, ResNet18 U-Net, DeepLabV3+, and Basic CNN.
    """
    # 1. SegFormer (Vision Transformer)
    if hasattr(model, "segformer"):
        if hasattr(model.segformer, "decode_head") and hasattr(model.segformer.decode_head, "linear_fuse"):
            return model.segformer.decode_head.linear_fuse
        elif hasattr(model.segformer, "encoder") and hasattr(model.segformer.encoder, "block"):
            return model.segformer.encoder.block[-1]
            
    # 2. U-Net Architecture (prefer decoder block for higher spatial fidelity)
    if hasattr(model, "dec1"):
        return model.dec1
    if hasattr(model, "dec2"):
        return model.dec2
    if hasattr(model, "layer4"):
        return model.layer4
        
    # 3. DeepLabV3+
    if hasattr(model, "aspp"):
        return model.aspp
        
    # 4. Basic CNN
    if hasattr(model, "dec1"):
        return model.dec1
    if hasattr(model, "bottleneck"):
        return model.bottleneck
        
    # Fallback to the last convolutional or norm module
    convs = [m for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d))]
    if convs:
        return convs[-1]
    raise ValueError("Could not automatically locate a valid feature layer for Grad-CAM.")

class SegmentationGradCAM:
    """
    Universal Class-Masked Grad-CAM for Semantic Segmentation.
    Computes class-specific gradient attributions masked to target predicted boundaries,
    preventing non-plant soil pixels from diluting the attention heatmap.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or find_target_layer(model)
        
        self.gradients = None
        self.activations = None
        
        # Register forward and backward hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if isinstance(grad, tuple):
            grad = grad[0]
        self.gradients = grad

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int):
        """
        Generates focused, class-masked Grad-CAM heatmap.
        Args:
            input_tensor (Tensor): (1, 3, H, W) normalized image tensor.
            target_class (int): 1 for Crop, 2 for Weed.
        Returns:
            cam (np.ndarray): (H, W) heatmap normalized to [0, 1].
        """
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(input_tensor) # (1, num_classes, H, W)
        
        # Masked Target Score: Sum ONLY over target-class activated pixels
        preds = torch.argmax(logits, dim=1)
        target_mask = (preds == target_class)
        
        if target_mask.sum() > 0:
            target_score = (logits[:, target_class, :, :] * target_mask.float()).sum()
        else:
            # Fallback: top 1% highest confidence pixels for this class
            k = max(1, int(logits.shape[2] * logits.shape[3] * 0.01))
            flat_logits = logits[:, target_class, :, :].reshape(-1)
            top_vals, _ = torch.topk(flat_logits, k=k)
            target_score = top_vals.sum()
            
        target_score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Handle 4D feature tensors (B, C, H, W) vs 3D sequence tokens (B, N, C)
        if len(self.activations.shape) == 4:
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        elif len(self.activations.shape) == 3:
            weights = torch.mean(self.gradients, dim=1, keepdim=True)
            cam = torch.sum(weights * self.activations, dim=2, keepdim=True)
            # Reshape tokens back to 2D
            hw = int(np.sqrt(cam.shape[1]))
            cam = cam.permute(0, 2, 1).reshape(1, 1, hw, hw)
        else:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        cam = F.relu(cam) # Retain positive contributions
        
        # Upsample to input image resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Robust min-max normalization
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def cleanup(self):
        try:
            self.forward_handle.remove()
            self.backward_handle.remove()
        except Exception:
            pass

def generate_xai_visualization(model, image_tensor, rgb_display_np, output_path=None):
    """
    Generates side-by-side Class-Specific Grad-CAM visualizations for Crop and Weed.
    """
    grad_cam = SegmentationGradCAM(model)
    
    # Generate CAM for Crop (1) and Weed (2)
    crop_cam = grad_cam.generate_cam(image_tensor, target_class=1)
    weed_cam = grad_cam.generate_cam(image_tensor, target_class=2)
    grad_cam.cleanup()
    
    # Create colored heatmaps
    crop_heatmap = cv2.applyColorMap(np.uint8(255 * crop_cam), cv2.COLORMAP_JET)
    crop_heatmap = cv2.cvtColor(crop_heatmap, cv2.COLOR_BGR2RGB)
    crop_overlay = np.uint8(0.5 * rgb_display_np + 0.5 * crop_heatmap)
    
    weed_heatmap = cv2.applyColorMap(np.uint8(255 * weed_cam), cv2.COLORMAP_JET)
    weed_heatmap = cv2.cvtColor(weed_heatmap, cv2.COLOR_BGR2RGB)
    weed_overlay = np.uint8(0.5 * rgb_display_np + 0.5 * weed_heatmap)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(rgb_display_np)
        axes[0].set_title("Input UAV RGB Image")
        axes[0].axis("off")
        
        axes[1].imshow(crop_overlay)
        axes[1].set_title("Grad-CAM: Crop Class (Sugar Beet)")
        axes[1].axis("off")
        
        axes[2].imshow(weed_overlay)
        axes[2].set_title("Grad-CAM: Weed Class Attribution")
        axes[2].axis("off")
        
        diff = np.abs(crop_cam - weed_cam)
        axes[3].imshow(diff, cmap="magma")
        axes[3].set_title("Crop vs. Weed Saliency Contrast")
        axes[3].axis("off")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
    return crop_cam, weed_cam, crop_overlay, weed_overlay

