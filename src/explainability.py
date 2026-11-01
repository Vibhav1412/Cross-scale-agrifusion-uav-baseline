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

class SegmentationGradCAM:
    """
    Grad-CAM implementation tailored for PyTorch Semantic Segmentation models.
    Hooks into the final encoder layer (layer4) to capture spatial semantic activations.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer or model.layer4
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int):
        """
        Generates Grad-CAM heatmap for a specific target class.
        Args:
            input_tensor (Tensor): (1, 3, H, W) normalized image tensor.
            target_class (int): 1 for Crop, 2 for Weed.
        Returns:
            cam (np.ndarray): (H, W) heatmap in range [0, 1].
        """
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(input_tensor) # (1, num_classes, H, W)
        
        # Target score: sum of logits corresponding to target class pixels
        target_score = logits[:, target_class, :, :].sum()
        target_score.backward(retain_graph=True)
        
        # Global Average Pooling of gradients to obtain neuron importance weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # (1, C, 1, 1)
        
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # (1, 1, H', W')
        cam = F.relu(cam) # Apply ReLU to focus on positive contributions
        
        # Upsample to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def cleanup(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

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

