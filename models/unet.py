"""
Primary Model: ResNet18-UNet Semantic Segmentation Architecture
Features:
- ResNet18 encoder with pre-trained ImageNet feature extraction.
- Skip connections preserving high-frequency boundary details.
- Modular decoder blocks with bilinear upsampling and double convolutions.
- Cross-Scale Fusion Hook: allows injecting satellite feature embeddings via residual modulation.
- Monte Carlo Dropout support for predictive uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ConvBlock(nn.Module):
    """Double Convolution Block: (Conv2d -> BatchNorm -> ReLU) x 2"""
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class CrossScaleFusionAdapter(nn.Module):
    """
    Modular Adapter for Phase-II Cross-Scale UAV-Satellite Fusion.
    Projects arbitrary satellite feature channels to match UAV decoder channels,
    aligns spatial resolution, and performs residual feature modulation.
    """
    def __init__(self, sat_channels, uav_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(sat_channels, uav_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(uav_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, uav_feat, sat_feat):
        if sat_feat.shape[2:] != uav_feat.shape[2:]:
            sat_feat = F.interpolate(sat_feat, size=uav_feat.shape[2:], mode="bilinear", align_corners=True)
        sat_proj = self.proj(sat_feat)
        return uav_feat + sat_proj

class DecoderBlock(nn.Module):
    """
    Decoder Block:
    Upsamples input, concatenates skip connection from encoder,
    and applies ConvBlock.
    """
    def __init__(self, in_channels, skip_channels, out_channels, dropout_prob=0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout_prob=dropout_prob)

    def forward(self, x, skip=None):
        # Bilinear upsample by factor of 2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
            
        return self.conv(x)

class ResNetUNet(nn.Module):
    """
    U-Net with ResNet18 Backbone for UAV Crop-Weed Segmentation.
    3-class output: 0=Soil/Background, 1=Crop, 2=Weed.
    """
    def __init__(
        self,
        num_classes=3,
        in_channels=3,
        pretrained=True,
        dropout_prob=0.1,
        satellite_aux_channels=16 # Configurable for Phase-II
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # 1. ResNet18 Encoder
        try:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base_model = resnet18(weights=weights)
        except Exception:
            base_model = resnet18(weights=None)

        if in_channels != 3:
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                base_model.bn1,
                base_model.relu
            )
        else:
            self.initial = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu
            )

        # Encoder stages
        self.maxpool = base_model.maxpool      # /4,  64 channels
        self.layer1 = base_model.layer1        # /4,  64 channels
        self.layer2 = base_model.layer2        # /8,  128 channels
        self.layer3 = base_model.layer3        # /16, 256 channels
        self.layer4 = base_model.layer4        # /32, 512 channels

        # 2. Decoder Stages
        self.dec4 = DecoderBlock(512, 256, 256, dropout_prob=dropout_prob)
        self.dec3 = DecoderBlock(256, 128, 128, dropout_prob=dropout_prob)
        self.dec2 = DecoderBlock(128, 64, 64, dropout_prob=dropout_prob)
        self.dec1 = DecoderBlock(64, 64, 32, dropout_prob=dropout_prob)

        # Phase-II Modular Cross-Scale Fusion Adapter
        self.fusion_adapter = CrossScaleFusionAdapter(satellite_aux_channels, 32)

        # Final upsample to original resolution (H, W)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 3. Final 1x1 Segmentation Head
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x, auxiliary_satellite_feat=None):
        """
        Forward pass.
        Args:
            x (Tensor): Input UAV image tensor of shape (B, 3, H, W).
            auxiliary_satellite_feat (Tensor, optional): Cross-scale feature tensor for Phase-II fusion.
        Returns:
            logits (Tensor): Raw logits of shape (B, num_classes, H, W).
        """
        # Encoder forward pass
        s0 = self.initial(x)         # (B, 64, H/2, W/2)
        s1 = self.maxpool(s0)        # (B, 64, H/4, W/4)
        s1 = self.layer1(s1)         # (B, 64, H/4, W/4)
        s2 = self.layer2(s1)         # (B, 128, H/8, W/8)
        s3 = self.layer3(s2)         # (B, 256, H/16, W/16)
        s4 = self.layer4(s3)         # (B, 512, H/32, W/32)

        # Decoder forward pass with skip connections
        d4 = self.dec4(s4, s3)       # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, s2)       # (B, 128, H/8, W/8)
        d2 = self.dec2(d3, s1)       # (B, 64, H/4, W/4)
        d1 = self.dec1(d2, s0)       # (B, 32, H/2, W/2)

        # Phase-II Fusion Hook (if satellite feature map is supplied)
        if auxiliary_satellite_feat is not None:
            d1 = self.fusion_adapter(d1, auxiliary_satellite_feat)

        # Final full-resolution projection
        out = self.final_upsample(d1) # (B, 32, H, W)
        logits = self.classifier(out)  # (B, num_classes, H, W)

        return logits

    def enable_mc_dropout(self):
        """Enable Monte Carlo dropout during inference for uncertainty estimation."""
        for m in self.modules():
            if isinstance(m, nn.Dropout2d):
                m.train()

def build_unet(num_classes=3, in_channels=3, pretrained=True, dropout_prob=0.1, satellite_aux_channels=16):
    return ResNetUNet(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
        dropout_prob=dropout_prob,
        satellite_aux_channels=satellite_aux_channels
    )
