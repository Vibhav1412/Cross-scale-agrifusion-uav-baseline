"""
Comparison Model 2: DeepLabV3+ Semantic Segmentation Architecture
Features:
- ResNet18 Backbone.
- Atrous Spatial Pyramid Pooling (ASPP) with multiple dilation rates for multi-scale context.
- High-level and Low-level feature fusion decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with rates [1, 6, 12, 18]"""
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ]
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with ResNet18 Backbone for UAV Crop-Weed Segmentation.
    """
    def __init__(self, num_classes=3, in_channels=3, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        try:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
        except Exception:
            backbone = resnet18(weights=None)
            
        self.initial = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1 # Low-level features: /4, 64 channels
        self.layer2 = backbone.layer2 # /8, 128 channels
        self.layer3 = backbone.layer3 # /16, 256 channels
        self.layer4 = backbone.layer4 # High-level features: /32, 512 channels
        
        # ASPP module on high-level features
        self.aspp = ASPP(in_channels=512, out_channels=256)
        
        # Low-level feature projection
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Encoder
        x = self.initial(x)
        low_level_feat = self.layer1(x) # (B, 64, H/4, W/4)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        high_level_feat = self.layer4(x) # (B, 512, H/32, W/32)
        
        # ASPP
        aspp_out = self.aspp(high_level_feat) # (B, 256, H/32, W/32)
        aspp_up = F.interpolate(aspp_out, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)
        
        # Low level projection
        low_proj = self.low_level_proj(low_level_feat) # (B, 48, H/4, W/4)
        
        # Concatenate and decode
        fused = torch.cat([aspp_up, low_proj], dim=1) # (B, 304, H/4, W/4)
        logits_low = self.decoder(fused)
        
        # Upsample to full input resolution
        logits = F.interpolate(logits_low, size=input_size, mode='bilinear', align_corners=False)
        return logits

def build_deeplabv3plus(num_classes=3, in_channels=3, pretrained=True):
    return DeepLabV3Plus(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)

