import torch
import torch.nn as nn
import torchvision.models as models

class ChannelAttention(nn.Module):
    """
    Channel Attention Module as implemented in the manuscript's reference code.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module as implemented in the manuscript's reference code.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) integrating Channel and Spatial Attention.
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class DetectionHead(nn.Module):
    """
    Detection head for sperm localization as described in the end-to-end framework.
    """
    def __init__(self, in_channels):
        super(DetectionHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 5, kernel_size=1)  # 4 for bbox + 1 for objectness
        )

    def forward(self, x):
        return self.conv(x)

class ClassificationHead(nn.Module):
    """
    Classification head utilizing CBAM and Adaptive Average Pooling.
    Returns raw logits for compatibility with BitwiseValleyFocalLoss.
    """
    def __init__(self, in_channels, num_classes=1):
        super(ClassificationHead, self).__init__()
        self.cbam = CBAM(in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  # Return raw logits

class SpermNormalityModel(nn.Module):
    """
    End-to-End Sperm Normality Classification Framework.
    """
    def __init__(self, backbone_type='densenet121', pretrained=True):
        super(SpermNormalityModel, self).__init__()
        if backbone_type == 'densenet121':
            backbone = models.densenet121(pretrained=pretrained)
            self.backbone = backbone.features
            in_channels = 1024
        elif backbone_type == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            in_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        self.detection_head = DetectionHead(in_channels)
        self.classification_head = ClassificationHead(in_channels)

    def forward(self, x):
        features = self.backbone(x)
        det_out = self.detection_head(features)
        cls_out = self.classification_head(features)
        return det_out, cls_out
