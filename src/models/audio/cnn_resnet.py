# src/models/audio/cnn_resnet.py
import torch
import torch.nn as nn
import torchvision.models as tvm

class ResNet18(nn.Module):
    """
    ResNet18 backbone + FC classifier for SER (speech-only).
    Expect input tensor shape: [B, 1, freq_bins, time_steps]
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # ① Reuse torchvision.resnet18, but change the first layer of conv to single channel.
        self.backbone = tvm.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # ② Remove the last FC of backbone (take global-avg-pool output)
        self.backbone.fc = nn.Identity()
        # ③ Own classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)          # [B, 512]
        logits = self.classifier(feats)   # [B, n_classes]
        return logits
