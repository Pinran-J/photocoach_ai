import torch.nn as nn
from torchvision import models

# ----------------------------
# Model
# ----------------------------
class AestheticResNet50(nn.Module):
    def __init__(self, pretrained=True, dropout_p=0.3):
        super().__init__()
        # Try modern torchvision weights API if available
        try:
            # torchvision >= 0.13
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
                self.backbone = models.resnet50(weights=weights)
            else:
                self.backbone = models.resnet50(weights=None)
        except Exception:
            # Fallback older API
            self.backbone = models.resnet50(pretrained=pretrained)

        num_ftrs = self.backbone.fc.in_features
        # Replace classifier head with a small MLP -> logits (no softmax here)
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 10)   # logits for 10 classes (1..10)
        )

    def forward(self, x):
        logits = self.backbone(x)
        return logits
