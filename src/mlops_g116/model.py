import torch
from torch import nn
import torchvision.models as models


class ResNet18(nn.Module):
    """ResNet18 classifier adapted for grayscale MRI images."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        
        # Load backbone with pre-trained ImageNet weights
        self.backbone = models.resnet18(weights="DEFAULT")
        
        # Modify input layer to accept 1-channel grayscale images
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # Replace classification head
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DenseNet121(nn.Module):
    """DenseNet121 classifier adapted for grayscale MRI images."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.backbone = models.densenet121(weights="DEFAULT")
        self.backbone.features.conv0 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class TumorDetectionModelSimple(nn.Module):
    """Basic tumor detection model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() != 4:
            raise ValueError("Expected input to be a 4D tensor [N, C, H, W].")
        if x.shape[1] != 1:
            raise ValueError(f"Expected input to have 1 channel, got {x.shape[1]}.")
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

if __name__ == "__main__":
    model = ResNet18(num_classes=4)
    print(f"Model: ResNet18 | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate dummy input (Batch, Channel, Height, Width)
    dummy_input = torch.randn(1, 1, 224, 224)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
