import torch
from torch import nn
import torchvision.models as models
import pytorch_lightning as pl


class BaseLightningClassifier(pl.LightningModule):
    """Base Lightning classifier with shared training and validation steps."""

    def __init__(self, num_classes: int = 4, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class ResNet18(BaseLightningClassifier):
    """ResNet18 classifier adapted for grayscale MRI images."""

    def __init__(self, num_classes: int = 4, lr: float = 1e-3) -> None:
        super().__init__(num_classes=num_classes, lr=lr)
        self.backbone = models.resnet18(weights="DEFAULT")
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DenseNet121(BaseLightningClassifier):
    """DenseNet121 classifier adapted for grayscale MRI images."""

    def __init__(self, num_classes: int = 4, lr: float = 1e-3) -> None:
        super().__init__(num_classes=num_classes, lr=lr)
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


class TumorDetectionModelSimple(BaseLightningClassifier):
    """Basic tumor detection model."""

    def __init__(self, num_classes: int = 4, lr: float = 1e-3) -> None:
        super().__init__(num_classes=num_classes, lr=lr)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    dummy_input = torch.randn(1, 1, 224, 224)

    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
