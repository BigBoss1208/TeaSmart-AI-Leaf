"""MobileNetV2 construction and the two training stages."""

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.mobilenetv2 import MobileNetV2

from config import CLASS_TO_IDX, WEIGHTS_CACHE, WEIGHTS_NAME, TrainingConfig


def build_model(*, pretrained: bool) -> MobileNetV2:
    if pretrained:
        torch.hub.set_dir(str(WEIGHTS_CACHE))
    weights = MobileNet_V2_Weights[WEIGHTS_NAME] if pretrained else None
    model = mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_TO_IDX))
    return model


def build_optimizer(model: MobileNetV2, config: TrainingConfig) -> torch.optim.AdamW:
    # Keep both groups across the phase change so classifier momentum is preserved.
    return torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": config.warmup_classifier_lr},
            {"params": model.features.parameters(), "lr": 0.0},
        ],
        weight_decay=config.weight_decay,
    )


def configure_stage(
    model: MobileNetV2, optimizer: torch.optim.AdamW, epoch: int, config: TrainingConfig
) -> str:
    finetuning = epoch > config.freeze_epochs
    model.train()
    model.features.eval().requires_grad_(False)
    if finetuning:
        for block in model.features[-config.finetune_blocks:]:
            block.train().requires_grad_(True)
        # Keep ImageNet running means and variances with small local batches.
        for layer in model.features.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    optimizer.param_groups[0]["lr"] = (
        config.finetune_classifier_lr if finetuning else config.warmup_classifier_lr
    )
    optimizer.param_groups[1]["lr"] = config.finetune_backbone_lr if finetuning else 0.0
    return "finetune" if finetuning else "classifier"
