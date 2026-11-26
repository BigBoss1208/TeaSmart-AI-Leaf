"""Aggregate metrics over an entire split using one multiclass confusion matrix."""

import torch


class EpochMetrics:
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.confusion = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64, device=device
        )
        self.loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        self.samples = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> None:
        predictions = logits.detach().argmax(dim=1)
        offsets = targets * self.num_classes + predictions
        self.confusion += torch.bincount(
            offsets, minlength=self.num_classes ** 2
        ).reshape_as(self.confusion)
        self.loss_sum += loss.detach().float() * targets.numel()
        self.samples += targets.numel()

    def summarize(self) -> dict[str, float]:
        if self.samples == 0:
            raise ValueError("The loader produced no samples.")
        confusion = self.confusion.cpu().to(torch.float64)
        true_positive = confusion.diagonal()
        totals = confusion.sum(dim=0) + confusion.sum(dim=1)
        class_f1 = 2 * true_positive / totals.clamp_min(1)
        return {
            "loss": self.loss_sum.item() / self.samples,
            "accuracy": true_positive.sum().item() / self.samples,
            "macro_f1": class_f1.mean().item(),
        }
