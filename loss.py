"""Loss functions."""

from typing import Optional

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class SoftBCEWithLogitsLoss(nn.Module):
    """Calculate BCE with soft labels.

    Replacement for nn.BCEWithLogitsLoss with few additions:
    - Support of ignore_index value
    - Support of label smoothing
    """

    __constants__ = [
        "weight",
        "pos_weight",
        "reduction",
        "ignore_index",
        "smooth_factor",
    ]

    def __init__(
        self,
        weight=None,
        ignore_index: Optional[int] = -100,
        reduction="mean",
        smooth_factor=None,
        pos_weight=None,
    ):
        """Initialize BCEWithLogitsLoss."""
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward method implementation."""
        if self.smooth_factor is not None:
            soft_targets = (
                (1 - target) * self.smooth_factor
                + target * (1 - self.smooth_factor)
            ).type_as(input)
        else:
            soft_targets = target.type_as(input)

        loss = F.binary_cross_entropy_with_logits(
            input,
            soft_targets,
            self.weight,  # type: ignore
            pos_weight=self.pos_weight,  # type: ignore
            reduction="none",
        )

        if self.ignore_index is not None:
            not_ignored_mask: Tensor = target != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss


class SoftBCETrainer(Trainer):
    """Trainer with SoftBCEWithLogitsLoss loss function."""

    def __init__(self, *args, label_smoothing_factor=None, **kwargs):
        """Initialize SoftBCETrainer."""
        super().__init__(*args, **kwargs)
        self.label_smoothing_factor = label_smoothing_factor

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss function."""
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = SoftBCEWithLogitsLoss(
            smooth_factor=self.label_smoothing_factor
        )
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

