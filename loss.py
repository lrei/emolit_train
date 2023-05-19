"""Loss functions."""

from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from torch.nn import MSELoss, SmoothL1Loss


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


class JSDLossWithLogits(nn.Module):
    """Calculate JS divergence loss.

    Assumes both input and target are (unnormalized) logits.
    """

    def __init__(self, smooth_factor=None):
        """Initialize KLDivLoss."""
        super().__init__()
        self.smooth_factor = smooth_factor

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward method implementation."""
        target = target.type_as(input)

        # convert to probabilities
        p = torch.sigmoid(input)
        q = torch.sigmoid(target)

        # compute the averge distribution
        m = 0.5 * (p + q)

        # clamp to avoid log(0) or too close
        p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
        q = torch.clamp(q, min=1e-7, max=1 - 1e-7)
        m = torch.clamp(m, min=1e-7, max=1 - 1e-7)

        # calculate the KL divergence
        kl_p_m = p * torch.log(p / m) + (1 - p) * torch.log((1 - p) / (1 - m))
        kl_q_m = q * torch.log(q / m) + (1 - q) * torch.log((1 - q) / (1 - m))

        # average
        loss = 0.5 * kl_p_m + 0.5 * kl_q_m

        return loss


class KDLoss(nn.Module):
    """Calculate the Kknowledge distillation loss."""

    def __init__(self, loss_function="ce", smooth_factor=None, alpha=0.5):
        """Initialize KLDivLoss."""
        super().__init__()
        self.smooth_factor = smooth_factor
        self.alpha = alpha
        self.loss_function = loss_function

    def forward(
        self, input: Tensor, teacher: Tensor, ground: Tensor
    ) -> Tensor:
        """Forward method implementation."""
        teacher = teacher.type_as(input)
        ground = ground.type_as(input)

        teacher = torch.sigmoid(teacher)

        loss_fn = SoftBCEWithLogitsLoss(smooth_factor=self.smooth_factor)
        loss_from_teacher = self.alpha * loss_fn(input, teacher)
        loss_from_ground = (1.0 - self.alpha) * loss_fn(input, ground)

        loss = loss_from_teacher + loss_from_ground

        return loss


class KDTrainer(Trainer):
    """Trainer for Knowledge Distillation.

    Assumes that both models have the same tokenizer.
    """

    def __init__(self, *args, teacher, label_smoothing_factor=None, **kwargs):
        """Initialize SoftBCETrainer."""
        super().__init__(*args, **kwargs)
        self.label_smoothing_factor = label_smoothing_factor
        self.teacher = teacher

        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss function."""
        # ensure models on same device
        self._move_model_to_device(self.teacher, model.device)

        labels = inputs.get("labels")

        # forward pass for teacher
        teacher_outputs = self.teacher(**inputs)
        teacher_logits = teacher_outputs.get("logits")

        # forward pass for student
        student_outputs = model(**inputs)
        student_logits = student_outputs.get("logits")

        # ensure both logits are on the same device as the models
        student_logits = student_logits.to(teacher_logits.device)

        # loss_fct = SmoothL1Loss()
        # loss_fct = MSELoss()
        # loss_fct = SoftBCEWithLogitsLoss(
        #     smooth_factor=self.label_smoothing_factor
        # )
        # here we had to sigmoid the logits for SoftBCE
        # teacher_logits = torch.sigmoid(teacher_logits)
        loss_fct = KDLoss(smooth_factor=self.label_smoothing_factor)

        # loss = loss_fct(student_logits, teacher_logits)
        # for the kd loss, we need the ground truth
        loss = loss_fct(student_logits, teacher_logits, labels)
        return (loss, student_outputs) if return_outputs else loss
