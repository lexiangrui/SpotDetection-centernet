from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def heatmap_mse_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """前景 MSE 损失：对 logits 做 sigmoid 后仅在 GT 前景区域计算。"""
    pred = logits.sigmoid()
    fg_mask = (gt > 0).float()
    valid_count = fg_mask.sum()
    if valid_count <= 0:
        return pred.new_zeros(())

    sq_error = (pred - gt) ** 2
    return (sq_error * fg_mask).sum() / valid_count


def heatmap_bce_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """标准 BCEWithLogits 损失：直接在 logits 上计算。"""
    return F.binary_cross_entropy_with_logits(logits, gt, reduction="mean")


def focal_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """CenterNet 风格 Focal Loss：直接在 logits 上计算。"""
    pred = logits.sigmoid()
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = F.logsigmoid(logits) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = F.logsigmoid(-logits) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def kl_divergence_loss(logits: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """KL 散度损失：对 logits 做 sigmoid 后再归一化为概率分布进行匹配。"""
    pred = logits.sigmoid()
    pred = pred + eps
    pred = pred / pred.sum(dim=[2, 3], keepdim=True)

    gt = gt + eps
    gt = gt / gt.sum(dim=[2, 3], keepdim=True)

    kl_div = gt * torch.log(gt / pred)

    pos_inds = gt.sum(dim=[2, 3]) > eps

    if pos_inds.sum() > 0:
        return kl_div.sum(dim=[2, 3])[pos_inds].mean()

    return kl_div.sum(dim=[2, 3]).mean()


def transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
    feat = feat.gather(1, ind)
    return feat


def reg_l1_loss(pred: torch.Tensor, target: torch.Tensor, ind: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = transpose_and_gather_feat(pred, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss


LOSS_REGISTRY = {
    "mse": heatmap_mse_loss,
    "bce": heatmap_bce_loss,
    "focal": focal_loss,
    "kl": kl_divergence_loss,
}


def get_heatmap_loss(name: str) -> Callable:
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]
