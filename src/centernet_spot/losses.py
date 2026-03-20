from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def heatmap_mse_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """前景 MSE 损失。

    预测热图已统一完成 sigmoid，这里仅在 GT 热图前景区域 (gt > 0)
    计算 MSE，背景像素不参与学习。
    """
    fg_mask = (gt > 0).float()
    valid_count = fg_mask.sum()
    if valid_count <= 0:
        return pred.new_zeros(())

    sq_error = (pred - gt) ** 2
    return (sq_error * fg_mask).sum() / valid_count


def heatmap_bce_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """前景 BCE 损失。

    预测热图已统一完成 sigmoid，这里直接在概率图上计算 BCE，
    仅在 GT 热图前景区域 (gt > 0) 参与学习。
    """
    fg_mask = (gt > 0).float()
    valid_count = fg_mask.sum()
    if valid_count <= 0:
        return pred.new_zeros(())

    pred = pred.clamp(1e-4, 1 - 1e-4)
    bce_map = F.binary_cross_entropy(pred, gt, reduction="none")
    return (bce_map * fg_mask).sum() / valid_count


def focal_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Focal Loss - 减少简单负样本的影响，增强难样本学习"""
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


def dice_loss(pred: torch.Tensor, gt: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Dice Loss - 基于区域重叠的损失函数，适用于不平衡数据集

    优点：
    - 关注预测与GT的区域重叠而非单个像素
    - 对小目标/稀疏目标效果好
    - 减少类别不平衡问题
    """
    pred = pred.view(-1)
    gt = gt.view(-1)

    intersection = (pred * gt).sum()
    dice_coeff = (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)

    return 1 - dice_coeff


def kl_divergence_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """KL散度损失 - 将热力图作为概率分布进行匹配

    优点：
    - 将预测和GT都视为概率分布进行匹配
    - 对小目标检测效果良好
    - 提供平滑的梯度信号
    - 适用于不确定性估计

    Args:
        pred: 预测热力图 [B, 1, H, W]
        gt: GT热力图 [B, 1, H, W]
        eps: 防止log(0)的 epsilon 值
    """
    # 归一化为概率分布
    pred = pred + eps
    pred = pred / pred.sum(dim=[2, 3], keepdim=True)
    
    gt = gt + eps
    gt = gt / gt.sum(dim=[2, 3], keepdim=True)
    
    # KL散度: KL(gt || pred) = sum(gt * log(gt/pred))
    # 使用 pred 作为 target, gt 作为 source (让预测去逼近GT)
    kl_div = gt * torch.log(gt / pred)
    
    # 只在有目标的位置计算损失
    pos_inds = gt.sum(dim=[2, 3]) > eps
    
    if pos_inds.sum() > 0:
        return kl_div.sum(dim=[2, 3])[pos_inds].mean()
    
    return kl_div.sum(dim=[2, 3]).mean()


def transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """转置并收集特征 - 用于回归损失计算"""
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
    feat = feat.gather(1, ind)
    return feat


def reg_l1_loss(pred: torch.Tensor, target: torch.Tensor, ind: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1回归损失 - 预测偏移量与GT的L1距离"""
    pred = transpose_and_gather_feat(pred, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss


# 损失函数注册表
LOSS_REGISTRY = {
    "mse": heatmap_mse_loss,
    "bce": heatmap_bce_loss,
    "focal": focal_loss,
    "dice": dice_loss,
    "kl": kl_divergence_loss,
}


def get_heatmap_loss(name: str) -> callable:
    """获取热图损失函数"""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]
