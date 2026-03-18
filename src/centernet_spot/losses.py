from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def heatmap_mse_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """MSE损失 - 基础的均方误差损失"""
    pred = pred.sigmoid()
    return F.mse_loss(pred, gt, reduction="mean")


def focal_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Focal Loss - 减少简单负样本的影响，增强难样本学习"""
    pred = pred.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 16)

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
    pred = pred.sigmoid().view(-1)
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
    pred = pred.sigmoid()
    
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

    
def gaussian_wasserstein_distance_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """高斯 Wasserstein 距离损失 - 适用于小目标检测

    优点：
    - 即使预测和GT不重叠也能提供有意义的梯度
    - 对小目标特别有效
    - 避免边界不连续问题

    将热力图视为2D高斯分布，计算Wasserstein距离
    """
    pred = pred.sigmoid()

    # 计算预测的加权中心（近似高斯均值）
    batch_size, _, height, width = pred.shape

    # 创建坐标网格
    y_coord = torch.arange(height, dtype=torch.float32, device=pred.device).view(1, 1, -1, 1)
    x_coord = torch.arange(width, dtype=torch.float32, device=pred.device).view(1, 1, 1, -1)

    # 计算预测分布的加权中心（均值）和方差
    pred_sum = pred.sum(dim=[2, 3], keepdim=True) + 1e-7
    mu_x_pred = (pred * x_coord).sum(dim=[2, 3], keepdim=True) / pred_sum
    mu_y_pred = (pred * y_coord).sum(dim=[2, 3], keepdim=True) / pred_sum

    # 计算方差
    var_x_pred = ((pred * (x_coord - mu_x_pred) ** 2).sum(dim=[2, 3], keepdim=True) / pred_sum).clamp(1e-5)
    var_y_pred = ((pred * (y_coord - mu_y_pred) ** 2).sum(dim=[2, 3], keepdim=True) / pred_sum).clamp(1e-5)

    # 对GT做同样的处理
    gt_sum = gt.sum(dim=[2, 3], keepdim=True) + 1e-7
    mu_x_gt = (gt * x_coord).sum(dim=[2, 3], keepdim=True) / gt_sum
    mu_y_gt = (gt * y_coord).sum(dim=[2, 3], keepdim=True) / gt_sum
    var_x_gt = ((gt * (x_coord - mu_x_gt) ** 2).sum(dim=[2, 3], keepdim=True) / gt_sum).clamp(1e-5)
    var_y_gt = ((gt * (y_coord - mu_y_gt) ** 2).sum(dim=[2, 3], keepdim=True) / gt_sum).clamp(1e-5)

    # 计算 1D Wasserstein 距离的平方 (对x和y分别计算)
    # W2^2 = (mu_x1 - mu_x2)^2 + (sigma_x1 - sigma_x2)^2
    wdist_x = (mu_x_pred - mu_x_gt) ** 2 + (torch.sqrt(var_x_pred) - torch.sqrt(var_x_gt)) ** 2
    wdist_y = (mu_y_pred - mu_y_gt) ** 2 + (torch.sqrt(var_y_pred) - torch.sqrt(var_y_gt)) ** 2

    # 2D Wasserstein 距离
    wdist = wdist_x + wdist_y

    # 只在有目标的位置计算损失
    pos_inds = gt.sum(dim=[2, 3]) > 0
    if pos_inds.sum() > 0:
        return wdist.view(batch_size, -1)[pos_inds].mean()

    return wdist.mean()


def combo_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Combo Loss - 组合多种损失函数

    结合 Focal Loss 和 Dice Loss 的优点：
    - Focal Loss: 处理类别不平衡
    - Dice Loss: 关注区域重叠

    Args:
        alpha: 权衡参数，0.5 表示两者同等重要
    """
    # Focal loss component
    pred_focal = pred.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    pos_loss_focal = torch.log(pred_focal) * torch.pow(1 - pred_focal, 2) * pos_inds
    neg_loss_focal = torch.log(1 - pred_focal) * torch.pow(pred_focal, 2) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    if num_pos > 0:
        focal = -(pos_loss_focal.sum() + neg_loss_focal.sum()) / num_pos
    else:
        focal = -neg_loss_focal.sum()

    # Dice loss component
    pred_dice = pred.sigmoid().view(-1)
    gt_dice = gt.view(-1)
    intersection = (pred_dice * gt_dice).sum()
    dice = (2. * intersection + 1e-5) / (pred_dice.sum() + gt_dice.sum() + 1e-5)
    dice_loss_val = 1 - dice

    # 组合
    return alpha * focal + (1 - alpha) * dice_loss_val


def earth_mover_distance_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Earth Mover's Distance (EMD) 损失 - 基于分布的匹配

    优点：
    - 避免高斯平滑假设
    - 提供一致的优化性能
    - 直接匹配预测和目标分布
    """
    pred = pred.sigmoid()
    batch_size, _, height, width = pred.shape

    # 展平空间维度
    pred_flat = pred.view(batch_size, -1)
    gt_flat = gt.view(batch_size, -1)

    # 归一化
    pred_flat = pred_flat / (pred_flat.sum(dim=1, keepdim=True) + 1e-7)
    gt_flat = gt_flat / (gt_flat.sum(dim=1, keepdim=True) + 1e-7)

    # 计算 cumulative distribution
    pred_cdf = torch.cumsum(pred_flat, dim=1)
    gt_cdf = torch.cumsum(gt_flat, dim=1)

    # EMD = sum(|CDF_pred - CDF_gt|)
    emd = (pred_cdf - gt_cdf).abs().sum(dim=1).mean()

    return emd


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
    "focal": focal_loss,
    "dice": dice_loss,
    "kl": kl_divergence_loss,
    "gwd": gaussian_wasserstein_distance_loss,
    "combo": combo_loss,
    "emd": earth_mover_distance_loss,
}


def get_heatmap_loss(name: str) -> callable:
    """获取热图损失函数"""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]
