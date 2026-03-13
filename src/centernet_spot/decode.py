from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from .losses import transpose_and_gather_feat
from .transforms import transform_preds


def _nms(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, ...]:
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), k)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), k)
    topk_clses = (topk_ind // k).int()
    topk_inds = topk_inds.view(batch, -1, 1).gather(1, topk_ind.unsqueeze(-1)).squeeze(-1)
    topk_ys = topk_ys.view(batch, -1, 1).gather(1, topk_ind.unsqueeze(-1)).squeeze(-1)
    topk_xs = topk_xs.view(batch, -1, 1).gather(1, topk_ind.unsqueeze(-1)).squeeze(-1)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_predictions(
    heatmap: torch.Tensor,
    reg: torch.Tensor,
    transform: dict[str, float | int],
    topk: int = 100,
    score_threshold: float = 0.2,
    nms_kernel: int = 3,
) -> List[dict]:
    heatmap = heatmap.sigmoid()
    heatmap = _nms(heatmap, kernel=nms_kernel)
    scores, inds, _, ys, xs = _topk(heatmap, k=topk)
    reg = transpose_and_gather_feat(reg, inds)

    xs = xs + reg[..., 0]
    ys = ys + reg[..., 1]

    scores = scores[0].detach().cpu().numpy()
    coords = np.stack(
        [
            xs[0].detach().cpu().numpy(),
            ys[0].detach().cpu().numpy(),
        ],
        axis=1,
    )
    coords = transform_preds(coords, transform)

    detections: List[dict] = []
    for score, (x, y) in zip(scores.tolist(), coords.tolist()):
        if score < score_threshold:
            continue
        detections.append(
            {
                "score": float(score),
                "x": float(x),
                "y": float(y),
                "class_id": 0,
            }
        )
    return detections
