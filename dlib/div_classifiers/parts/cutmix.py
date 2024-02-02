"""
Original repository: https://github.com/clovaai/CutMix-PyTorch
"""

import numpy as np
import torch

__all__ = ['cutmix']


def cutmix(x: torch.Tensor,
           target: torch.Tensor,
           beta: float,
           std_cams : torch.Tensor = None,
           lndmks_heatmap: torch.Tensor = None,
           au_heatmap: torch.Tensor = None,
           heatmap_seg: torch.Tensor = None,
           bin_heatmap_seg: torch.Tensor = None
           ):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(x.size()[0]).cuda()

    target_a = target.clone().detach()
    target_b = target[rand_index].clone().detach()

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]

    if std_cams is not None:
        std_cams[:, :, bbx1:bbx2, bby1:bby2] = std_cams[
                                               rand_index, :, bbx1:bbx2,
                                               bby1:bby2]

    if lndmks_heatmap is not None:
        lndmks_heatmap[:, :, bbx1:bbx2, bby1:bby2] = lndmks_heatmap[
                                                     rand_index, :,
                                                     bbx1:bbx2, bby1:bby2]

    if au_heatmap is not None:
        au_heatmap[:, :, bbx1:bbx2, bby1:bby2] = au_heatmap[
                                                 rand_index, :,
                                                 bbx1:bbx2,
                                                 bby1:bby2]

    if heatmap_seg is not None:
        heatmap_seg[:, :, bbx1:bbx2, bby1:bby2] = heatmap_seg[
                                                  rand_index, :,
                                                  bbx1:bbx2, bby1:bby2]

    if bin_heatmap_seg is not None:
        bin_heatmap_seg[:, :, bbx1:bbx2, bby1:bby2] = bin_heatmap_seg[
                                                      rand_index, :,
                                                      bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, std_cams, lndmks_heatmap, au_heatmap, heatmap_seg, \
           bin_heatmap_seg, target_a, target_b, lam


def rand_bbox(size, lam):
    w = size[2]
    h = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)

    return bbx1, bby1, bbx2, bby2
