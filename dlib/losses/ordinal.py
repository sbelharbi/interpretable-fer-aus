import sys
from os.path import dirname, abspath
from typing import Union, Tuple
import math

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.losses.elementary import ElementaryLoss
from dlib.configure import constants


__all__ = ['OrdinalMeanLoss', 'OrdinalVarianceLoss', 'OrdIneqUnimodLoss']


class OrdinalMeanLoss(ElementaryLoss):
    """
    Loss: mean ordinal classification: (expected label - true label)** <= eps.
    """

    def __init__(self, **kwargs):
        super(OrdinalMeanLoss, self).__init__(**kwargs)

        self.eps = 0.0
        self.use_elb: bool = False
        self.already_set: bool = False

    def set_it(self, use_elb: bool, eps: float):

        assert isinstance(eps, float), type(eps)
        assert eps >= 0, eps
        self.eps = eps

        if use_elb:
            assert isinstance(self.elb, ELB)

        self.use_elb = use_elb
        self.already_set = True

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(OrdinalMeanLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert cl_logits is not None
        assert cl_logits.ndim == 2, cl_logits.ndim

        ncls = cl_logits.shape[1]
        probs = torch.softmax(cl_logits, dim=1)

        c = torch.arange(0, ncls, dtype=torch.float32).cuda()
        c = c.view(1, -1)
        mean_cl = (probs * c).sum(dim=1, keepdim=True).squeeze(1)  # bc
        diff = (mean_cl - glabel) ** 2
        diff = diff.contiguous().view(-1, )

        if self.use_elb:
            loss = self.elb(diff - self.eps)

        else:
            loss = ((diff - self.eps) ** 2).mean()

        return self.lambda_ * loss


class OrdinalVarianceLoss(ElementaryLoss):
    """
    Loss: variance ordinal classification: variance label <=  eps.
    """

    def __init__(self, **kwargs):
        super(OrdinalVarianceLoss, self).__init__(**kwargs)

        self.eps = 0.0
        self.use_elb: bool = False
        self.already_set: bool = False

    def set_it(self, use_elb: bool, eps: float):

        assert isinstance(eps, float), type(eps)
        assert eps >= 0, eps
        self.eps = eps

        if use_elb:
            assert isinstance(self.elb, ELB)

        self.use_elb = use_elb
        self.already_set = True

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(OrdinalVarianceLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert cl_logits is not None
        assert cl_logits.ndim == 2, cl_logits.ndim

        ncls = cl_logits.shape[1]
        probs = torch.softmax(cl_logits, dim=1)

        c = torch.arange(0, ncls, dtype=torch.float32).cuda()
        c = c.view(1, -1)  # 1, ncl
        mean_cl = (probs * c).sum(dim=1, keepdim=True).squeeze(1)  # bc
        mean_cl = mean_cl.view(-1, 1)  # bc, 1

        variance = probs * ((c - mean_cl) ** 2)  # bc, ncl
        variance = variance.sum(dim=1, keepdim=True)  # bc, 1

        if self.use_elb:
            loss = self.elb((variance - self.eps).view(-1, ))

        else:
            loss = ((variance - self.eps) ** 2).mean()

        return self.lambda_ * loss


class _NeighborDifferentiator(nn.Module):
    """
    Related to: OrdIneqUnimodLoss().
    Computes the difference between the neighbors: `Delta_a^b(s)`.
    use 1D convolution for ease, clean, and speed.

    The differences are computed over all the adjacent neighbors, from left
    to right. i.e., s(i) - s(i+1).

    If s in R^c, the output of this differentiator is in R^(c-1).
    It operates over a 2D matrix over rows (teh convolution is performed over
    the rows).
    """
    def __init__(self):
        """
        Init. function.
        """
        super(_NeighborDifferentiator, self).__init__()
        # constant convolution kernel.
        kernel2rightct = torch.tensor([+1, -1], requires_grad=False
                                      ).float().view(1, 1, 2)
        self.register_buffer("kernel2right", kernel2rightct)

    def forward(self, x):
        """
        Compute the difference between all the adjacent neighbors.
        :param x: torch tensor of size (nbr_samples, nbr_calsses)
        contains scores.
        :return: left to right differences. torch tensor of size
        (nbr_samples, nbr_calsses - 1).
        """
        assert x.ndim == 2, x.ndim

        assert x.shape[1] > 1, x.shape[1]
        h, w = x.shape
        output2right = F.conv1d(input=x.view(h, 1, w), weight=self.kernel2right,
                                bias=None, stride=1, padding=0, dilation=1,
                                groups=1)
        assert output2right.shape == (h, 1, w - 1), f"{output2right.shape} | " \
                                                    f"{(h, 1, w - 1)}"
        return output2right.squeeze(1)  # h, w - 1.


class _RightAndLeftDelta(nn.Module):
    """
    Related to: OrdIneqUnimodLoss().

    Computes the Delta on the left and the right of a reference label.
    Then, put everything in one single matrix (or a row for one single sample).
    """
    def __init__(self):
        """
        Init. function.
        """
        super(_RightAndLeftDelta, self).__init__()
        self.differentiator = _NeighborDifferentiator()

    def forward(self, x: torch.Tensor, rlabels: torch.Tensor):
        """
        Compute the difference between neighbors on the left and right. Then,
        put all the differences within the same vector.
        :param x: torch tensor of size (nbr_samples, nbr_calsses)
        contains scores.
        :param rlabels: torch.long tensor. Reference labels to which we
        need to compute left and right differences. It can be the ground
        truth, the predicted labels, or any other labels.
        :return: left and right differences in one single matrix. torch
        tensor of size (nbr_samples, nbr_calsses - 1).
        """

        assert x.ndim == 2, x.ndim

        assert x.shape[1] > 1, x.shape[1]

        assert rlabels.ndim == 1, rlabels.ndim
        assert rlabels.numel() == x.shape[0], f"{rlabels.numel()} | " \
                                              f"{x.shape[0]}"
        y = rlabels.view(-1, 1)
        n, c = x.shape

        seeleft = self.differentiator(x)  # n, c-1
        assert seeleft.shape[0] == n, f"{seeleft.shape[0]} | {n}"
        assert seeleft.shape[1] == c - 1, f"{seeleft.shape[1]} | {c - 1}"

        seeright = - seeleft

        h, w = seeleft.shape
        yy = y.repeat(1, w).float()
        idx = torch.arange(start=0, end=w, step=1, dtype=seeleft.dtype,
                           device=seeleft.device, requires_grad=False)
        idx = idx.repeat(h, 1)
        # ======================================================================
        #                       LEFT TO RIGHT
        # ======================================================================
        leftonrightoff = (idx < yy).type(seeleft.dtype).to(
            seeleft.device).requires_grad_(False)  # idx < yy
        leftside = leftonrightoff * seeleft

        # ======================================================================
        #                       RIGHT TO LEFT
        # ======================================================================
        leftoffrighton = 1 - leftonrightoff  # idx > yy or idx == yy.
        rightside = leftoffrighton * seeright

        # ======================================================================
        #                  BOTH SIDES IN ONE SINGLE MATRIX
        # ======================================================================

        return leftside + rightside


class OrdIneqUnimodLoss(ElementaryLoss):
    """
    Apply uni-modality over logits/probs using inequality constraints.
    Given y as ground truth.
    constraints:
    s_k < s_(k+1), for k < y
    s_(k+1) < sk, for k <= y.
    """

    def __init__(self, **kwargs):
        super(OrdIneqUnimodLoss, self).__init__(**kwargs)

        self.differ = _RightAndLeftDelta().to(self._device)
        self.data_type = constants.LOGITS
        self.already_set: bool = False

    def set_it(self, data_type: str):
        assert data_type in constants.DATA_TYPE, data_type

        assert isinstance(self.elb, ELB)

        self.data_type = data_type
        self.already_set = True

    def forward(self,
                model=None,
                epoch=0,
                cams_inter=None,
                fcams=None,
                cl_logits=None,
                glabel=None,
                raw_img=None,
                x_in=None,
                im_recon=None,
                seeds=None,
                heatmap=None,
                seg_map=None,
                bin_seg=None
                ):
        super(OrdIneqUnimodLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        data = cl_logits
        assert data.ndim == 2, data.ndim  # bc, cls

        if self.data_type == constants.PROBS:
            data = torch.softmax(data, dim=1)

        deltas = self.differ(data, glabel)  # bc, cls - 1
        deltas = deltas.contiguous().view(-1, )

        loss = self.elb(deltas)

        return self.lambda_ * loss

