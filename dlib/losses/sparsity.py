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
from dlib.losses.entropy import Entropy



__all__ = ['SparseLinearFeaturesLoss',
           'SparseLinClassifierWeightsLoss',
           'SparseApvitAttentionLoss',
           'AttentionSizeLoss',
           'LowEntropyAttentionLoss'
           ]


class SparseLinearFeaturesLoss(ElementaryLoss):
    """
    Apply sparsity over the input features of the last linear classifier at
    the net output.
    """
    def __init__(self, **kwargs):
        super(SparseLinearFeaturesLoss, self).__init__(**kwargs)

        self.method: str = ''
        self.p: float = 1.
        self.c: float = 0.
        self.average_it: bool = False
        self.use_elb: bool = False
        self.already_set: bool = False

    def set_it(self,
               method: str,
               p: float,
               c: float,
               use_elb: bool,
               average_it:bool
               ):
        assert isinstance(method, str), type(method)
        assert method in constants.SPARSE_TECHS, method

        assert isinstance(p, float), type(p)
        assert 0. < p <= 1., p

        assert isinstance(c, float), type(c)
        assert c >= 0. , c

        assert isinstance(use_elb, bool), type(use_elb)
        if use_elb:
            assert isinstance(self.elb, ELB)

        assert isinstance(average_it, bool), type(average_it)

        self.method = method
        self.p = p
        self.c = c
        self.average_it = average_it
        self.use_elb = use_elb
        self.already_set = True

    def norm_it(self, x: torch.Tensor) -> torch.Tensor:

        assert x.ndim == 2, x.ndim

        if self.method == constants.SPARSE_L1:
            norm = torch.norm(input=x, p=1, dim=1, keepdim=False)

        elif self.method == constants.SPARSE_L2:
            norm = torch.norm(input=x, p=2, dim=1, keepdim=False)

        elif self.method == constants.SPARSE_INF:
            norm = torch.norm(input=x, p=float('inf'), dim=1, keepdim=False)

        else:
            raise NotImplementedError(self.method)

        return norm


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
        super(SparseLinearFeaturesLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert model is not None

        l_ft = model.linear_features
        assert l_ft.ndim == 2, l_ft.ndim  # bsz, d

        bsz, d = l_ft.shape
        n = min(int(self.p * d), d)
        n = max(n, 1)

        out = torch.topk(input=l_ft, k=n, dim=1, largest=False, sorted=True)[0]
        norm = self.norm_it(out)  # dim: 1.

        if self.average_it and (self.method != constants.SPARSE_INF):
            norm = norm / float(n)

        diff = norm - self.c

        if self.use_elb:
            loss = self.elb(diff.contiguous().view(-1, ))

        else:
            loss = (diff ** 2).mean()

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        return self.lambda_ * loss


class SparseLinClassifierWeightsLoss(SparseLinearFeaturesLoss):
    """
    Apply sparsity over the weights of the last linear classifier at
    the net output.
    """
    def __init__(self, **kwargs):
        super(SparseLinClassifierWeightsLoss, self).__init__(**kwargs)

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
        super(SparseLinearFeaturesLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert model is not None

        w = model.linear_w  # cls, nf
        assert w.ndim == 2, w.ndim  # cls, nf

        cls, d = w.shape
        n = min(int(self.p * d), d)
        n = max(n, 1)

        out = torch.topk(input=w, k=n, dim=1, largest=False, sorted=True)[0]
        norm = self.norm_it(out)  # dim: 1.

        if self.average_it and (self.method != constants.SPARSE_INF):
            norm = norm / float(n)

        diff = norm - self.c

        if self.use_elb:
            loss = self.elb(diff.contiguous().view(-1, ))

        else:
            loss = (diff ** 2).mean()

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        return self.lambda_ * loss



class SparseApvitAttentionLoss(SparseLinearFeaturesLoss):
    """
    Apply sparsity over the attention of the APVIT model.
    """
    def __init__(self, **kwargs):
        super(SparseApvitAttentionLoss, self).__init__(**kwargs)

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
        super(SparseLinearFeaturesLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert model is not None
        assert model.name == constants.APVIT, f"{model.name} |" \
                                              f" {constants.APVIT}"

        att = model.att_maps  # bsz, 1, h', w'
        assert att.ndim == 4, att.ndim  # bsz, 1, h', w'

        att = att.contiguous().view((att.shape[0], -1))  # bs, h'xw'.

        cls, d = att.shape
        n = min(int(self.p * d), d)
        n = max(n, 1)

        out = torch.topk(input=att, k=n, dim=1, largest=False, sorted=True)[0]
        norm = self.norm_it(out)  # dim: 1.

        if self.average_it and (self.method != constants.SPARSE_INF):
            norm = norm / float(n)

        diff = norm - self.c

        if self.use_elb:
            loss = self.elb(diff.contiguous().view(-1, ))

        else:
            loss = (diff ** 2).mean()

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        return self.lambda_ * loss



class AttentionSizeLoss(ElementaryLoss):
    """
    Constrain attention size.
    """
    def __init__(self, **kwargs):
        super(AttentionSizeLoss, self).__init__(**kwargs)

        self.low_b: float = 0.0
        self.up_b: float = 1.0
        assert isinstance(self.elb, ELB)

        self.already_set: bool = False

    def set_it(self, low_b: float, up_b: float):
        assert isinstance(low_b, float), type(low_b)
        assert 0.0 <= low_b < 1., low_b

        assert isinstance(up_b, float), type(up_b)
        assert 0.0 < up_b <= 1., up_b
        assert low_b < up_b, f"{low_b} | {up_b}"

        self.low_b = low_b
        self.up_b = up_b

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
        super(AttentionSizeLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert model is not None
        # todo: generalize to resnet.
        assert model.name == constants.APVIT, f"{model.name} |" \
                                              f" {constants.APVIT}"

        att = model.att_maps  # bsz, 1, h', w'
        assert att.ndim == 4, att.ndim  # bsz, 1, h', w'

        att = att.contiguous().view((att.shape[0], -1))  # bs, h'xw'.
        sz = att.mean(dim=1).view(-1, )  # bs

        # l <= sz <= h
        low = self.low_b - sz
        loss_l = self.elb(low)

        high = sz - self.up_b
        loss_h = self.elb(high)

        loss = loss_l + loss_h

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        return self.lambda_ * loss


class LowEntropyAttentionLoss(ElementaryLoss):
    """
    Low entropy constraint ver attention.
    """
    def __init__(self, **kwargs):
        super(LowEntropyAttentionLoss, self).__init__(**kwargs)

        self.loss = Entropy().to(self._device)

        self.already_set: bool = True

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
        super(LowEntropyAttentionLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert model is not None
        # todo: generalize to resnet.
        assert model.name == constants.APVIT, f"{model.name} |" \
                                              f" {constants.APVIT}"

        att = model.att_maps  # bsz, 1, h', w'
        assert att.ndim == 4, att.ndim  # bsz, 1, h', w'

        bsz, c, h, w = att.shape
        assert c == 1, c

        att = torch.cat((1. - att, att), dim=1)

        probs = att.permute(0, 2, 3, 1).contiguous().view(bsz * h * w, c + 1)

        return self.lambda_ * self.loss(probs).mean()