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


__all__ = ['HighEntropy']


class HighEntropy(ElementaryLoss):
    """
    Regularizer that favors classification distribution with high entropy.
    """
    def __init__(self, **kwargs):
        super(HighEntropy, self).__init__(**kwargs)

        self.type_reg = constants.MAX_ENTROPY
        self.alpha: float = 0.0

        self._entropy = Entropy()

        self.already_set = False

    def set_it(self, type_reg: str, alpha: float):
        assert isinstance(type_reg, str), type(type_reg)
        assert type_reg in constants.HIGH_ENTROPY_REGS, type_reg

        assert 0. < alpha < 1., alpha

        self.type_reg = type_reg
        self.alpha = alpha

        self.already_set = True

    def kl_uniform(self, probs: torch.Tensor) -> torch.Tensor:
        assert probs.ndim == 2, probs.ndim
        bsz, ncls = probs.shape
        assert ncls > 0, ncls

        u = torch.ones((bsz, ncls), dtype=probs.dtype, requires_grad=False,
                       device=probs.device) * (1. / ncls)
        loss = self._entropy(p=u, q=probs).mean()
        return loss


    def max_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        return self._entropy(p=probs, q=None).mean()

    def generalized_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        # https://arxiv.org/abs/2005.00820
        # q: uniform
        # G: -entropy.

        assert probs.ndim == 2, probs.ndim
        bsz, ncls = probs.shape
        assert ncls > 0, ncls

        u = torch.ones((bsz, ncls), dtype=probs.dtype, requires_grad=False,
                       device=probs.device) * (1. / ncls)

        dist_p = - self._entropy(p=probs, q=None).mean()
        dist_u = - self._entropy(p=u, q=None).mean()

        a = self.alpha
        comb_p = (1. - a) * u + a * probs

        dist_c = - self._entropy(p=comb_p, q=None).mean()

        loss = (1. / (a * (1. - a))) * ((1. - a) * dist_u + a * dist_p - dist_c)

        return loss

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
        super(HighEntropy, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert cl_logits.ndim == 2, cl_logits.ndim  # bsz, n_cls
        probs = F.softmax(cl_logits, dim=1)

        if self.type_reg == constants.KL_UNIFORM:
            loss = self.kl_uniform(probs)

        elif self.type_reg == constants.MAX_ENTROPY:
            loss = - self.max_entropy(probs)

        elif self.type_reg == constants.GEN_ENTROPY:
            loss = self.generalized_entropy(probs)

        else:
            raise NotImplementedError(self.type_reg)

        return self.lambda_ * loss