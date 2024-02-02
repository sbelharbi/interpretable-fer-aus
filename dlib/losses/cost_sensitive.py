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


__all__ = ['SelfCostSensitiveLoss']


class SelfCostSensitiveLoss(ElementaryLoss):
    """
    Apply sample-cost sensitive penalty.
    """

    def __init__(self, **kwargs):
        super(SelfCostSensitiveLoss, self).__init__(**kwargs)

        self.apply_to: str = constants.LOGITS
        self.n_cls: int = 1
        self.norm: str = constants.NORM_SCORES_MAX
        self.confusion_func: str = constants.LINEAR_CONF_FUNC
        self.top_k: int = -1  # all classes.
        self.reduction: str = constants.REDUCE_MEAN

        self.already_set: bool = False

    def set_it(self,
               apply_to: str,
               n_cls: int,
               norm: str,
               confusion_func: str,
               top_k: int,
               reduction: str
               ):
        assert isinstance(apply_to, str), type(apply_to)
        assert apply_to in [constants.LOGITS, constants.PROBS], apply_to

        assert isinstance(n_cls, int), type(n_cls)
        assert n_cls > 0, n_cls
        self.n_cls = n_cls

        assert isinstance(norm, str), type(norm)
        assert norm in constants.NORM_SCORES, norm
        self.norm = norm

        assert isinstance(confusion_func, str), type(confusion_func)
        assert confusion_func in constants.CONFUSION_FUNCS, confusion_func
        self.confusion_func = confusion_func

        assert isinstance(top_k, int), type(top_k)
        if top_k != -1:
            assert n_cls > 1, n_cls
            assert 0 < top_k <= n_cls - 1, f"0 < {top_k} <= {n_cls - 1}"

        self.top_k = top_k

        assert isinstance(reduction, str), type(reduction)
        assert reduction in constants.REDUCTIONS, reduction
        self.reduction = reduction

        self.apply_to = apply_to

        self.already_set = True

    def normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        assert isinstance(scores, torch.Tensor), type(scores)
        assert scores.ndim == 2, scores.ndim  # bsz, n_cls - 1.

        if self.norm == constants.NORM_SCORES_NONE:
            return scores

        if self.norm == constants.NORM_SCORES_MAX:
            return self._max_norm_scores(scores)

        if self.norm == constants.NORM_SCORES_SUM:
            return self._sum_norm_scores(scores)

        if self.norm == constants.NORM_SCORES_MEAN:
            return self._mean_norm_scores(scores)

        raise NotImplementedError(self.norm)

    def _max_norm_scores(self, scores: torch.Tensor) -> torch.Tensor:
        assert isinstance(scores, torch.Tensor), type(scores)
        assert scores.ndim == 2, scores.ndim  # bsz, n_cls - 1.

        assert self.norm == constants.NORM_SCORES_MAX, self.norm

        normed: torch.Tensor = scores * 1.

        b, c = scores.shape

        for i in range(b):
            v = scores[i]
            n_v = v * 1.
            # norm negative vals
            _min_neg = v.min()
            if _min_neg < 0:
                n_v[v < 0] = n_v[v < 0] / torch.abs(_min_neg)

            # norm positive vals
            _max_pos = v.max()
            if _max_pos > 0:
                n_v[v > 0] = n_v[v > 0] / _max_pos

            # zero: stays 0.

            normed[i] = n_v

        return normed

    def _sum_norm_scores(self, scores: torch.Tensor) -> torch.Tensor:
        assert isinstance(scores, torch.Tensor), type(scores)
        assert scores.ndim == 2, scores.ndim  # bsz, n_cls - 1.

        assert self.norm == constants.NORM_SCORES_MAX, self.norm

        normed: torch.Tensor = scores * 1.

        b, c = scores.shape

        for i in range(b):
            v = scores[i]
            n_v = v * 1.
            # norm negative vals
            _min_neg = v.min()
            if _min_neg < 0:
                z = v[v < 0].abs().sum()
                n_v[v < 0] = n_v[v < 0] / z

            # norm positive vals
            _max_pos = v.max()
            if _max_pos > 0:
                z = v[v > 0].sum()
                n_v[v > 0] = n_v[v > 0] / z

            # zero: stays 0.

            normed[i] = n_v

        return normed

    def _mean_norm_scores(self, scores: torch.Tensor) -> torch.Tensor:
        assert isinstance(scores, torch.Tensor), type(scores)
        assert scores.ndim == 2, scores.ndim  # bsz, n_cls - 1.

        assert self.norm == constants.NORM_SCORES_MAX, self.norm

        normed: torch.Tensor = scores * 1.

        b, c = scores.shape

        for i in range(b):
            v = scores[i]
            n_v = v * 1.
            # norm negative vals
            _min_neg = v.min()
            if _min_neg < 0:
                z = v[v < 0].abs().mean()
                n_v[v < 0] = n_v[v < 0] / z

            # norm positive vals
            _max_pos = v.max()
            if _max_pos > 0:
                z = v[v > 0].mean()
                n_v[v > 0] = n_v[v > 0] / z

            # zero: stays 0.

            normed[i] = n_v

        return normed

    def _confusion_scores(self, scores: torch.Tensor) -> torch.Tensor:

        assert isinstance(scores, torch.Tensor), type(scores)
        assert scores.ndim == 2, scores.ndim  # bsz, n_cls - 1.

        if self.confusion_func == constants.LINEAR_CONF_FUNC:
            return scores

        if self.confusion_func == constants.EXP_CONF_FUNC:
            return torch.exp(scores)

        raise NotImplementedError(self.confusion_func)

    def _get_top_k(self, scores: torch.Tensor) -> torch.Tensor:

        assert isinstance(scores, torch.Tensor), type(scores)
        assert scores.ndim == 2, scores.ndim  # bsz, n_cls - 1.

        b, c = scores.shape

        if self.top_k in [-1, self.n_cls - 1]:
            assert c == self.n_cls - 1, f"{c} | {self.n_cls - 1}"

            return scores

        assert 0 < self.top_k <= c, f"{self.top_k} | {c}"

        out = torch.topk(input=scores, k=self.top_k, dim=1, largest=True,
                         sorted=True)[0]

        return out

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

        super(SelfCostSensitiveLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert cl_logits.ndim == 2, cl_logits.ndim

        scores = cl_logits

        if self.apply_to == constants.PROBS:
            scores = F.softmax(scores, dim=1)

        ref_scores = scores.gather(dim=1, index=glabel.view(-1, 1))

        diff = scores - ref_scores
        b, cl = diff.shape

        idx_diff = F.one_hot(glabel, num_classes=self.n_cls)
        diff = diff[(1 - idx_diff).bool()].reshape(b, -1)  # bsz, n_cl - 1

        scores = self.normalize_scores(diff)

        conf_scores = self._confusion_scores(scores)

        conf_scores = self._get_top_k(conf_scores)

        if self.reduction == constants.REDUCE_SUM:
            loss = conf_scores.sum(dim=1)

        elif self.reduction == constants.REDUCE_MEAN:
            loss = conf_scores.mean(dim=1)

        else:
            raise NotImplementedError(self.reduction)

        return self.lambda_ * loss.mean()



