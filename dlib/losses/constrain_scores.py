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


__all__ = ['ConstraintScoresLoss']


class ConstraintScoresLoss(ElementaryLoss):
    """
    Constrain score (logit) of the true class to be higher than logits of
    other classes.
    logit(true_label) >= logit(other_labels).

    """
    def __init__(self, **kwargs):
        super(ConstraintScoresLoss, self).__init__(**kwargs)

        assert isinstance(self.elb, ELB)

        self.n_cls: int = 1
        self.con_scores_min: float = 0.0

        self.already_set = False

    def set_it(self, n_cls: int, con_scores_min: float):
        assert isinstance(n_cls, int), type(n_cls)
        assert n_cls > 0, n_cls
        self.n_cls = n_cls

        assert isinstance(con_scores_min, float), type(con_scores_min)
        assert con_scores_min >= 0., con_scores_min

        self.con_scores_min = con_scores_min


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
        super(ConstraintScoresLoss, self).forward(epoch=epoch)

        assert self.already_set

        if not self.is_on():
            return self._zero

        logits = cl_logits
        true_logits = logits.gather(dim=1, index=glabel.view(-1, 1))

        diff = true_logits - logits  # bsz, n_cl
        b, cl = diff.shape

        idx_diff = F.one_hot(glabel, num_classes=self.n_cls)
        diff = diff[(1 - idx_diff).bool()].reshape(b, -1)  # bsz, n_cl - 1
        diff = diff.contiguous().view(-1, )

        diff = diff - self.con_scores_min

        loss = self.elb(- diff)

        return self.lambda_ * loss



