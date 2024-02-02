import sys
from os.path import dirname, abspath
from typing import List

import re
import torch.nn as nn
import torch
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.configure import constants

__all__ = [
    'ElementaryLoss'
]


class ElementaryLoss(nn.Module):
    def __init__(self,
                 cuda_id,
                 name=None,
                 lambda_=1.,
                 elb=nn.Identity(),
                 logit=False,
                 support_background=False,
                 multi_label_flag=False,
                 sigma_rgb=15.,
                 sigma_xy=100.,
                 scale_factor=0.5,
                 start_epoch=None,
                 end_epoch=None,
                 seg_ignore_idx=-255
                 ):
        super(ElementaryLoss, self).__init__()
        self._name = name

        assert isinstance(lambda_, float), type(lambda_)
        assert lambda_ >= 0, lambda_
        self.lambda_ = lambda_

        self.elb = elb
        self.logit = logit
        self.support_background = support_background

        assert not multi_label_flag
        self.multi_label_flag = multi_label_flag

        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor

        if end_epoch == -1:
            end_epoch = None

        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.c_epoch = 0

        if self.logit:
            assert isinstance(self.elb, ELB)

        self.loss = None
        self._device = torch.device(cuda_id)

        self._zero = torch.tensor([0.0], device=self._device,
                                  requires_grad=False, dtype=torch.float)

        self.seg_ignore_idx = seg_ignore_idx

        # per class weights.
        self.base_per_class_weights = None
        self.per_class_weights = None
        self.per_class_w_style = constants.CLWNONE
        self.clw_initialized = False
        self.cl_w_updated = True
        self.n_cls = 1

        self.cutmix_holder = None

    def init_class_weights(self,
                           n_cls: int,
                           w: torch.Tensor = None,
                           style: str = constants.CLWNONE):

        assert isinstance(n_cls, int), type(n_cls)
        assert n_cls > 0, n_cls

        self.n_cls = n_cls

        assert style in constants.CLW, style

        if style == constants.CLWNONE:
            self.base_per_class_weights = None
            self.per_class_weights = None

        if style in [constants.CLWFIXEDTECH1, constants.CLWFIXEDTECH2]:
            assert w.ndim == 1, w.ndim
            assert w.numel() == n_cls, f"{w.numel()} {n_cls}"

            self.base_per_class_weights = w.to(self._device)
            self.per_class_weights = w.to(self._device)

        if style == constants.CLWADAPTIVE:
            ones = torch.ones(size=(n_cls,), dtype=torch.float32,
                              device=self._device, requires_grad=False)

            self.base_per_class_weights = ones
            self.per_class_weights = ones

        if style == constants.CLWMIXED:
            ones = torch.ones(size=(n_cls,), dtype=torch.float32,
                              device=self._device, requires_grad=False)

            self.base_per_class_weights = w.to(self._device)
            self.per_class_weights = ones

        self.clw_initialized = True
        self.per_class_w_style = style
        self.cl_w_updated = True

    def update_per_class_weights(self, w: torch.Tensor):
        assert self.clw_initialized, self.clw_initialized
        assert self.style in [constants.CLWADAPTIVE,
                              constants.CLWMIXED], self.style

        assert w.numel() == self.n_cls, f"{w.numel()}  {self.n_cls}"
        assert w.ndim == 1, w.ndim

        if self.style == constants.CLWADAPTIVE:
            self.per_class_weights = w.to(self._device)
            self.cl_w_updated = True

        if self.style == constants.CLWMIXED:
            base = self.base_per_class_weights
            self.per_class_weights = base + w.to(self._device)
            self.cl_w_updated = True


    def set_cutmix_holder(self, cutmix_holder: List):
        assert isinstance(cutmix_holder, list)
        # [target_a, target_b, lam]
        assert len(cutmix_holder) == 3, len(cutmix_holder)

        self.cutmix_holder = cutmix_holder

    def reset_cutmix_holder(self):
        self.cutmix_holder = None

    def is_on(self, _epoch=None):
        if _epoch is None:
            c_epoch = self.c_epoch
        else:
            assert isinstance(_epoch, int)
            c_epoch = _epoch

        if (self.start_epoch is None) and (self.end_epoch is None):
            return True

        l = [c_epoch, self.start_epoch, self.end_epoch]
        if all([isinstance(z, int) for z in l]):
            return self.start_epoch <= c_epoch <= self.end_epoch

        if self.start_epoch is None and isinstance(self.end_epoch, int):
            return c_epoch <= self.end_epoch

        if isinstance(self.start_epoch, int) and self.end_epoch is None:
            return c_epoch >= self.start_epoch

        return False

    def unpacke_low_cams(self, cams_low, glabel):
        n = cams_low.shape[0]
        select_lcams = [None for _ in range(n)]

        for i in range(n):
            llabels = [glabel[i]]

            if self.support_background:
                llabels = [xx + 1 for xx in llabels]
                llabels = [0] + llabels

            for l in llabels:
                tmp = cams_low[i, l, :, :].unsqueeze(
                        0).unsqueeze(0)
                if select_lcams[i] is None:
                    select_lcams[i] = tmp
                else:
                    select_lcams[i] = torch.cat((select_lcams[i], tmp), dim=1)

        return select_lcams

    def update_t(self):
        if isinstance(self.elb, ELB):
            self.elb.update_t()

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            out = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
            if isinstance(self.elb, ELB):
                out = out + '_elb'
            if self.logit:
                out = out + '_logit'
            return out
        else:
            return self._name

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
        self.c_epoch = epoch
        assert self.clw_initialized, self.clw_initialized