import sys
from os.path import dirname, abspath
from typing import Union, Tuple
import math

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from torch.nn.functional import normalize


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.losses.elb import ELB
from dlib.losses.elementary import ElementaryLoss
from dlib.configure import constants


__all__ = ['FreeOrthFeatures', 'GuidedOrthFeatures', 'OrthoLinearWeightLoss']


class FreeOrthFeatures(ElementaryLoss):
    """
    Constrain features of samples:
    - same class: to have maximum cosine.
    - different class: to have cosine = 0.
    """
    def __init__(self, **kwargs):
        super(FreeOrthFeatures, self).__init__(**kwargs)

        self.use_layers = []
        self.use_elb = False

        self.same_cl = False
        self.diff_cl = False

        self.already_set = False

    def set_it(self,
               use_layers: list,
               same_cl: bool,
               diff_cl: bool,
               use_elb: bool
               ):
        assert isinstance(use_elb, bool), type(use_elb)
        assert isinstance(same_cl, bool), type(same_cl)
        assert isinstance(diff_cl, bool), type(diff_cl)

        z = use_layers.split('-')
        z = [int(s) for s in z]
        assert len(z) > 0, len(z)
        for i in z:
            assert i > 0, i  # layers count starts from 0. but layer 0 is not
            # allowed to be used for alignment. it holds the input image.
            # we dont check here the conformity to the maximum allowed value.
            # we do that in the forward. the maximum allowed value will be
            # determined automatically based on the length of the feature
            # holder.

        self.use_layers = z

        if use_elb:
            assert isinstance(self.elb, ELB)

        self.same_cl = same_cl
        self.diff_cl = diff_cl

        assert same_cl or diff_cl, f"{same_cl}, {diff_cl}"

        self.use_elb = use_elb
        self.already_set = True

    @staticmethod
    def build_labels_idx(glabel: torch.Tensor) -> torch.Tensor:
        assert isinstance(glabel, torch.Tensor), type(glabel)
        assert glabel.ndim == 1, glabel.ndim

        n = glabel.numel()

        # mtx[i, j] = 1 if label[i] == label[j] else 0

        rw = glabel.view(-1, 1).repeat(1, n)  # n, n
        rh = glabel.view(1, -1).repeat(n, 1)  # n, w

        mtx = ((rw - rh) == 0).float()

        return mtx

    def process_one_layer(self,
                          ft: torch.Tensor,
                          lbls_idx: torch.Tensor) -> torch.Tensor:

        assert self.same_cl or self.diff_cl, f"{self.same_cl}, {self.diff_cl}"

        assert ft.ndim == 2, ft.ndim  # b, feature_dim
        assert lbls_idx.ndim == 2, lbls_idx.ndim  # b, b
        msg = f"{ft.shape[0]}, {lbls_idx.shape[0]}"
        assert ft.shape[0] == lbls_idx.shape[0], msg
        msg = f"{lbls_idx.shape[0]}, {lbls_idx.shape[1]}"
        assert lbls_idx.shape[0] == lbls_idx.shape[1], msg

        b = ft.shape[0]
        loss = self._zero

        # pairwise cosine
        p_cosine = F.cosine_similarity(ft[:,:,None], ft.t()[None,:,:])  # b, b
        assert p_cosine.shape == (b, b), p_cosine.shape

        p_cosine = p_cosine.contiguous().view(b * b)  # b * b
        lbls = lbls_idx.contiguous().view(b * b)  # b * b


        # same classes
        if self.same_cl:  # max cosine
            idx_same = torch.nonzero(lbls, as_tuple=False).squeeze()
            if idx_same.numel() > 0:
                val_cosine = p_cosine[idx_same]

                if self.use_elb:
                    loss = self.elb(- val_cosine.contiguous().view(-1, ))
                else:
                    loss = (- val_cosine).mean()

        # different classes
        if self.diff_cl:  # cosine = 0 (orthogonal)
            idx_diff = torch.nonzero(1. - lbls, as_tuple=False).squeeze()
            if idx_diff.numel() > 0:
                val_cosine = p_cosine[idx_diff]

                if self.use_elb:
                    loss = loss + self.elb(
                        val_cosine.abs().contiguous().view(-1, ))
                else:
                    loss = loss + val_cosine.abs().mean()

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
        super(FreeOrthFeatures, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert model is not None
        assert self.same_cl or self.diff_cl, f"{self.same_cl}, {self.diff_cl}"

        layerwise_2dfeatures: list = model.features

        # layers count starts from 0.
        n_l = len(layerwise_2dfeatures)
        msg = f"{max(self.use_layers)}, {n_l - 1}"
        assert max(self.use_layers) <= n_l - 1, msg

        msg = f"{len(self.use_layers)}, {n_l}"
        assert len(self.use_layers) <= n_l, msg

        msg = f"{min(self.use_layers)}, 0"
        assert min(self.use_layers) > 0, msg

        # build flatten features
        flaten_features = []
        for l in self.use_layers:
            features_2d = layerwise_2dfeatures[l]
            # either ed or dense features.
            cnd = (features_2d.ndim == 4)
            cnd |= (features_2d.ndim == 2)
            assert cnd, features_2d.ndim

            if features_2d.ndim == 4:
                ft = F.adaptive_avg_pool2d(features_2d, (1, 1)) # b, c, 1, 1
                ft = ft.squeeze(dim=(3, 2))  # b, c

            else:
                ft = features_2d  # b, c

            flaten_features.append(ft)

        # labels indexer
        lbls_idx = self.build_labels_idx(glabel)

        z = 0.
        loss = 0.0
        for ft in flaten_features:
            loss_l = self.process_one_layer(ft=ft, lbls_idx=lbls_idx)

            loss = loss + loss_l

            z += 1.

        assert z > 0, z
        loss = loss / z
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        return self.lambda_ * loss


class GuidedOrthFeatures(ElementaryLoss):
    def __init__(self, **kwargs):
        super(GuidedOrthFeatures, self).__init__(**kwargs)

        self.n_cls = 0
        self.use_layers = []
        self.use_elb = False

        self.already_set = False

    def set_it(self,
               use_layers: list,
               n_cls: int,
               use_elb: bool
               ):
        assert isinstance(use_elb, bool), type(use_elb)
        assert isinstance(n_cls, int), type(n_cls)
        assert n_cls > 0, n_cls

        z = use_layers.split('-')
        z = [int(s) for s in z]
        assert len(z) > 0, len(z)
        for i in z:
            assert i > 0, i  # layers count starts from 0. but layer 0 is not
            # allowed to be used for alignment. it holds the input image.
            # we dont check here the conformity to the maximum allowed value.
            # we do that in the forward. the maximum allowed value will be
            # determined automatically based on the length of the feature
            # holder.

        self.use_layers = z

        if use_elb:
            assert isinstance(self.elb, ELB)

        self.n_cls = n_cls

        self.use_elb = use_elb
        self.already_set = True

    def build_basis(self, d: int) -> torch.Tensor:
        assert d > 0, d
        assert isinstance(d, int), type(d)
        mtx = torch.zeros((self.n_cls, d), dtype=torch.float32,
                          requires_grad=False, device=self._device
                          )
        l_idx = torch.arange(0, d, 1)
        bsis_sz = int(math.floor(d / self.n_cls))
        assert bsis_sz > 0, bsis_sz

        for i in range(self.n_cls):
            l = l_idx[i * bsis_sz:i * bsis_sz + bsis_sz]
            _min = l.min()
            _max = l.max()

            if i == self.n_cls - 1:
                mtx[i, _min:] = 1.

            else:
                mtx[i, _min: _max + 1] = 1.

        return mtx

    def process_one_layer(self,
                          ft: torch.Tensor,
                          glabel: torch.Tensor) -> torch.Tensor:

        assert ft.ndim == 2, ft.ndim  # b, feature_dim
        assert glabel.ndim == 1, glabel.ndim
        msg = f"{glabel.shape[0]}, {ft.shape[0]}"
        assert glabel.shape[0] == ft.shape[0], msg

        n = ft.shape[0]
        d = ft.shape[1]
        cl_basis = self.build_basis(d)
        basis = cl_basis[glabel]  # b, d
        assert basis.shape == (n, d), f"{basis.shape}, ({n}, {d})"

        # set inverse basis to 0: ensure orthogonality between ft of diff.cl
        inv_basis = 1. - basis
        _ft = ft.contiguous().view(-1)
        inv_basis = inv_basis.contiguous().view(-1)
        idx = torch.nonzero(inv_basis, as_tuple=False).squeeze()

        loss = self._zero
        if idx.numel() > 0:
            vals = _ft[idx]
            if self.use_elb:
                loss = self.elb(vals.abs().contiguous().view(-1, ))
            else:
                loss = vals.abs().mean()

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
        super(GuidedOrthFeatures, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert model is not None

        layerwise_2dfeatures: list = model.features
        # layers count starts from 0.
        n_l = len(layerwise_2dfeatures)
        msg = f"{max(self.use_layers)}, {n_l - 1}"
        assert max(self.use_layers) <= n_l - 1, msg

        msg = f"{len(self.use_layers)}, {n_l}"
        assert len(self.use_layers) <= n_l, msg

        msg = f"{min(self.use_layers)}, 0"
        assert min(self.use_layers) > 0, msg

        # build flatten features
        flaten_features = []
        for l in self.use_layers:
            features_2d = layerwise_2dfeatures[l]
            # either ed or dense features.
            cnd = (features_2d.ndim == 4)
            cnd |= (features_2d.ndim == 2)
            assert cnd, features_2d.ndim

            if features_2d.ndim == 4:
                ft = F.adaptive_avg_pool2d(features_2d, (1, 1))  # b, c, 1, 1
                ft = ft.squeeze(dim=(3, 2))  # b, c

            else:
                ft = features_2d  # b, c

            flaten_features.append(ft)

        z = 0.
        loss = 0.0
        for ft in flaten_features:
            loss_l = self.process_one_layer(ft=ft, glabel=glabel)

            loss = loss + loss_l

            z += 1.

        assert z > 0, z
        loss = loss / z
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        return self.lambda_ * loss


class OrthoLinearWeightLoss(ElementaryLoss):
    """
    Apply orthogonality over the linear weight of the last linear classifier at
    the net output.
    """
    def __init__(self, **kwargs):
        super(OrthoLinearWeightLoss, self).__init__(**kwargs)

        self.method: str = constants.ORTHOG_SOFT
        self.spec_iter: int = 3
        self.eps: float = 1e-12
        self.already_set: bool = False

    def set_it(self,
               method: str,
               spec_iter: int
               ):
        assert isinstance(method, str), type(method)
        assert method in constants.ORTHOG_TECHS, method

        assert isinstance(spec_iter, int), type(spec_iter)
        assert spec_iter > 0, spec_iter

        self.method = method
        self.spec_iter = spec_iter

        self.already_set = True

    def _orth_soft(self, w: torch.Tensor) -> torch.Tensor:

        assert self.method == constants.ORTHOG_SOFT, f"{self.method} | " \
                                                     f"{constants.ORTHOG_SOFT}"

        assert w.ndim == 2, w.ndim  # cls, nf
        c, d = w.shape
        ident = torch.eye(d, m=d, dtype=w.dtype, device=w.device,
                          requires_grad=False)

        wt = w.t()  # d, c
        wtw = torch.matmul(wt, w)  # d, d
        diff = (wtw - ident)**2

        return diff.sum()

    def _orth_d_soft(self, w: torch.Tensor) -> torch.Tensor:

        assert self.method == constants.ORTHOG_D_SOFT, f"{self.method} | " \
                                                     f"{constants.ORTHOG_D_SOFT}"

        assert w.ndim == 2, w.ndim  # cls, nf
        c, d = w.shape
        # 1
        ident = torch.eye(d, m=d, dtype=w.dtype, device=w.device,
                          requires_grad=False)

        wt = w.t()  # d, c
        wtw = torch.matmul(wt, w)  # d, d
        diff1 = (wtw - ident) ** 2

        # 2
        ident = torch.eye(c, m=c, dtype=w.dtype, device=w.device,
                          requires_grad=False)
        wwt = torch.matmul(w, wt)  # c, c
        diff2 = (wwt - ident) ** 2

        return diff1.sum() + diff2.sum()

    def _orth_mc(self, w: torch.Tensor) -> torch.Tensor:

        assert self.method == constants.ORTHOG_MC, f"{self.method} | " \
                                                     f"{constants.ORTHOG_MC}"

        assert w.ndim == 2, w.ndim  # cls, nf
        c, d = w.shape
        ident = torch.eye(d, m=d, dtype=w.dtype, device=w.device,
                          requires_grad=False)

        wt = w.t()  # d, c
        wtw = torch.matmul(wt, w)  # d, d
        norm = torch.norm((wtw - ident), p=float('inf'))

        return norm

    def spectral_norm_power_iter(self, x: torch.Tensor):
        assert self.method == constants.ORTHOG_SRIP, f"{self.method} | " \
                                                   f"{constants.ORTHOG_SRIP}"

        assert x.ndim == 2, x.ndim  # d, d
        c, d = x.shape
        assert c == d, f"{c} | {d}"

        u = normalize(x.new_empty(d).normal_(0, 1), dim=0, eps=self.eps)

        with torch.no_grad():
            for _ in range(self.spec_iter):
                # power iteration: https://arxiv.org/pdf/1802.05957.pdf
                # SpectralNorm: https://github.com/pytorch/pytorch/blob/eebe0ee
                # 14182be6e24d5ab171ab94b17a3f6923e/torch/nn/utils/spectral_
                # norm.py#L45

                v = normalize(torch.matmul(x.t(), u), dim=0, eps=self.eps)
                u = normalize(torch.matmul(x, v), dim=0, eps=self.eps)

        sigma = torch.dot(u, torch.matmul(x, v))
        return (torch.norm(sigma,p=2))**2



    def _orth_srip(self, w: torch.Tensor) -> torch.Tensor:

        assert self.method == constants.ORTHOG_SRIP, f"{self.method} | " \
                                                     f"{constants.ORTHOG_SRIP}"

        assert w.ndim == 2, w.ndim  # cls, nf
        c, d = w.shape
        ident = torch.eye(d, m=d, dtype=w.dtype, device=w.device,
                          requires_grad=False)

        wt = w.t()  # d, c
        wtw = torch.matmul(wt, w)  # d, d
        x = wtw - ident  # d, d

        return self.spectral_norm_power_iter(x)

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
        super(OrthoLinearWeightLoss, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert model is not None

        w = model.linear_w  # cls, nf
        assert w.ndim == 2, w.ndim  # cls, nf

        if self.method == constants.ORTHOG_SOFT:
            loss = self._orth_soft(w)

        elif self.method == constants.ORTHOG_D_SOFT:
            loss = self._orth_d_soft(w)

        elif self.method == constants.ORTHOG_MC:
            loss = self._orth_mc(w)

        elif self.method == constants.ORTHOG_SRIP:
            loss = self._orth_srip(w)

        else:
            raise NotImplementedError(self.method)


        return self.lambda_ * loss
