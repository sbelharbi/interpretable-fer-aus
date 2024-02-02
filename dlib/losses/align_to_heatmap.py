import sys
from os.path import dirname, abspath
from typing import Union, Tuple

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


__all__ = ['AlignToHeatMap', 'AunitsSegmentationLoss']


class BuildAttention(nn.Module):
    def __init__(self, attention_type: str, p: float, q: float):
        """

        :param attention_type: attention type.
        :param p: percentage of RANDOM feature maps to be considered for
        averaging. A Bernoulli distribution with probability p will
        be used to randomly sample which features are selected for averaging.
        p in ]0., 1.]. applicable only for: attention_type=ALIGN_RANDOM_AVG.
        :param q: percentage of FIXED feature maps to be considered for
        averaging. q in ]0., 1.].
        """
        super(BuildAttention, self).__init__()

        assert attention_type in constants.ALIGNMENTS, attention_type
        self.attention_type = attention_type

        assert isinstance(p, float), type(p)
        assert 0. < p <= 1., p
        self.p = p

        assert isinstance(q, float), type(q)
        assert 0. < q <= 1., q
        self.q = q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # b, c, h, w

        b, c, h, w = x.shape
        n = max(int(c / 2), 1)
        attention = None
        if self.attention_type == constants.ALIGN_AVG_HALF_LEFT:
            attention = x[:, :n, :, :].mean(dim=1, keepdim=True)

        elif self.attention_type == constants.ALIGN_AVG_HALF_RIGHT:
            attention = x[:, n:, :, :].mean(dim=1, keepdim=True)

        elif self.attention_type == constants.ALIGN_AVG_FIXED_Q:
            n = max(int(c * self.q), 1)
            attention = x[:, :n, :, :].mean(dim=1, keepdim=True)

        elif self.attention_type == constants.ALIGN_AVG:
            attention = x.mean(dim=1, keepdim=True)

        elif self.attention_type == constants.ALIGN_RANDOM_AVG:

            select = self.sample_random_selection(x)
            assert select.ndim == 4, select.ndim  # b, c, 1, 1
            assert select.shape == (b, c, 1, 1), f"{select.shape} " \
                                                 f"{(b, c, 1, 1)}"
            attention = x * select
            deno = select.sum(dim=(1, 2, 3), keepdim=False)  # b
            deno[deno == 0] = 1.  # avoid div by 0.
            deno = deno.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # b, 1, 1, 1
            attention = attention.sum(dim=1, keepdim=True)  #b, 1, h, w
            attention = attention / deno

        else:
            raise NotImplementedError(self.attention_type)

        assert attention is not None

        return attention  # b, 1, h, w

    def sample_random_selection(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # b, c, h, w

        b, c, h, w = x.shape
        probs = torch.ones((b, c, 1, 1), device=x.device, requires_grad=False)
        probs = probs * self.p
        # print(self.p, 'p')
        bern_dist = Bernoulli(probs=probs)
        select = bern_dist.sample()  # b, c, 1, 1
        assert select.shape == (b, c, 1, 1), f"{select.shape} {(b, c, 1, 1)}"

        return select


class _AlignLoss(nn.Module):
    def __init__(self, loss_type: str, norm_att: str, use_elb: bool,
                 elb: Union[ELB, None] = None):
        super(_AlignLoss, self).__init__()

        assert loss_type in constants.ALIGN_LOSSES, loss_type
        self.loss_type = loss_type

        assert norm_att in constants.NORMS_ATTENTION, norm_att
        self.norm_att = norm_att

        if use_elb:
            assert loss_type in [constants.A_COSINE,
                                 constants.A_STD_COSINE], loss_type

            assert isinstance(elb, ELB)

        self.use_elb = use_elb
        self.elb = elb

        if loss_type == constants.A_L1:
            self.loss = nn.L1Loss(reduction='mean')

        elif loss_type == constants.A_L2:
            self.loss = nn.MSELoss(reduction='mean')

        elif loss_type == constants.A_KL:
            self.loss = nn.KLDivLoss(reduction='batchmean', log_target=False)

        elif loss_type in [constants.A_COSINE, constants.A_STD_COSINE]:
            self.loss = nn.CosineSimilarity(dim=1, eps=1e-8)

        else:
            raise NotImplementedError(loss_type)

    def forward(self, att: torch.Tensor, heatmap: torch.Tensor) -> torch.Tensor:
        assert att.ndim == 4, att.ndim
        assert att.shape == heatmap.shape, f"{att.shape} {heatmap.shape}"

        b, c, h, w = att.shape

        if (self.norm_att == constants.NORM_SOFTMAX) or (self.loss_type ==
                                                         constants.A_KL):
            att = self.normalize_att_softmax(att)

        if self.loss_type == constants.A_L1:
            loss = self.loss(input=att, target=heatmap)

        elif self.loss_type == constants.A_L2:
            loss = self.loss(input=att, target=heatmap)

        elif self.loss_type == constants.A_KL:
            _att = att.contiguous().view(-1, 1)
            _heatmap = heatmap.contiguous().view(-1, 1)

            _att = torch.cat((1. - _att, _att), dim=1)
            _heatmap = torch.cat((1. - _heatmap, _heatmap), dim=1)

            loss = self.loss(input=_att.log(), target=_heatmap)

        elif self.loss_type in [constants.A_COSINE, constants.A_STD_COSINE]:

            _att = att.contiguous().view(b, -1)
            _heatmap = heatmap.contiguous().view(b, -1)

            if self.loss_type == constants.A_STD_COSINE:
                _att_avg = _att.mean(dim=-1, keepdim=True)
                _att_std = torch.std(input=_att, dim=-1, keepdim=True,
                                     correction=1) + 1.e-4
                _att = (_att - _att_avg) / _att_std

            cosine = self.loss(x1=_att, x2=_heatmap)

            if self.use_elb:
                loss = self.elb(- cosine.contiguous().view(-1, ))
            else:
                loss = (1. - cosine).mean()

        else:
            raise NotImplementedError(self.loss_type)

        return loss  # tensor scalar.

    def normalize_att_softmax(self, att: torch.Tensor) -> torch.Tensor:
        assert self.norm_att == constants.NORM_SOFTMAX or self.loss_type == \
               constants.A_KL, f"{self.norm_att}  {self.loss_type}"

        assert att.ndim == 4, att.ndim
        b, _, _, _ = att.shape

        exp = torch.exp(att)
        z = exp.contiguous().view(b, -1).sum(dim=1).view(b, -1, 1, 1)

        normed = exp / z

        return normed


class AlignToHeatMap(ElementaryLoss):
    def __init__(self, **kwargs):
        super(AlignToHeatMap, self).__init__(**kwargs)

        self.alignment = constants.ALIGN_AVG
        self.p = 1.
        self.q = 1.
        self.attention_layers = []
        self.att_builder = BuildAttention(attention_type=self.alignment,
                                          p=self.p,
                                          q=self.q
                                          ).to(self._device)
        self.loss = None
        self.scale_to = constants.SCALE_TO_ATTEN
        self.use_elb = False
        self.norm_att = constants.NORM_NONE

        self.already_set = False

    def set_it(self,
               alignment: str,
               p: float,
               q: float,
               atten_layers: list,
               loss_type: str,
               scale_to: str,
               use_elb: bool,
               norm_att: str
               ):

        assert alignment in constants.ALIGNMENTS, alignment
        assert loss_type in constants.ALIGN_LOSSES, loss_type
        assert scale_to in constants.SCALE_TO, scale_to
        assert norm_att in constants.NORMS_ATTENTION, norm_att

        if use_elb:
            assert loss_type in [constants.A_COSINE,
                                 constants.A_STD_COSINE], loss_type

            assert isinstance(self.elb, ELB)

        assert isinstance(p, float), type(p)
        assert 0. < p <= 1., p

        assert isinstance(q, float), type(q)
        assert 0. < q <= 1., q

        z = atten_layers.split('-')
        z = [int(s) for s in z]
        assert len(z) > 0, len(z)
        for i in z:
            assert i > 0, i  # layers count starts from 0. but layer 0 is not
            # allowed to be used for alignment. it holds the input image.
            # we dont check here the conformity to the maximum allowed value.
            # we do that in the forward. the maximum allowed value will be
            # determined automatically based on the length of the feature
            # holder.

        self.attention_layers = z

        self.alignment = alignment
        self.p = p
        self.q = q
        self.norm_att = norm_att

        self.att_builder = BuildAttention(attention_type=self.alignment,
                                          p=self.p,
                                          q=self.q
                                          ).to(self._device)
        self.loss = _AlignLoss(loss_type=loss_type,
                               norm_att=norm_att,
                               use_elb=use_elb,
                               elb=self.elb).to(self._device)
        self.scale_to = scale_to
        self.use_elb = use_elb
        self.already_set = True

    def process_one_att(self,
                        att: torch.Tensor,
                        heatmap: torch.Tensor
                        ) -> torch.Tensor:

        assert att.ndim == 4, att.ndim  # b, 1, h`, w`
        assert heatmap.ndim == 4, heatmap.ndim  # b, 1, h, w.
        assert att.shape[0] == heatmap.shape[0], f"{att.shape[0]}" \
                                                 f" {heatmap.shape[0]}"
        assert att.shape[1] == heatmap.shape[1], f"{att.shape[1]}" \
                                                 f" {heatmap.shape[1]}"
        assert att.shape[1] == 1, att.shape[1]

        if self.scale_to == constants.SCALE_TO_HEATM:

            _min, _max = self._get_min_max(att)

            att = F.interpolate(input=att,
                                size=heatmap.shape[2:],
                                mode='bilinear',
                                align_corners=True,
                                antialias=True
                                )

            att = self._clamp_minibatch(_min, _max, att)

        elif self.scale_to == constants.SCALE_TO_ATTEN:

            _min, _max = self._get_min_max(heatmap)

            heatmap = F.interpolate(input=heatmap,
                                    size=att.shape[2:],
                                    mode='bilinear',
                                    align_corners=True,
                                    antialias=True
                                   )

            heatmap = self._clamp_minibatch(_min, _max, heatmap)

        else:
            raise NotImplementedError(self.scale_to)

        return self.loss(att=att, heatmap=heatmap)

    @staticmethod
    def _get_min_max(maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert maps.ndim == 4, maps.ndim

        b, _, _, _ = maps.shape
        x = maps.contiguous().view(b, -1)
        _min = x.min(dim=1, keepdim=False)[0]
        _max = x.max(dim=1, keepdim=False)[0]

        return _min, _max

    @staticmethod
    def _clamp_minibatch(_min: torch.Tensor,
                         _max: torch.Tensor,
                         maps: torch.Tensor) -> torch.Tensor:
        assert maps.ndim == 4, maps.ndim
        b, _, _, _ = maps.shape
        _min_b = _min.numel()
        _max_b = _max.numel()
        assert _min_b == b, f"{_min_b} , {b}"
        assert _max_b == b, f"{_max_b} , {b}"

        for i in range(b):
            maps[i] = torch.clamp(maps[i], min=_min[i], max=_max[i])

        return maps

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
        super(AlignToHeatMap, self).forward(epoch=epoch)

        assert self.already_set, self.already_set

        if not self.is_on():
            return self._zero

        assert model is not None
        assert heatmap is not None
        assert self.loss is not None

        assert heatmap.ndim == 4, heatmap.ndim  # b, 1, h, w.
        assert heatmap.shape[1] == 1, heatmap.shape[1]

        layerwise_features: list = model.features
        # layers count starts from 0.
        n_l = len(layerwise_features)
        msg = f"{max(self.attention_layers)}, {n_l - 1}"
        assert max(self.attention_layers) <= n_l - 1, msg

        msg = f"{len(self.attention_layers)}, {n_l}"
        assert len(self.attention_layers) <= n_l, msg

        id_valid = self.find_valid_samples(heatmap=heatmap)

        if id_valid.numel() == 0:
            return self._zero

        heatmap = heatmap[id_valid]
        if heatmap.ndim == 3: # when id_valid has single item.
            assert id_valid.numel() == 1, id_valid.numel()
            heatmap = heatmap.unsqueeze(0)


        loss = 0.0
        z = 0.
        for l in self.attention_layers:
            features = layerwise_features[l]  # b, c, h, w.
            features = features[id_valid]
            if features.ndim == 3:  # when id_valid has single item.
                assert id_valid.numel() == 1, id_valid.numel()
                features = features.unsqueeze(0)

            att = self.att_builder(x=features)  # b, 1, h, w.
            loss = loss + self.process_one_att(att=att, heatmap=heatmap)
            z += 1.

        assert z > 0, z
        loss = loss / z
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        return self.lambda_ * loss

    @staticmethod
    def find_valid_samples(heatmap: torch.Tensor) -> torch.Tensor:
        assert heatmap.ndim == 4, heatmap.ndim  # b, 1, h, w
        assert heatmap.shape[1] == 1, heatmap.shape[1]
        b, c, h, w = heatmap.shape

        x = heatmap.contiguous().view(b, -1)
        z = torch.isinf(x).sum(dim=1)  # b
        id_valid = torch.where(z == 0)[0]

        return id_valid


class AunitsSegmentationLoss(ElementaryLoss):
    """
    Learn to segment action units.
    """
    def __init__(self, **kwargs):
        super(AunitsSegmentationLoss, self).__init__(**kwargs)

        self.loss = nn.BCEWithLogitsLoss(reduction="mean").to(self._device)

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

        super(AunitsSegmentationLoss, self).forward(epoch=epoch)

        if not self.is_on():
            return self._zero

        assert model is not None
        assert seg_map is not None
        assert bin_seg is not None
        assert self.loss is not None

        logits = model.segmentation_head.segment_logits
        assert logits.ndim == 4, logits.ndim  # bsz, 1, h, w
        assert logits.shape[1] == 1, logits.shape[1]
        assert logits.shape[0] == bin_seg.shape[0], f"{logits.shape[0]} | " \
                                              f"{bin_seg.shape[0]}"

        id_valid = self.find_valid_samples(heatmap=seg_map)

        if id_valid.numel() == 0:
            return self._zero

        bin_seg = bin_seg[id_valid]
        logits = logits[id_valid]

        if bin_seg.ndim == 3:  # when id_valid has single item.
            assert id_valid.numel() == 1, id_valid.numel()

            bin_seg = bin_seg.unsqueeze(0)
            logits = logits.unsqueeze(0)

        logits = F.interpolate(logits,
                               size=bin_seg.shape[2:],
                               mode='bicubic',
                               align_corners=True
                               )

        loss = self.loss(logits, bin_seg.float())

        return self.lambda_ * loss

    @staticmethod
    def find_valid_samples(heatmap: torch.Tensor) -> torch.Tensor:
        assert heatmap.ndim == 4, heatmap.ndim  # b, 1, h, w
        assert heatmap.shape[1] == 1, heatmap.shape[1]
        b, c, h, w = heatmap.shape

        x = heatmap.contiguous().view(b, -1)
        z = torch.isinf(x).sum(dim=1)  # b
        id_valid = torch.where(z == 0)[0]

        return id_valid



