import sys
from os.path import join, dirname, abspath
import random
from typing import Tuple, Optional, List, Any, Union
import numbers
from collections.abc import Sequence
import warnings

import PIL.Image
import math
import time
import matplotlib.pyplot as plt
from typing import Iterator

from scipy.stats import multivariate_normal

import tqdm
from torch import Tensor
import torch
import numpy as np
import yaml
import os
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from PIL.Image import NEAREST
import cv2
from kornia.morphology import dilation

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import _interpolation_modes_from_int
import torchvision.transforms.functional as TF
import torchvision.transforms.functional_pil as F_pil
import torchvision.transforms.functional_tensor as F_t

import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode

try:
    import accimage
except ImportError:
    accimage = None

PROB_THRESHOLD = 0.5  # probability threshold.

"Credit: https://github.com/clovaai/wsolevaluation/blob/master/data_loaders.py"

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.functional import _functional as dlibf
from dlib.configure import constants

from dlib.utils.shared import reformat_id
from dlib.utils.shared import configure_metadata
from dlib.utils.shared import get_image_ids
from dlib.utils.shared import get_class_labels
from dlib.utils.shared import get_landmarks
from dlib.utils.shared import get_bounding_boxes
from dlib.utils.shared import get_mask_paths
from dlib.utils.shared import get_image_sizes
from dlib.utils.tools import chunk_it
from dlib.face_landmarks.action_units import build_all_action_units
from dlib.utils.shared import cl_w_tech1
from dlib.utils.shared import cl_w_tech2


_SPLITS = (constants.TRAINSET, constants.VALIDSET, constants.TESTSET)

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


def get_cams_paths(root_data_cams: str, image_ids: list) -> dict:
    paths = dict()
    for idx_ in image_ids:
        paths[idx_] = join(root_data_cams, '{}.pt'.format(reformat_id(idx_)))

    return paths

def _preprape_au_hmap(heatmap: np.ndarray,
                      normalize: bool
                      ) -> np.ndarray:

    assert isinstance(heatmap, np.ndarray), type(map)
    assert heatmap.ndim == 2, heatmap.ndim

    h, w = heatmap.shape
    if np.isinf(heatmap[0, 0]):  # invalid heatmap.
        return np.zeros((h, w), dtype=np.float32) + np.inf

    assert isinstance(normalize, bool), type(normalize)

    _heatmap: np.ndarray = heatmap

    if normalize:
        _min = heatmap.min()
        _max = heatmap.max()
        _deno = _max - _min

        if _deno == 0:
            _deno = 1.
        _heatmap = (_heatmap - _min) / _deno

    _heatmap = _heatmap.astype(np.float32)

    return _heatmap  # h, w


class WSOLImageLabelDataset(Dataset):
    def __init__(self,
                 args,
                 data_root,
                 metadata_root,
                 transform,
                 proxy,
                 resize_size,
                 crop_size,
                 set_mode: str,
                 split: str,
                 num_sample_per_class=0,
                 root_data_cams='',
                 image_ids: Optional[list] = None
                 ):
        self.args = args
        assert split in constants.SPLITS, f"{split} {constants.SPLITS}"
        self.split = split
        assert set_mode in constants.SET_MODES, set_mode
        self.set_mode = set_mode
        self.data_root = data_root
        self.metadata = configure_metadata(metadata_root)
        self.transform = transform

        self.use_daug_mask_img_heatmap = any(
            [isinstance(op, RandomImMaskViaHeatMap
                        ) for op in transform.transforms])

        # sanity check
        if self.use_daug_mask_img_heatmap:
            for op in transform.transforms:
                if isinstance(op, RandomImMaskViaHeatMap):
                    msg = f"{op.set_mode} | {self.set_mode}"
                    assert op.set_mode == self.set_mode, msg

        if image_ids is not None:
            self.image_ids = image_ids
        else:
            self.image_ids: list = get_image_ids(self.metadata, proxy=proxy)

        self.image_labels: dict = get_class_labels(self.metadata)
        self.num_sample_per_class = num_sample_per_class

        # debug: -----------------------------------
        # if split == constants.TRAINSET:
        #     origin = self.image_ids.copy()
        #     self.image_ids = []
        #     for idx in origin:
        #         if self.image_labels[idx] in [0, 1]:  # surprise vs fear.
        #             self.image_ids.append(idx)
        # ------------------------------------------

        self.cams_paths = None
        if os.path.isdir(root_data_cams):
            self.cams_paths = get_cams_paths(root_data_cams=root_data_cams,
                                             image_ids=self.image_ids)

        if args.align_atten_to_heatmap:
            assert args.align_atten_to_heatmap_type_heatmap in \
                   constants.HEATMAP_TYPES, args.align_atten_to_heatmap_type_heatmap

        self.landmarks = None
        facial_anchors = [constants.HEATMAP_LNDMKS,
                          constants.HEATMAP_AUNITS_LNMKS,
                          constants.HEATMAP_GENERIC_AUNITS_LNMKS
                          ]
        cnd = args.align_atten_to_heatmap
        cnd &= (args.align_atten_to_heatmap_type_heatmap in facial_anchors)

        if self.set_mode == constants.DS_TRAIN:
            cnd2 = args.train_daug_mask_img_heatmap
            cnd2 &= (args.train_daug_mask_img_heatmap_type in facial_anchors)

        elif self.set_mode == constants.DS_EVAL:
            cnd2 = args.eval_daug_mask_img_heatmap
            cnd2 &= (args.eval_daug_mask_img_heatmap_type in facial_anchors)

        else:
            raise NotImplementedError(self.set_mode)

        cnd3 = args.model['do_segmentation']
        cnd3 &= args.aus_seg
        cnd3 &= (args.aus_seg_heatmap_type in [
            constants.HEATMAP_LNDMKS,
            constants.HEATMAP_AUNITS_LNMKS,
            constants.HEATMAP_GENERIC_AUNITS_LNMKS
        ])

        if cnd or cnd2 or cnd3:
            self.landmarks = get_landmarks(self.metadata)


        self.resize_size = resize_size
        self.crop_size = crop_size

        folds_path = join(root_dir, self.args.metadata_root)
        path_class_id = join(folds_path, 'class_id.yaml')
        with open(path_class_id, 'r') as fcl:
            cl_int = yaml.safe_load(fcl)

        self.cl_to_int = cl_int
        self.int_to_cl: dict = self.switch_key_val_dict(cl_int)

        self._adjust_samples_per_class()

    @staticmethod
    def switch_key_val_dict(d: dict) -> dict:
        out = dict()
        for k in d:
            assert d[k] not in out, 'more than 1 key with same value. wrong.'
            out[d[k]] = k

        return out

    def _adjust_samples_per_class(self):
        if self.num_sample_per_class == 0:
            return
        image_ids = np.array(self.image_ids)
        image_labels = np.array([self.image_labels[_image_id]
                                 for _image_id in self.image_ids])
        unique_labels = np.unique(image_labels)

        new_image_ids = []
        new_image_labels = {}
        for _label in unique_labels:
            indices = np.where(image_labels == _label)[0]
            sampled_indices = np.random.choice(
                indices, self.num_sample_per_class, replace=False)
            sampled_image_ids = image_ids[sampled_indices].tolist()
            sampled_image_labels = image_labels[sampled_indices].tolist()
            new_image_ids += sampled_image_ids
            new_image_labels.update(
                **dict(zip(sampled_image_ids, sampled_image_labels)))

        self.image_ids = new_image_ids
        self.image_labels = new_image_labels

    def _load_lndmks_heatmap(self,
                             image_id: str,
                             img_w: int,
                             img_h: int,
                             use_precomputed: bool,
                             path_precomputed: str,
                             lndmk_variance: float,
                             normalize: bool,
                             jaw: bool
                             ) -> torch.Tensor:

        assert self.landmarks is not None

        # use_precomputed = self.args.align_atten_to_heatmap_use_precomputed
        # path_precomputed = self.args.align_atten_to_heatmap_folder
        # lndmk_variance = self.args.align_atten_to_heatmap_lndmk_variance
        # _normalize = self.args.align_atten_to_heatmap_normalize
        # _jaw = self.args.align_atten_to_heatmap_jaw

        if use_precomputed:
            assert os.path.isdir(path_precomputed), path_precomputed

        if not use_precomputed:
            _w = img_w
            _h = img_h

            cov = np.zeros((2, 2), dtype=np.float32)
            np.fill_diagonal(cov, lndmk_variance)
            lndmks = self.landmarks[image_id]
            if not jaw:
                lndmks = lndmks[17:]
            lndmks_heatmap: np.ndarray = landmarks_to_heatmap(
                h=_h, w=_w, lndmks=lndmks, cov=cov, normalize=normalize
            )
        else:
            path_heatmap = join(path_precomputed,
                                f"{reformat_id(image_id)}.npy")

            lndmks_heatmap: np.ndarray = np.load(path_heatmap,
                                                 allow_pickle=False,
                                                 fix_imports=True)
            assert lndmks_heatmap.ndim == 2, lndmks_heatmap.ndim  # h, w.

        lndmks_heatmap = torch.from_numpy(lndmks_heatmap).to(torch.float32)
        assert lndmks_heatmap.ndim == 2, lndmks_heatmap.ndim  # h, w.

        lndmks_heatmap: torch.Tensor = lndmks_heatmap.unsqueeze(0)  # 1, h',
        # w'.

        return lndmks_heatmap

    def _load_au_heatmap(self,
                         image_id: str,
                         img_w: int,
                         img_h: int,
                         use_precomputed: bool,
                         path_precomputed: str,
                         image_label: int,
                         normalize: bool,
                         aus_type: str
                         ) -> torch.Tensor:

        assert aus_type in [constants.HEATMAP_AUNITS_LNMKS,
                            constants.HEATMAP_GENERIC_AUNITS_LNMKS,
                            constants.HEATMAP_AUNITS_LEARNED_SEG], aus_type

        if aus_type in [constants.HEATMAP_AUNITS_LNMKS,
                        constants.HEATMAP_GENERIC_AUNITS_LNMKS]:
            assert self.landmarks is not None

        elif aus_type == constants.HEATMAP_AUNITS_LEARNED_SEG:
            assert use_precomputed

        else:
            raise NotImplementedError(aus_type)

        if use_precomputed:
            assert os.path.isdir(path_precomputed), path_precomputed

            path_heatmap = join(path_precomputed,
                                f"{reformat_id(image_id)}.npy")

            heatmap: np.ndarray = np.load(path_heatmap,
                                          allow_pickle=False,
                                          fix_imports=True
                                          )  # h, w

        else:
            _w = img_w
            _h = img_h

            cl: str = self.int_to_cl[image_label]

            heatmaps: np.ndarray = build_all_action_units(
                lndmks=self.landmarks[image_id], h=_h, w=_w, cl=cl,
                aus_type=aus_type
            )  # z, h, w
            heatmap = heatmaps.max(axis=0, initial=-1)  # h, w

        assert heatmap.ndim == 2, heatmap.ndim  # h, w.
        au_heatmap: np.ndarray = _preprape_au_hmap(heatmap=heatmap,
                                                   normalize=normalize)  # h, w

        au_heatmap = torch.from_numpy(au_heatmap).to(torch.float32)
        assert au_heatmap.ndim == 2, au_heatmap.ndim  # h, w.

        au_heatmap: torch.Tensor = au_heatmap.unsqueeze(0)  # 1, h, w.

        return au_heatmap

    def _load_all_per_class_aus_heatmap(self,
                            image_id: str,
                            img_w: int,
                            img_h: int
                            ) -> torch.Tensor:
        args = self.args

        assert self.use_daug_mask_img_heatmap
        assert self.set_mode == constants.DS_EVAL, self.set_mode
        assert args.eval_daug_mask_img_heatmap, \
            args.eval_daug_mask_img_heatmap
        assert args.eval_daug_mask_img_heatmap_type == \
               constants.HEATMAP_PER_CLASS_AUNITS_LNMKS, \
            args.eval_daug_mask_img_heatmap_type

        heatmap_type = args.eval_daug_mask_img_heatmap_type

        use_precomputed = args.eval_daug_mask_img_heatmap_use_precomputed
        path_precomputed = args.eval_daug_mask_img_heatmap_folder
        normalize = args.eval_daug_mask_img_heatmap_normalize

        cl_int_cls = sorted(list(self.int_to_cl.keys()), reverse=False)
        l_classes = [self.int_to_cl[k] for k in cl_int_cls]
        n_cls = len(l_classes)

        _w = img_w
        _h = img_h

        if use_precomputed:
            assert os.path.isdir(path_precomputed), path_precomputed

            path_heatmap = join(path_precomputed,
                                f"{reformat_id(image_id)}.npy")

            heatmaps: np.ndarray = np.load(path_heatmap,
                                           allow_pickle=False,
                                           fix_imports=True
                                           )  # n_cl, h, w

            aus: np.ndarray = heatmaps

        else:
            lndmks = self.landmarks[image_id]

            if lndmks[0][0] == np.inf:
                heatmaps = np.zeros((n_cls, _h, _w)) + np.inf
                return torch.from_numpy(heatmaps).to(torch.float32)

            _aus_type = constants.HEATMAP_AUNITS_LNMKS

            all_per_class_heatmaps: np.ndarray = None

            for cl in l_classes:

                assert isinstance(cl, str), type(cl)

                _heatmaps: np.ndarray = build_all_action_units(
                    lndmks=lndmks, h=_h, w=_w, cl=cl, aus_type=_aus_type
                )  # z, h, w
                _heatmap = _heatmaps.max(axis=0, initial=-1)  # h, w

                assert _heatmap.ndim == 2, _heatmap.ndim

                _heatmap = np.expand_dims(_heatmap, axis=0)  # 1, h, w
                if all_per_class_heatmaps is None:
                    all_per_class_heatmaps = _heatmap

                else:
                    all_per_class_heatmaps = np.concatenate(
                        (all_per_class_heatmaps, _heatmap), axis=0)

            aus: np.ndarray = all_per_class_heatmaps

        assert aus.ndim == 3, aus.ndim  # n_cls, h, w.
        assert aus.shape == (n_cls, _h, _w), f"{aus.shape} | {(n_cls, _h, _w)}"

        for k in range(n_cls):
            hmap = aus[k]  # h, w
            hmap = _preprape_au_hmap(heatmap=hmap,
                                     normalize=normalize)  # h, w
            aus[k] = hmap

        aus: torch.Tensor = torch.from_numpy(aus).to(torch.float32)  # n_cls,h,w

        return aus

    def _load_d_aug_heatmap(self,
                            image_id: str,
                            img_w: int,
                            img_h: int,
                            image_label: int
                            ) -> torch.Tensor:
        assert self.use_daug_mask_img_heatmap

        args = self.args

        if self.set_mode == constants.DS_TRAIN:

            assert args.train_daug_mask_img_heatmap

            heatmap_type = args.train_daug_mask_img_heatmap_type

            use_precomputed = args.train_daug_mask_img_heatmap_use_precomputed
            path_precomputed = args.train_daug_mask_img_heatmap_folder
            normalize = args.train_daug_mask_img_heatmap_normalize

            if heatmap_type == constants.HEATMAP_LNDMKS:

                lndmk_variance = args.train_daug_mask_img_heatmap_lndmk_variance
                jaw = args.train_daug_mask_img_heatmap_jaw

                return self._load_lndmks_heatmap(image_id=image_id,
                                                 img_w=img_w,
                                                 img_h=img_h,
                                                 use_precomputed=use_precomputed,
                                                 path_precomputed=path_precomputed,
                                                 lndmk_variance=lndmk_variance,
                                                 normalize=normalize,
                                                 jaw=jaw
                                                 )  # 1, h', w'.

            if heatmap_type in [constants.HEATMAP_AUNITS_LNMKS,
                                constants.HEATMAP_GENERIC_AUNITS_LNMKS,
                                constants.HEATMAP_AUNITS_LEARNED_SEG
                                ]:

                aus_type = heatmap_type

                return self._load_au_heatmap(image_id=image_id,
                                             img_w=img_w,
                                             img_h=img_h,
                                             use_precomputed=use_precomputed,
                                             path_precomputed=path_precomputed,
                                             image_label=image_label,
                                             normalize=normalize,
                                             aus_type=aus_type
                                             )  # 1, h, w.

            raise NotImplementedError(heatmap_type)


        if self.set_mode == constants.DS_EVAL:

            assert args.eval_daug_mask_img_heatmap
            heatmap_type = args.eval_daug_mask_img_heatmap_type

            use_precomputed = args.eval_daug_mask_img_heatmap_use_precomputed
            path_precomputed = args.eval_daug_mask_img_heatmap_folder
            normalize = args.eval_daug_mask_img_heatmap_normalize

            if heatmap_type == constants.HEATMAP_LNDMKS:

                lndmk_variance = args.eval_daug_mask_img_heatmap_lndmk_variance
                jaw = args.eval_daug_mask_img_heatmap_jaw

                return self._load_lndmks_heatmap(image_id=image_id,
                                                 img_w=img_w,
                                                 img_h=img_h,
                                                 use_precomputed=use_precomputed,
                                                 path_precomputed=path_precomputed,
                                                 lndmk_variance=lndmk_variance,
                                                 normalize=normalize,
                                                 jaw=jaw
                                                 )  # 1, h', w'.


            if heatmap_type in [constants.HEATMAP_AUNITS_LNMKS,
                                constants.HEATMAP_GENERIC_AUNITS_LNMKS,
                                constants.HEATMAP_AUNITS_LEARNED_SEG
                                ]:

                aus_type = heatmap_type

                return self._load_au_heatmap(image_id=image_id,
                                             img_w=img_w,
                                             img_h=img_h,
                                             use_precomputed=use_precomputed,
                                             path_precomputed=path_precomputed,
                                             image_label=image_label,
                                             normalize=normalize,
                                             aus_type=aus_type
                                             )  # 1, h, w.

            raise NotImplementedError(heatmap_type)

        raise NotImplementedError(self.set_mode)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = Image.open(join(self.data_root, image_id))
        image = image.convert('RGB')
        unmasked_img = image.copy()
        raw_img = image.copy()

        std_cam = None
        if self.cams_paths is not None:
            std_cam_path = self.cams_paths[image_id]
            # h', w'
            std_cam: torch.Tensor = torch.load(f=std_cam_path,
                                               map_location=torch.device('cpu'))
            assert std_cam.ndim == 2
            std_cam = std_cam.unsqueeze(0)  # 1, h', w'

        lndmks_heatmap = None

        cnd1 = self.args.align_atten_to_heatmap
        cnd1 &= (self.args.align_atten_to_heatmap_type_heatmap ==
                 constants.HEATMAP_LNDMKS)

        if cnd1:
            _w, _h = image.size
            args = self.args
            lndmks_heatmap = self._load_lndmks_heatmap(
                image_id=image_id,
                img_w=_w,
                img_h=_h,
                use_precomputed=args.align_atten_to_heatmap_use_precomputed,
                path_precomputed=args.align_atten_to_heatmap_folder,
                lndmk_variance=args.align_atten_to_heatmap_lndmk_variance,
                normalize=args.align_atten_to_heatmap_normalize,
                jaw=args.align_atten_to_heatmap_jaw
            )  # 1, h', w'.

        # heatmaps of action units.
        au_heatmap = None
        cnd2 = self.args.align_atten_to_heatmap
        cnd2 &= (self.args.align_atten_to_heatmap_type_heatmap in
                 [constants.HEATMAP_AUNITS_LNMKS,
                  constants.HEATMAP_GENERIC_AUNITS_LNMKS]
                 )

        if cnd2:
            _w, _h = image.size
            args = self.args

            if self.set_mode == constants.DS_TRAIN:
                assert self.split == constants.TRAINSET, self.split

            au_heatmap = self._load_au_heatmap(
                image_id=image_id,
                img_w=_w,
                img_h=_h,
                use_precomputed=args.align_atten_to_heatmap_use_precomputed,
                path_precomputed=args.align_atten_to_heatmap_folder,
                image_label=image_label,
                normalize=args.align_atten_to_heatmap_normalize,
                aus_type=args.align_atten_to_heatmap_type_heatmap
            )  # 1, h, w.

        # maskout data augmentation
        d_aug_heatmap = None

        if self.set_mode == constants.DS_TRAIN:
            cnd = self.args.train_daug_mask_img_heatmap
            cnd &= (self.args.train_daug_mask_img_heatmap_type in [
                constants.HEATMAP_AUNITS_LNMKS,
                constants.HEATMAP_GENERIC_AUNITS_LNMKS,
                constants.HEATMAP_LNDMKS,
                constants.HEATMAP_AUNITS_LEARNED_SEG
            ])

        elif self.set_mode == constants.DS_EVAL:

            cnd = self.args.eval_daug_mask_img_heatmap
            cnd &= (self.args.eval_daug_mask_img_heatmap_type in [
                constants.HEATMAP_AUNITS_LNMKS,
                constants.HEATMAP_GENERIC_AUNITS_LNMKS,
                constants.HEATMAP_LNDMKS,
                constants.HEATMAP_AUNITS_LEARNED_SEG
            ])

        else:
            raise NotImplementedError(self.set_mode)

        if cnd:
            assert self.use_daug_mask_img_heatmap
            _w, _h = image.size
            d_aug_heatmap = self._load_d_aug_heatmap(image_id=image_id,
                                                     img_w=_w,
                                                     img_h=_h,
                                                     image_label=image_label
                                                     ) # 1, h, w.

        # HEATMAP_PER_CLASS_AUNITS_LNMKS augmentation:
        aug_imgs_per_cl_aus = None

        cnd = (self.set_mode == constants.DS_EVAL)
        cnd &= (self.args.eval_daug_mask_img_heatmap_type ==
                constants.HEATMAP_PER_CLASS_AUNITS_LNMKS)

        if cnd:
            assert d_aug_heatmap is None
            assert self.use_daug_mask_img_heatmap
            _w, _h = image.size
            all_cls_aus_aug_heatmap = self._load_all_per_class_aus_heatmap(
                image_id=image_id,
                img_w=_w,
                img_h=_h
            )  # n_cls, h, w

            aug_imgs_per_cl_aus: list = self.aug_img_with_all_cls_aus(
                all_cls_aus_aug_heatmap,
                image,
                unmasked_img,
                raw_img,
                std_cam,
                lndmks_heatmap,
                au_heatmap
            )

        # heatmap segmentation
        _w, _h = image.size
        heatmap_seg = self._get_heatmap_seg(image_id, _h, _w, image_label)
        # 1, h, w. torch.Tensor or None.
        bin_heatmap_seg = None
        if heatmap_seg is not None:
            assert heatmap_seg.shape[0] == 1, heatmap_seg.shape[0]
            assert heatmap_seg.ndim == 3, heatmap_seg.ndim  # 1, h, w

            if not np.isinf(heatmap_seg[0, 0, 0]):
                z_np = heatmap_seg.squeeze(0).numpy()  # h, 1
                otsu_thresh = threshold_otsu(z_np, nbins=256)
                bin_heatmap_seg = (heatmap_seg >= otsu_thresh).float()
                # 1, h, w. tensor
                assert isinstance(bin_heatmap_seg, torch.Tensor), type(
                    bin_heatmap_seg)
            else:
                bin_heatmap_seg = heatmap_seg.float()  # 1, h, w


        z = self.transform(image,
                           unmasked_img,
                           raw_img,
                           std_cam,
                           lndmks_heatmap,
                           au_heatmap,
                           d_aug_heatmap,
                           heatmap_seg,
                           bin_heatmap_seg
                           )

        image, unmasked_img, raw_img, std_cam, lndmks_heatmap, au_heatmap, \
        d_aug_heatmap, heatmap_seg, bin_heatmap_seg  = z

        raw_img = np.array(raw_img, dtype=np.float32)  # h, w, 3
        raw_img = dlibf.to_tensor(raw_img).permute(2, 0, 1)  # 3, h, w.

        if std_cam is None:
            std_cam = 0

        # Landmarks heatmap.
        if lndmks_heatmap is None:
            lndmks_heatmap = 0

        else:
            if not torch.isinf(lndmks_heatmap[0, 0, 0]).item():
                lndmks_heatmap = torch.clip(lndmks_heatmap, 0.0, 1.0)

            _, _h, _w = lndmks_heatmap.shape
            ch, cw = int(_h / 2), int(_w / 2)
            assert not torch.isnan(lndmks_heatmap[0, ch, cw]).item(), image_id

        # Action units heatmap
        if au_heatmap is None:
            au_heatmap = 0

        else:
            if not torch.isinf(au_heatmap[0, 0, 0]).item():
                au_heatmap = torch.clip(au_heatmap, 0.0, 1.0)

            _, _h, _w = au_heatmap.shape
            ch, cw = int(_h / 2), int(_w / 2)
            assert not torch.isnan(au_heatmap[0, ch, cw]).item(), image_id

        if heatmap_seg is None:
            heatmap_seg = 0

        else:
            if not torch.isinf(heatmap_seg[0, 0, 0]).item():
                heatmap_seg = torch.clip(heatmap_seg, 0.0, 1.0)

            _, _h, _w = heatmap_seg.shape
            ch, cw = int(_h / 2), int(_w / 2)
            assert not torch.isnan(heatmap_seg[0, ch, cw]).item(), image_id

        if bin_heatmap_seg is None:
            bin_heatmap_seg = 0

        else:
            if not torch.isinf(bin_heatmap_seg[0, 0, 0]).item():
                bin_heatmap_seg = torch.clip(bin_heatmap_seg, 0.0, 1.0)

            _, _h, _w = bin_heatmap_seg.shape
            ch, cw = int(_h / 2), int(_w / 2)
            assert not torch.isnan(bin_heatmap_seg[0, ch, cw]).item(), image_id

        if self.set_mode == constants.DS_TRAIN:
            assert aug_imgs_per_cl_aus is None

            return image, unmasked_img, image_label, image_id, raw_img, \
                   std_cam, lndmks_heatmap, au_heatmap, heatmap_seg, \
                   bin_heatmap_seg

        elif self.set_mode == constants.DS_EVAL:
            if aug_imgs_per_cl_aus is None:
                aug_imgs_per_cl_aus = 0

            else:
                assert isinstance(aug_imgs_per_cl_aus, list), type(
                    aug_imgs_per_cl_aus)
                z = [t.unsqueeze(0) for t in aug_imgs_per_cl_aus]
                aug_imgs_per_cl_aus = torch.cat(z, dim=0)  # ncls, 3, h, w.

            return image, unmasked_img, image_label, image_id, raw_img, \
                   std_cam, lndmks_heatmap, au_heatmap, aug_imgs_per_cl_aus, \
                   heatmap_seg, bin_heatmap_seg

    def _get_heatmap_seg(self,
                         image_id: str,
                         h: int,
                         w: int,
                         image_label: int
                         ) -> Union[torch.Tensor, None]:

        args = self.args

        cnd1 = args.model['do_segmentation']
        cnd1 &= args.aus_seg
        cnd1 &= (args.aus_seg_heatmap_type == constants.HEATMAP_LNDMKS)

        cnd2 = args.model['do_segmentation']
        cnd2 &= args.aus_seg
        cnd2 &= (args.aus_seg_heatmap_type in
                 [constants.HEATMAP_AUNITS_LNMKS,
                  constants.HEATMAP_GENERIC_AUNITS_LNMKS]
                 )

        # heatmap landmarks
        if cnd1:
            return self._load_lndmks_heatmap(
                image_id=image_id,
                img_w=w,
                img_h=h,
                use_precomputed=args.aus_seg_use_precomputed,
                path_precomputed=args.aus_seg_folder,
                lndmk_variance=args.aus_seg_lndmk_variance,
                normalize=args.aus_seg_normalize,
                jaw=args.aus_seg_jaw
            )  # 1, h', w'.

        elif cnd2:

            if self.set_mode == constants.DS_TRAIN:
                assert self.split == constants.TRAINSET, self.split

            return self._load_au_heatmap(
                image_id=image_id,
                img_w=w,
                img_h=h,
                use_precomputed=args.aus_seg_use_precomputed,
                path_precomputed=args.aus_seg_folder,
                image_label=image_label,
                normalize=args.aus_seg_normalize,
                aus_type=args.aus_seg_heatmap_type
            )  # 1, h, w.

        else:
            return None


    def aug_img_with_all_cls_aus(self,
                                 all_cls_aus_aug_heatmap: torch.Tensor,
                                 image: PIL.Image.Image,
                                 unmasked_img: PIL.Image.Image,
                                 raw_img: PIL.Image.Image,
                                 std_cam: torch.Tensor,
                                 lndmks_heatmap: torch.Tensor,
                                 au_heatmap: torch.Tensor
                                 ) -> list:

        aug_maps = all_cls_aus_aug_heatmap

        assert isinstance(aug_maps, torch.Tensor), type(aug_maps)
        assert aug_maps.ndim == 3, aug_maps.ndim
        # n_cls, h, w

        c = len(list(self.int_to_cl.keys()))
        assert aug_maps.shape[0] == c, f"{aug_maps.shape[0]} | {c}"

        out = []
        for i in range(c):
            d_aug_heatmap = aug_maps[i].unsqueeze(0)  # 1, h, w

            image_aug, _, _, _, _, _, _, _, _ = self.transform(image,
                                                               unmasked_img,
                                                               raw_img,
                                                               std_cam,
                                                               lndmks_heatmap,
                                                               au_heatmap,
                                                               d_aug_heatmap,
                                                               None,
                                                               None
                                                               )
            out.append(image_aug)

        return out

    def __len__(self):
        return len(self.image_ids)


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class Compose(object):
    def __init__(self, mytransforms: list):
        self.transforms = mytransforms

        for t in mytransforms:
            assert any([isinstance(t, Resize),
                        isinstance(t, RandomCrop),
                        isinstance(t, RandomImMaskViaHeatMap),
                        isinstance(t, RandomResizedCrop),
                        isinstance(t, RandomRotation),
                        isinstance(t, RandomHorizontalFlip),
                        isinstance(t, ToTensor),
                        isinstance(t, RandomGrayscale),
                        isinstance(t, RandomColorJitter),
                        isinstance(t, transforms.RandomErasing),
                        isinstance(t, transforms.Normalize)]
                       )

    def chec_if_random(self, transf):
        if isinstance(transf, RandomCrop):
            return True

    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):
        for op in self.transforms:
            if isinstance(op, (RandomHorizontalFlip,
                              RandomCrop,
                              RandomImMaskViaHeatMap,
                              RandomResizedCrop,
                              Resize,
                              RandomRotation,
                              RandomGrayscale,
                              RandomColorJitter,
                              ToTensor
                              )
                          ):

                z = op(img,
                       unmasked_img,
                       raw_img,
                       std_cam,
                       lndmks_heatmap,
                       au_heatmap,
                       d_aug_heatmap,
                       heatmap_seg,
                       bin_heatmap_seg
                )

                img, unmasked_img, raw_img, std_cam, lndmks_heatmap, \
                au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg = z

            else:
                img = op(img)

        return img, unmasked_img, raw_img, std_cam, lndmks_heatmap, \
               au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ColorJitter(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError(f"{name} values should be between {bound}")
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(
        brightness: Optional[List[float]],
        contrast: Optional[List[float]],
        saturation: Optional[List[float]],
        hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def sample_params(self) -> Tuple[Tensor, Optional[float], Optional[float],
                                     Optional[float], Optional[float]]:
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        return fn_idx, brightness_factor, contrast_factor, saturation_factor,\
               hue_factor

    def forward(self,
                img,
                fn_idx: torch.Tensor,
                brightness_factor: float,
                contrast_factor: float,
                saturation_factor: float,
                hue_factor: float
                ):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = TF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = TF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = TF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = TF.adjust_hue(img, hue_factor)

        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s


class _BasicTransform(object):
    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):
        raise NotImplementedError

    def _prep_op_hmap(self, hmap: torch.Tensor) -> bool:
        assert hmap is not None
        assert isinstance(hmap, torch.Tensor), type(hmap)
        assert hmap.ndim == 3, hmap.ndim

        c, h, w = hmap.shape
        assert c == 1, c
        ch, cw = int(h / 2), int(w / 2)
        valid = (hmap[0, ch, cw].item() != np.inf)

        return valid

    def clone_invalid(self, hmap: torch.Tensor) -> torch.Tensor:
        assert hmap is not None
        assert isinstance(hmap, torch.Tensor), type(hmap)

        dtype = hmap.dtype
        device = hmap.device
        req_g = hmap.requires_grad
        hmap = torch.full(hmap.size(),
                          fill_value=torch.inf,
                          device=device,
                          requires_grad=req_g
                          )

        hmap = hmap.to(dtype)
        return hmap

    def op_heatmap(self, hmap: Union[torch.Tensor, None]
                   ) -> Union[torch.Tensor, None]:
        raise NotImplementedError


class RandomHorizontalFlip(_BasicTransform):
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):

        if self.p == 1 or random.random() < self.p:
            std_cam_ = std_cam
            if std_cam_ is not None:
                std_cam_ = TF.hflip(std_cam)

            lndmks_heatmap_ = self.op_heatmap(lndmks_heatmap)
            au_heatmap_ = self.op_heatmap(au_heatmap)
            d_aug_heatmap_ = self.op_heatmap(d_aug_heatmap)
            heatmap_seg_ = self.op_heatmap(heatmap_seg)
            bin_heatmap_seg_ = self.op_heatmap(bin_heatmap_seg)


            return TF.hflip(img), TF.hflip(unmasked_img), \
                   TF.hflip(raw_img), std_cam_, \
                   lndmks_heatmap_, au_heatmap_, d_aug_heatmap_, \
                   heatmap_seg_, bin_heatmap_seg_

        return img, unmasked_img, raw_img, std_cam, lndmks_heatmap, \
               au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg

    def op_heatmap(self, hmap: Union[torch.Tensor, None]
                   ) -> Union[torch.Tensor, None]:
        if hmap is None:
            return None

        valid = self._prep_op_hmap(hmap)

        if valid:
            hmap = TF.hflip(hmap)

        return hmap


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotation(_BasicTransform):
    def __init__(self,
                 degrees,
                 interpolation=InterpolationMode.BICUBIC,
                 expand=False,
                 center=None,
                 fill=0,
                 p=PROB_THRESHOLD
                 ):
        super(RandomRotation, self).__init__()
        self.p = p
        self.degrees = self.setup_angle(degrees, name="degrees", req_sizes=(2,))

        if center is not None:
            self.check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

        self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill
        assert self.fill == 0, self.fill  # fill different object types:
        # image, heatmaps, bin masks. better to be 0.

    @staticmethod
    def check_sequence_input(x, name, req_sizes):
        msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join(
            [str(s) for s in req_sizes])
        if not isinstance(x, Sequence):
            raise TypeError(f"{name} should be a sequence of length {msg}.")
        if len(x) not in req_sizes:
            raise ValueError(f"{name} should be a sequence of length {msg}.")

    def setup_angle(self, x, name, req_sizes=(2,)):
        if isinstance(x, numbers.Number):
            if x < 0:
                raise ValueError(
                    f"If {name} is a single number, it must be positive.")
            x = [-x, x]
        else:
            self.check_sequence_input(x, name, req_sizes)

        return [float(d) for d in x]

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random
            rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]),
                                              float(degrees[1])).item())
        return angle

    def get_angle_fill(self, img):
        """
        Get random angle.
        :param img: PIL Image or Tensor.
        :return:
        """
        fill = self.fill
        channels, _, _ = TF.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        return angle, fill

    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):

        if self.p == 1 or random.random() < self.p:
            angle, fill = self.get_angle_fill(img)

            std_cam_ = std_cam
            if std_cam_ is not None:

                std_cam_ = TF.rotate(std_cam_, angle, self.interpolation,
                                     self.expand, self.center, fill)

            lndmks_heatmap_ = self.op_heatmap(lndmks_heatmap, angle, fill)
            au_heatmap_ = self.op_heatmap(au_heatmap, angle, fill)
            d_aug_heatmap_ = self.op_heatmap(d_aug_heatmap, angle, fill)
            heatmap_seg_ = self.op_heatmap(heatmap_seg, angle, fill)
            bin_heatmap_seg_ = self.op_heatmap(bin_heatmap_seg, angle, fill)


            raw_img_ = raw_img
            img_ = img
            unmasked_img_ = unmasked_img
            raw_img_ = TF.rotate(raw_img_, angle, self.interpolation,
                                 self.expand, self.center, fill)
            img_ = TF.rotate(img_, angle, self.interpolation,
                             self.expand, self.center, fill)

            unmasked_img_ = TF.rotate(unmasked_img_, angle, self.interpolation,
                                      self.expand, self.center, fill)

            return img_, unmasked_img_, raw_img_, std_cam_, lndmks_heatmap_, \
                   au_heatmap_, d_aug_heatmap_, heatmap_seg_, bin_heatmap_seg_

        return img, unmasked_img, raw_img, std_cam, lndmks_heatmap, \
               au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg

    def op_heatmap(self,
                   hmap: Union[torch.Tensor, None],
                   angle,
                   fill,
                   ) -> Union[torch.Tensor, None]:
        if hmap is None:
            return None

        valid = self._prep_op_hmap(hmap)
        interpolation = InterpolationMode.NEAREST  # for tensor. nearest:
        # avoid creating new values.

        if valid:
            hmap = TF.rotate(hmap, angle, interpolation, self.expand,
                             self.center, fill)

        return hmap

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(degrees={self.degrees}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", expand={self.expand}"
        if self.center is not None:
            format_string += f", center={self.center}"
        if self.fill is not None:
            format_string += f", fill={self.fill}"
        format_string += f", p={self.p})"
        return format_string


class RandomImMaskViaHeatMap(_BasicTransform):
    """
    Perform data augmnetation over the image using heatmap: Mask the input
    image using a binary mask obtained from thresholding an attention map.
    The attention mape can be either: facial action units heatmap, or facial
    landmarks heatmap (exclusive).
    Can be run at inference time. but p has to be 1.
    """
    def __init__(self,
                 p: float = 0.5,
                 bg_filler: str = constants.BG_F_BLACK,
                 bg_filler_gauss_sigma: float = 30.,
                 roi_dilation: bool = False,
                 roi_dilation_radius: int = 1,
                 set_mode: str = constants.DS_EVAL,
                 avg_train_pixel: np.ndarray = None,
                 heatmap_type: str = constants.HEATMAP_AUNITS_LNMKS
                 ):
        super(RandomImMaskViaHeatMap, self).__init__()

        self._device = torch.device('cpu')

        assert set_mode in constants.SET_MODES, set_mode
        self.set_mode = set_mode

        assert isinstance(p, float), type(p)
        assert 0. <= p <= 1., p
        self.p = p  # the probability to use this transformation.

        assert bg_filler in constants.BG_FILLERS, bg_filler
        self.bg_filler = bg_filler

        assert isinstance(bg_filler_gauss_sigma, float), type(
            bg_filler_gauss_sigma)
        assert bg_filler_gauss_sigma > 0, bg_filler_gauss_sigma
        self.bg_filler_gauss_sigma = bg_filler_gauss_sigma

        assert isinstance(roi_dilation, bool), type(roi_dilation)
        self.roi_dilation = roi_dilation

        assert isinstance(roi_dilation_radius, int), type(roi_dilation_radius)
        assert roi_dilation_radius > 0, roi_dilation_radius
        self.roi_dilation_radius = roi_dilation_radius

        assert isinstance(avg_train_pixel, np.ndarray), type(avg_train_pixel)
        assert avg_train_pixel.ndim == 3, avg_train_pixel.ndim  # 1, 1, 3.
        assert avg_train_pixel.shape == (1, 1, 3), avg_train_pixel.shape

        self.avg_train_pixel = avg_train_pixel

        assert heatmap_type in constants.HEATMAP_TYPES, heatmap_type
        if heatmap_type == constants.HEATMAP_PER_CLASS_AUNITS_LNMKS:
            assert set_mode == constants.DS_EVAL, f"{set_mode} |{heatmap_type}"

        self.heatmap_type = heatmap_type

    def get_masked_img_black(self,
                             img: np.ndarray,
                             roi: np.ndarray) -> np.ndarray:
        assert img.ndim == 3, img.ndim
        assert roi.ndim == 2, roi.ndim

        _roi = np.expand_dims(roi.copy(), axis=2)

        return (img * _roi).astype(img.dtype)

    def get_masked_img_avg(self,
                           img: np.ndarray,
                           roi: np.ndarray) -> np.ndarray:
        assert img.ndim == 3, img.ndim
        assert roi.ndim == 2, roi.ndim

        _roi = np.expand_dims(roi.copy(), axis=2)

        # avg = img.mean(0).mean(0).reshape((1, 1, 3))
        # avg = avg.clip(0, 255).astype(img.dtype)
        avg = self.avg_train_pixel.astype(img.dtype)
        _im_avg = img * 0.0 + avg

        new_img = img * _roi + _im_avg * (1. - _roi)

        return new_img.astype(img.dtype)

    def get_masked_img_blur(self,
                            img: np.ndarray,
                            roi: np.ndarray,
                            sigma: float) -> np.ndarray:
        assert img.ndim == 3, img.ndim
        assert roi.ndim == 2, roi.ndim
        assert sigma > 0, sigma

        _roi = np.expand_dims(roi.copy(), axis=2)

        _blurred_img = gaussian(img, sigma=sigma, preserve_range=True,
                                channel_axis=-1)

        new_img = img * _roi + _blurred_img * (1. - _roi)

        return new_img.astype(img.dtype)

    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):

        if self.p == 1 or random.random() < self.p:
            heatmap = d_aug_heatmap

            if heatmap is None:

                assert self.heatmap_type == \
                       constants.HEATMAP_PER_CLASS_AUNITS_LNMKS, self.heatmap_type
                assert self.set_mode == constants.DS_EVAL, self.set_mode
                return img, unmasked_img, raw_img, std_cam, lndmks_heatmap, \
                       au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg

            assert heatmap is not None

            valid = self._prep_op_hmap(heatmap)

            if not valid:
                return img, unmasked_img, raw_img, std_cam, lndmks_heatmap, \
                       au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg

            # valid...
            np_heatmap = heatmap.squeeze(0)
            assert np_heatmap.ndim == 2, np_heatmap.ndim  # h, w
            np_heatmap = np_heatmap.numpy()
            h, w = np_heatmap.shape
            im_w, im_h = TF.get_image_size(img)
            assert w == im_w, f"{w} | {im_w}"
            assert h == im_h, f"{h} | {im_h}"

            th = threshold_otsu(np_heatmap, nbins=256)

            binary_roi: np.ndarray = (np_heatmap >= th).astype(np.float32)
            # h, w

            if self.roi_dilation:
                binary_roi = binary_dilation(binary_roi,
                                             footprint=disk(
                                                 radius=self.roi_dilation_radius,
                                                 dtype=np.float32,
                                                 strict_radius=True
                                                 )
                                                 ).astype(np.float32)


            assert isinstance(img, PIL.Image.Image), type(img)
            np_img = np.array(img)  # h, w, 3. uint8

            if self.bg_filler == constants.BG_F_BLACK:
                pertb_img = self.get_masked_img_black(np_img, binary_roi)

            elif self.bg_filler == constants.BG_F_IM_AVG:
                pertb_img = self.get_masked_img_avg(np_img, binary_roi)

            elif self.bg_filler == constants.BG_F_GAUSSIAN_BLUR:
                pertb_img = self.get_masked_img_blur(np_img, binary_roi,
                                                     self.bg_filler_gauss_sigma)

            else:
                raise NotImplementedError(self.bg_filler)

            pil_img = Image.fromarray(pertb_img, 'RGB')


            return pil_img, unmasked_img, raw_img, std_cam, lndmks_heatmap, \
                   au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg

        return img, unmasked_img, raw_img, std_cam, lndmks_heatmap, \
               au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg


class RandomCrop(_BasicTransform):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]
                   ) -> Tuple[int, int, int, int]:

        w, h = TF.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image "
                "size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self,
                 size,
                 padding=None,
                 pad_if_needed=False,
                 fill=0,
                 padding_mode="constant"
                 ):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two "
                            "dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

        assert self.fill == 0, self.fill  # fill different types of data. so
        # far, 0 is safer to not change the nature of data.

    def forward(self, img):
        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = TF.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        return img

    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):
        img_ = self.forward(img)
        raw_img_ = self.forward(raw_img)
        unmasked_img_ = self.forward(unmasked_img)
        assert img_.size == raw_img_.size
        assert img_.size == unmasked_img_.size

        i, j, h, w = self.get_params(img_, self.size)
        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = self.forward(std_cam)
            std_cam_ = TF.crop(std_cam_, i, j, h, w)

        lndmks_heatmap_ = self.op_heatmap(lndmks_heatmap, i, j, h, w)
        au_heatmap_ = self.op_heatmap(au_heatmap, i, j, h, w)
        d_aug_heatmap_ = self.op_heatmap(d_aug_heatmap, i, j, h, w)
        heatmap_seg_ = self.op_heatmap(heatmap_seg, i, j, h, w)
        bin_heatmap_seg_ = self.op_heatmap(bin_heatmap_seg, i, j, h, w)


        return TF.crop(img_, i, j, h, w), TF.crop(unmasked_img_, i, j, h, w), \
               TF.crop(raw_img_, i, j, h, w), std_cam_, lndmks_heatmap_, \
               au_heatmap_, d_aug_heatmap_, heatmap_seg_, bin_heatmap_seg_

    def op_heatmap(self,
                   hmap: Union[torch.Tensor, None],
                   i, j, h, w,
                   ) -> Union[torch.Tensor, None]:
        if hmap is None:
            return None

        valid = self._prep_op_hmap(hmap)

        hmap = self.forward(hmap)
        hmap = TF.crop(hmap, i, j, h, w)

        if not valid:
            hmap = self.clone_invalid(hmap)

        return hmap

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding)


class RandomResizedCrop(_BasicTransform):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
     dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge.
        If size is an
            int instead of sequence like (h, w), a square output size
            ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as
             (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a
                sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the
        random area of the crop,
            before resizing. The scale is defined with respect to the area of
            the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect
        ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is
             ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``,
            ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g.
            ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15.
            Please use InterpolationMode enum.
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3.0 / 4.0, 4.0 / 3.0),
                 interpolation=InterpolationMode.BILINEAR
                 ):
        super().__init__()

        self.size = _setup_size(size, error_msg="Please provide only two "
                                                "dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument 'interpolation' of type int is deprecated since"
                " 0.13 and will be removed in 0.15. "
                "Please use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float]
                   ) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio
            cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = TF.get_dimensions(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]
                                                         ).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(
                log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):

        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        img_ = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        raw_img_ = TF.resized_crop(raw_img, i, j, h, w, self.size,
                                   self.interpolation)
        unmasked_img_ = TF.resized_crop(unmasked_img, i, j, h, w, self.size,
                                        self.interpolation)
        assert img_.size == raw_img_.size
        assert img_.size == unmasked_img_.size

        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = TF.resized_crop(std_cam_, i, j, h, w, self.size,
                                       self.interpolation)

        lndmks_heatmap_ = self.op_heatmap(lndmks_heatmap, i, j, h, w,
                                          InterpolationMode.NEAREST)
        au_heatmap_ = self.op_heatmap(au_heatmap, i, j, h, w,
                                      InterpolationMode.NEAREST)
        d_aug_heatmap_ = self.op_heatmap(d_aug_heatmap, i, j, h, w,
                                         InterpolationMode.NEAREST)
        heatmap_seg_ = self.op_heatmap(heatmap_seg, i, j, h, w,
                                       InterpolationMode.NEAREST)
        bin_heatmap_seg_ = self.op_heatmap(bin_heatmap_seg, i, j, h, w,
                                           InterpolationMode.NEAREST)


        return img_, unmasked_img_, raw_img_, std_cam_,lndmks_heatmap_, \
               au_heatmap_, d_aug_heatmap_, heatmap_seg_, bin_heatmap_seg_

    def op_heatmap(self,
                   hmap: Union[torch.Tensor, None],
                   i, j, h, w,
                   interpolation: str
                   ) -> Union[torch.Tensor, None]:
        if hmap is None:
            return None

        valid = self._prep_op_hmap(hmap)

        hmap = TF.resized_crop(hmap, i, j, h, w, self.size, interpolation)

        if not valid:
            hmap = self.clone_invalid(hmap)

        return hmap

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string


class RandomGrayscale(_BasicTransform):
    """Randomly convert image to grayscale with a probability of p
    (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of
    leading dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with
        probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly grayscaled image.
        """

        num_output_channels, _, _ = TF.get_dimensions(img)

        if self.p == 1 or torch.rand(1) < self.p:
            num_output_channels, _, _ = TF.get_dimensions(img)

            img_ =  TF.rgb_to_grayscale(img,
                                       num_output_channels=num_output_channels)
            unmasked_img_  = TF.rgb_to_grayscale(
                unmasked_img, num_output_channels=num_output_channels)
            # raw image is expected to stay in its original state.

            # num_output_channels, _, _ = TF.get_dimensions(raw_img)
            # raw_img_ = TF.rgb_to_grayscale(raw_img,
            #                            num_output_channels=num_output_channels)
            raw_img_ = raw_img

            return img_, unmasked_img_, raw_img_, std_cam, lndmks_heatmap, \
                   au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg

        return img, unmasked_img, raw_img, std_cam, lndmks_heatmap, \
               au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomColorJitter(_BasicTransform):
    """Randomly ColorJitter.
    The same as transforms.ColorJitter but it is applied randomly over
    samples. instead over all samples.
    p: probability of aplying ColorJitter.
    """

    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0.,p=0.1):
        super().__init__()
        self.p = p
        self.colorjitter = ColorJitter(brightness=brightness,
                                       contrast=contrast,
                                       saturation=saturation,
                                       hue=hue
                                       )

    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Randomly color-jittered.
        """

        if self.p == 1 or torch.rand(1) < self.p:
            params = self.colorjitter.sample_params()
            fn_idx, brightness_factor, contrast_factor, saturation_factor, \
            hue_factor = params

            img_ = self.colorjitter(img, fn_idx, brightness_factor,
                                    contrast_factor, saturation_factor,
                                    hue_factor)
            unmasked_img_ = self.colorjitter(unmasked_img, fn_idx,
                                             brightness_factor,
                                             contrast_factor,
                                             saturation_factor,
                                             hue_factor)

            return img_, unmasked_img_, raw_img, std_cam, lndmks_heatmap, \
                   au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg

        return img, unmasked_img, raw_img, std_cam, lndmks_heatmap, \
               au_heatmap, d_aug_heatmap, heatmap_seg, bin_heatmap_seg

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class Resize(_BasicTransform):
    def __init__(self, size,
                 interpolation=TF.InterpolationMode.BILINEAR
                 ):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. "
                            "Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, "
                             "it should have 1 or 2 values")
        self.size = size
        self.interpolation = interpolation

    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):

        std_cam_ = std_cam
        if std_cam_ is not None:
            std_cam_ = TF.resize(std_cam_, self.size, self.interpolation,
                                 antialias=True)

        lndmks_heatmap_ = self.op_heatmap(lndmks_heatmap)
        au_heatmap_ = self.op_heatmap(au_heatmap)
        d_aug_heatmap_ = self.op_heatmap(d_aug_heatmap)
        heatmap_seg_ = self.op_heatmap(heatmap_seg)
        bin_heatmap_seg_ = self.op_heatmap(bin_heatmap_seg)


        return TF.resize(
            img, self.size, self.interpolation, antialias=True), \
               TF.resize(
                   unmasked_img, self.size, self.interpolation,
                   antialias=True), \
               TF.resize(
                   raw_img, self.size, self.interpolation, antialias=True), \
               std_cam_, lndmks_heatmap_, au_heatmap_, d_aug_heatmap_, \
               heatmap_seg_, bin_heatmap_seg_

    def op_heatmap(self, hmap: Union[torch.Tensor, None]
                   ) -> Union[torch.Tensor, None]:
        if hmap is None:
            return None

        valid = self._prep_op_hmap(hmap)
        interpolation = InterpolationMode.NEAREST  # dont change the range.

        hmap = TF.resize(hmap, self.size, interpolation)

        if not valid:
            hmap = self.clone_invalid(hmap)

        return hmap

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


class ToTensor(_BasicTransform):
    """
    The same as transforms.ToTensor().
    Except, we can control the scaling of the image on demand.
    *** OPERATES ONLY ON THE IMAGE IMG ***
    Helpfu if you want to keep the input image in its original scale.
    """
    def __init__(self, scale: bool = True):
        """
        Init.
        :param scale: bool. if true (default), we apply standard
        transofrms.ToTensor() where image values are scaled down into [0, 1
        ] from [0, 255]. if false, we do not perform this scaling.
        """
        assert isinstance(scale, bool), type(scale)
        self.scale_it = scale

        self.std_to_tensor = transforms.ToTensor()

    def __call__(self,
                 img,
                 unmasked_img,
                 raw_img,
                 std_cam,
                 lndmks_heatmap,
                 au_heatmap,
                 d_aug_heatmap,
                 heatmap_seg,
                 bin_heatmap_seg
                 ):

        lndmks_heatmap_ = lndmks_heatmap
        if lndmks_heatmap_ is not None:
            assert isinstance(lndmks_heatmap_, torch.Tensor), \
                type(lndmks_heatmap_)

        au_heatmap_ = au_heatmap
        if au_heatmap_ is not None:
            assert isinstance(au_heatmap_, torch.Tensor), \
                type(au_heatmap_)

        d_aug_heatmap_ = d_aug_heatmap
        if d_aug_heatmap_ is not None:
            assert isinstance(d_aug_heatmap_, torch.Tensor), \
                type(d_aug_heatmap_)

        heatmap_seg_ = heatmap_seg
        if heatmap_seg_ is not None:
            assert isinstance(heatmap_seg_, torch.Tensor), \
                type(heatmap_seg_)

        bin_heatmap_seg_ = bin_heatmap_seg
        if bin_heatmap_seg_ is not None:
            assert isinstance(bin_heatmap_seg_, torch.Tensor), \
                type(bin_heatmap_seg_)

        if self.scale_it:
            return self.std_to_tensor(img), self.std_to_tensor(unmasked_img),\
                   raw_img, std_cam, lndmks_heatmap_, au_heatmap_, \
                   d_aug_heatmap_, heatmap_seg_, bin_heatmap_seg_

        else:  # alternative is * 255 but may lose precision.
            return to_tensor(img), to_tensor(unmasked_img), raw_img, std_cam, \
                   lndmks_heatmap_, au_heatmap_, d_aug_heatmap_, \
                   heatmap_seg_, bin_heatmap_seg_


def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)

def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}

def to_tensor(pic) -> Tensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    if not (F_pil._is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError(f"pic should be 2/3 dimensional. "
                         f"Got {pic.ndim} dimensions.")

    default_float_dtype = torch.get_default_dtype()

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
        # backward compatibility
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype) #  .div(255)
        else:
            return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic).to(dtype=default_float_dtype)

    # handle PIL Image
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))

    if pic.mode == "1":
        img = 255 * img
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1)).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=default_float_dtype)  # .div(255)
    else:
        return img


def landmarks_to_heatmap(h: int,
                         w: int,
                         lndmks: list,
                         cov: np.ndarray,
                         normalize: bool
                         ) -> np.ndarray:
    """
    Generate a heatmap from landmarks.

    :param h: int. height of the heatmap.
    :param w: int. width of the heatmap.
    :param lndmks: list. [(x1, y1), ...] where x is the width. y is the height.
    :param cov: covariance matrix 2d for the Gaussian.
    :param normalize: bool. if true, the final heatmap is normalized.
    :return: heatmap (float32) shape (h, w).

    Note: some image samples have failed to yield landmarks. These cases are
    tagged with value inf for all landmarks. if inf is found in landmarks,
    a heatmap filled with inf is returned to indicate that this heatmap is
    invalid.
    """

    if lndmks[0][0] == np.inf:
        return np.ones((h, w), dtype=np.float32) * np.inf

    grid = np.dstack(np.mgrid[0:h:1, 0:w:1])
    # coordinates for multivariate_normal are (h, w).
    n = len(lndmks)
    heatmaps = np.zeros((n, h, w), dtype=np.float64)

    # todo: slow. to speedup.
    for i in range(n):
        z = lndmks[i]
        heatmaps[i] = multivariate_normal(mean=z[::-1], cov=cov).pdf(grid)

    heatmap = heatmaps.max(axis=0, initial=-1)  # h, w
    assert heatmap.shape == (h, w), f"{heatmap.shape} {(h, w)}"

    if normalize:
        _min = heatmap.min()
        _max = heatmap.max()
        _deno = _max - _min

        if _deno == 0:
            _deno = 1.
        heatmap = (heatmap - _min) / _deno

    return heatmap.astype(np.float32)



def get_image_ids_bucket(args, tr_bucket: int, split: str,
                         metadata_root: str) -> list:
    assert split == constants.TRAINSET
    chunks = list(range(constants.NBR_CHUNKS_TR[args.dataset]))
    buckets = list(chunk_it(chunks, constants.BUCKET_SZ))
    assert tr_bucket < len(buckets)

    _image_ids = []
    for i in buckets[tr_bucket]:
        metadata = {'image_ids': join(metadata_root, split,
                                      f'train_chunk_{i}.txt')}
        _image_ids.extend(get_image_ids(metadata, proxy=False))

    return _image_ids


class SamplerIterator:

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        for item in self.sampler:
            yield item

    def __len__(self):
        return len(self.sampler)


def sampler_to_distributed(sampler,
                           num_replicas: int = None,
                           rank: int = None,
                           shuffle: bool = False):

    return DistributedSampler(SamplerIterator(sampler),
                              num_replicas=num_replicas,
                              rank=rank,
                              shuffle=shuffle
                              )

def get_per_cl_number_samples(args) -> list:
    n_cls = args.num_classes

    path_stats = join(root_dir, args.metadata_root, 'per_class_weight.yaml')
    with open(path_stats, 'r') as f:
        stats = yaml.safe_load(f)
    assert isinstance(stats, dict), type(stats)
    ks = len(list(stats.keys()))
    assert ks == n_cls, f"{ks} {n_cls}"
    w = [1. for _ in range(n_cls)]
    for k in stats:
        w[k] = stats[k]

    return w


def get_sampler_cl_weights(args) -> list:
    n_cls = args.num_classes
    w_style = args.data_weighted_sampler_w
    msg = f"{w_style} {constants.DATA_SAMPLER_CLW}"
    assert w_style in constants.DATA_SAMPLER_CLW, msg

    w = get_per_cl_number_samples(args)

    # tech 1: CLWADAPTIVE unnecessary...
    if w_style in [constants.CLWFIXEDTECH1,
                               constants.CLWADAPTIVE]:

        w = cl_w_tech1(w=w, n_cls=n_cls)

    # tech 2:
    elif w_style == constants.CLWFIXEDTECH2:
        w = cl_w_tech2(w=w)

    else:
        raise NotImplementedError(w_style)


    return w.numpy().tolist()


class BalancedPerClassRandomSampler:
    def __init__(self, samples_labels: list, num_samples_per_cl: int):

        assert isinstance(samples_labels, list), type(samples_labels)
        assert len(samples_labels) > 0, len(samples_labels)

        assert isinstance(num_samples_per_cl, int), type(num_samples_per_cl)
        assert num_samples_per_cl > 0, num_samples_per_cl

        self.samples_labels = samples_labels
        self.num_samples_per_cl = num_samples_per_cl
        self.classes: list = np.unique(np.array(samples_labels)).tolist()
        self.n_cls: int = len(self.classes)
        self.num_samples: int = self.n_cls * num_samples_per_cl

        self.array_samples_labels = np.array(samples_labels)

    def _sample_round(self) -> list:
        samples = []

        for c in self.classes:
            samples_c = self._random_sample_one_class(cl=c,
                                                      n=self.num_samples_per_cl
                                                      )
            samples = samples + samples_c

        for i in range(100):
            random.shuffle(samples)

        return samples

    def _random_sample_one_class(self, cl: int, n: int) -> list:
        assert cl in self.classes, f"{cl} -- {self.classes}"
        assert isinstance(n, int), type(n)
        assert n > 0, n

        idx = np.where(self.array_samples_labels == cl)[0]
        # uniform sampling
        replace = idx.size < n

        s: np.ndarray = np.random.choice(idx, size=(n, ), replace=replace,
                                         p=None)

        s: list = s.tolist()

        assert len(s) == n, f"{len(s)} -- {n}"

        return s



    def __iter__(self) -> Iterator[int]:
        random_samples = self._sample_round()
        yield from iter(random_samples)


    def __len__(self) -> int:
        return self.num_samples


def get_eval_tranforms(args,
                       crop_size,
                       scale_img,
                       img_mean,
                       img_std,
                       avg_train_pixel):

    if args.eval_daug_mask_img_heatmap:
        p = 1.

        bg_filler = args.eval_daug_mask_img_heatmap_bg_filler
        bg_filler_gauss_sigma = args.eval_daug_mask_img_heatmap_gauss_sigma
        roi_dilation = args.eval_daug_mask_img_heatmap_dilation
        roi_dilation_radius = args.eval_daug_mask_img_heatmap_radius
        heatmap_type = args.eval_daug_mask_img_heatmap_type

        return Compose([
            Resize((crop_size, crop_size)),
            RandomImMaskViaHeatMap(p=p,
                                   bg_filler=bg_filler,
                                   bg_filler_gauss_sigma=bg_filler_gauss_sigma,
                                   roi_dilation=roi_dilation,
                                   roi_dilation_radius=roi_dilation_radius,
                                   set_mode=constants.DS_EVAL,
                                   avg_train_pixel=avg_train_pixel,
                                   heatmap_type=heatmap_type
                                   ),
            ToTensor(scale=scale_img),
            transforms.Normalize(img_mean, img_std)
        ])
    else:
        return Compose([
            Resize((crop_size, crop_size)),
            ToTensor(scale=scale_img),
            transforms.Normalize(img_mean, img_std)
        ])

def get_train_transforms(args,
                         resize_size,
                         crop_size,
                         scale_img,
                         img_mean,
                         img_std,
                         avg_train_pixel
                         ):

    if args.train_daug_mask_img_heatmap:

        p = args.train_daug_mask_img_heatmap_p
        bg_filler = args.train_daug_mask_img_heatmap_bg_filler
        bg_filler_gauss_sigma = args.train_daug_mask_img_heatmap_gauss_sigma
        roi_dilation = args.train_daug_mask_img_heatmap_dilation
        roi_dilation_radius = args.train_daug_mask_img_heatmap_radius
        heatmap_type = args.train_daug_mask_img_heatmap_type

        return Compose([
            Resize((resize_size, resize_size)),
            RandomImMaskViaHeatMap(p=p,
                                   bg_filler=bg_filler,
                                   bg_filler_gauss_sigma=bg_filler_gauss_sigma,
                                   roi_dilation=roi_dilation,
                                   roi_dilation_radius=roi_dilation_radius,
                                   set_mode=constants.DS_TRAIN,
                                   avg_train_pixel=avg_train_pixel,
                                   heatmap_type=heatmap_type
                                   ),
            RandomRotation(degrees=[-6, 6]),
            RandomResizedCrop(size=crop_size, scale=(0.8, 1.0),
                              ratio=(1. / 1., 1. / 1.)),
            RandomHorizontalFlip(),
            RandomGrayscale(p=0.2),
            RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                              hue=0.1, p=0.8),
            ToTensor(scale=scale_img),
            # todo: erasing may affect localization.
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33),
                                     ratio=(0.3, 3.3), value='random'),
            transforms.Normalize(img_mean, img_std)
        ])
    else:
        return Compose([
            Resize((resize_size, resize_size)),
            RandomRotation(degrees=[-6, 6]),
            RandomResizedCrop(size=crop_size, scale=(0.8, 1.0),
                              ratio=(1. / 1., 1. / 1.)),
            RandomHorizontalFlip(),
            RandomGrayscale(p=0.2),
            RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                              hue=0.1, p=0.8),
            ToTensor(scale=scale_img),
            # todo: erasing may affect localization.
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33),
                                     ratio=(0.3, 3.3), value='random'),
            transforms.Normalize(img_mean, img_std)
        ])

def get_data_loader(args,
                    data_roots,
                    metadata_root,
                    batch_size,
                    eval_batch_size,
                    workers,
                    resize_size,
                    crop_size,
                    proxy_training_set,
                    num_val_sample_per_class=0,
                    std_cams_folder=None,
                    get_splits_eval=None,
                    tr_bucket: Optional[int] = None,
                    curriculum_tr_ids: Optional[List] = None,
                    isdistributed=True
                    ):
    train_sampler = None

    scale_img = True
    img_mean = [0., 0., 0.]
    img_std = [1., 1., 1.]
    if args.model['encoder_weights'] == constants.VGGFACE2:
        scale_img = False
        img_mean = [131.0912, 103.8827, 91.4953]  # RGB.
        img_std = [1., 1., 1.]

    if args.model['encoder_weights'] == constants.IMAGENET:
        scale_img = True
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]

    avg_train_pixel = constants.AVG_IMG_PIXEL_TRAINSETS[args.dataset]
    assert isinstance(avg_train_pixel, list), type(avg_train_pixel)
    if args.dataset in [constants.RAFDB, constants.AFFECTNET]:
        assert len(avg_train_pixel) == 3, len(avg_train_pixel)
        avg_train_pixel = np.array(avg_train_pixel).reshape((1, 1, 3))

    if isinstance(get_splits_eval, list):
        assert len(get_splits_eval) > 0
        eval_datasets = {
            split: WSOLImageLabelDataset(
                    args=args,
                    data_root=data_roots[split],
                    metadata_root=join(metadata_root, split),
                    transform=get_eval_tranforms(args,
                                                 crop_size,
                                                 scale_img,
                                                 img_mean,
                                                 img_std,
                                                 avg_train_pixel
                                                 ),
                    proxy=False,
                    resize_size=resize_size,
                    crop_size=crop_size,
                    set_mode=constants.DS_EVAL,
                    split=split,
                    num_sample_per_class=0,
                    root_data_cams=''
                )
            for split in get_splits_eval
        }
        loaders = {
            split: DataLoader(
                eval_datasets[split],
                batch_size=eval_batch_size,
                shuffle=False,
                sampler=DistributedSampler(dataset=eval_datasets[split],
                                           shuffle=False) if isdistributed
                else None,
                num_workers=workers)
            for split in get_splits_eval
        }
        return loaders, train_sampler

    dataset_transforms = dict(
        train=get_train_transforms(args,
                                   resize_size,
                                   crop_size,
                                   scale_img,
                                   img_mean,
                                   img_std,
                                   avg_train_pixel
                                   ),
        val=get_eval_tranforms(args,
                               crop_size,
                               scale_img,
                               img_mean,
                               img_std,
                               avg_train_pixel
                               ),
        test=get_eval_tranforms(args,
                                crop_size,
                                scale_img,
                                img_mean,
                                img_std,
                                avg_train_pixel
                                )
    )

    image_ids = {
        split: None for split in _SPLITS
    }

    if not args.ds_chunkable:
        assert tr_bucket in [0, None]

        if curriculum_tr_ids is not None:
            image_ids[constants.TRAINSET] = curriculum_tr_ids

    elif tr_bucket is not None:
        assert args.dataset == constants.ILSVRC
        image_ids[constants.TRAINSET] = get_image_ids_bucket(
            args=args, tr_bucket=tr_bucket, split=constants.TRAINSET,
            metadata_root=metadata_root)

    def get_mode(split: str) -> str:
        if split in [constants.VALIDSET, constants.TESTSET]:
            return constants.DS_EVAL

        elif split == constants.TRAINSET:
            return constants.DS_TRAIN

        else:
            raise NotImplementedError(split)

    datasets = {
        split: WSOLImageLabelDataset(
                args=args,
                data_root=data_roots[split],
                metadata_root=join(metadata_root, split),
                transform=dataset_transforms[split],
                proxy=proxy_training_set and split == constants.TRAINSET,
                resize_size=resize_size,
                crop_size=crop_size,
                set_mode=get_mode(split),
                split=split,
                num_sample_per_class=(num_val_sample_per_class
                                      if split == constants.VALIDSET else 0),
                root_data_cams=std_cams_folder[split],
                image_ids=image_ids[split]
            )
        for split in _SPLITS
    }

    train_sampler = None

    if constants.TRAINSET in _SPLITS:
        train_sampler = DistributedSampler(dataset=datasets[constants.TRAINSET],
                                           shuffle=True
                                           )

        if args.data_weighted_sampler:

            _tr_ds = datasets[constants.TRAINSET]
            n = len(_tr_ds)
            labels = [_tr_ds.image_labels[_xid] for _xid in _tr_ds.image_ids]

            assert args.data_weighted_sampler_per_cl in \
                   constants.DATA_SAMPLER_PER_CL, args.data_weighted_sampler_per_cl

            if args.data_weighted_sampler_per_cl == constants.PER_CL_NONE:
                w = get_sampler_cl_weights(args)
                samples_w = [w[l] for l in labels]
                balanced_sampler = WeightedRandomSampler(weights=samples_w,
                                                         num_samples=n,
                                                         replacement=True
                                                         )

            elif args.data_weighted_sampler_per_cl == constants.PER_CL_MIN_CL:

                n_per_cl: list = get_per_cl_number_samples(args)
                num_samples_per_cl = int(min(n_per_cl))

                balanced_sampler = BalancedPerClassRandomSampler(
                    samples_labels=labels,
                    num_samples_per_cl=num_samples_per_cl
                )

            else:
                raise NotImplementedError(args.data_weighted_sampler_per_cl)

            train_sampler = sampler_to_distributed(sampler=balanced_sampler,
                                                   shuffle=True
                                                   )


    samplers = {
        split: DistributedSampler(dataset=datasets[split],
                                  shuffle=False
                                  ) if split != constants.TRAINSET else train_sampler
        for split in _SPLITS
    }

    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size if split == constants.TRAINSET else eval_batch_size,
            shuffle=False,
            sampler=samplers[split],
            num_workers=workers
        )
        for split in _SPLITS
    }

    if constants.TRAINSET in _SPLITS:
        train_sampler = samplers[constants.TRAINSET]

    return loaders, train_sampler


def fast_draw_landmarks(img: np.ndarray,
                        heatmap: np.ndarray,
                        x_h: list,
                        y_w: list,
                        wfp,
                        binary_roi: np.ndarray = None,
                        img_msk_black: np.ndarray = None,
                        img_msk_avg: np.ndarray = None,
                        img_msk_blur: np.ndarray = None
                        ):

    alpha = 1.
    markersize = 4.5
    lw = 1.
    color = 'r'
    markeredgecolor = 'red'

    height, width = img.shape[:2]
    # just for plotting
    # avoid plotting right over thr border. dots spill outside the image.
    x_h = [max(min(x, height - 1), 0) for x in x_h]
    y_w = [max(min(y, width - 1), 0) for y in y_w]

    ncols = 3

    if binary_roi is not None:
        ncols += 1

    if img_msk_black is not None:
        ncols += 1

    if img_msk_avg is not None:
        ncols += 1

    if img_msk_blur is not None:
        ncols += 1

    fig, axes = plt.subplots(nrows=1, ncols=ncols, squeeze=False)

    fontsize = 7 if ncols == 2 else 3

    col = 0
    axes[0, col].imshow(img[:, :, ::-1])

    nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

    # close eyes and mouths
    plot_close = lambda i1, i2: axes[0, col].plot([x_h[i1], x_h[i2]],
                                                  [y_w[i1], y_w[i2]],
                                                  color=color,
                                                  lw=lw,
                                                  alpha=alpha - 0.1
                                                  )
    plot_close(41, 36)
    plot_close(47, 42)
    plot_close(59, 48)
    plot_close(67, 60)

    for ind in range(len(nums) - 1):
        l, r = nums[ind], nums[ind + 1]
        axes[0, col].plot(x_h[l:r], y_w[l:r], color=color, lw=lw,
                          alpha=alpha - 0.1)

        axes[0, col].plot(x_h[l:r], y_w[l:r], marker='o', linestyle='None',
                          markersize=markersize, color=color,
                          markeredgecolor=markeredgecolor, alpha=alpha)

    col += 1
    axes[0, col].imshow(img[:, :, ::-1])
    axes[0, col].imshow(heatmap, alpha=0.7)

    col += 1

    axes[0, col].imshow(heatmap, alpha=1.0)

    col += 1

    if binary_roi is not None:
        axes[0, col].imshow(binary_roi.astype(np.uint8) * 255, cmap='gray')
        axes[0, col].text(
            3, 40, 'ROI',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        col += 1

    if img_msk_black is not None:
        axes[0, col].imshow(img_msk_black[:, :, ::-1])
        axes[0, col].text(
            3, 40, 'Black masking',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        col += 1

    if img_msk_avg is not None:
        axes[0, col].imshow(img_msk_avg[:, :, ::-1])
        axes[0, col].text(
            3, 40, 'Average masking',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        col += 1

    if img_msk_black is not None:
        axes[0, col].imshow(img_msk_blur[:, :, ::-1])
        axes[0, col].text(
            3, 40, 'Gaussian blur masking',
            fontsize=fontsize,
            bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                  'edgecolor':'none'}
        )
        col += 1


    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())

    # plt.show()

    plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
    plt.close()

def interpolate_via_pytorch(x: np.ndarray,
                            new_h: int,
                            new_w: int
                            ) -> np.ndarray:
    a = torch.from_numpy(x)
    assert a.ndim == 2, a.ndim
    a = a.unsqueeze(0).unsqueeze(0)  # 1, 1, h, w.
    new_a = F.interpolate(input=a,
                          size=(new_h, new_w),
                          mode='bicubic'
                          )
    new_a = new_a.squeeze()
    return new_a.numpy().astype(x.dtype)


def test_heatmap_from_lndmks(ds: str, split: str, show_jaw: bool):
    from dlib.utils.tools import get_root_wsol_dataset

    announce_msg(f"Start building test ehatmaps of landmarks. "
                 f"Dataset: {ds}. Split: {split}. Show jaw: {show_jaw}")

    metadata_root = join(root_dir, f"folds/wsol-done-right-splits/{ds}/{split}")
    metadata = configure_metadata(metadata_root)
    landmarks = get_landmarks(metadata)
    n = len(list(landmarks.keys()))

    baseurl = get_root_wsol_dataset()
    variance = 64.
    fdout = join(root_dir,
                 f'data/debug/out/heatmap-lndmks/{ds}-var-{variance}-jaw-'
                 f'{show_jaw}')
    os.makedirs(fdout, exist_ok=True)

    h, w = 256, 256
    cov = np.zeros((2, 2), dtype=np.float32)
    np.fill_diagonal(cov, variance)
    normalize = True
    scale = 64.
    show_local = False

    jj = 0

    for im_id in tqdm.tqdm(landmarks, total=n, ncols=80):

        if not show_local:
            path = join(baseurl, ds, im_id)
            assert os.path.isfile(path)

            img = cv2.imread(path)

        t0 = time.perf_counter()
        lndmks = landmarks[im_id]
        if not show_jaw:
            lndmks = lndmks[17:]

        _h, _w = img.shape[:2]
        h = _h
        w = _w

        heatmap = landmarks_to_heatmap(h=h,
                                       w=w,
                                       lndmks=lndmks,
                                       cov=cov,
                                       normalize=normalize
                                       )
        t1 = time.perf_counter()
        # print(f"Time: {t1 - t0} (s)")
        if 'train_11811' in im_id:
            print(im_id, '***********************',
                  (np.isnan(heatmap).sum() > 0))
            # input('OK')


        if np.isinf(heatmap).sum() == 0:
            x, y = zip(*landmarks[im_id])

            isnan = (np.isnan(heatmap).sum() > 0)
            if isnan:
                print(im_id, f' has NAN: {np.isnan(heatmap).sum()}')
                input('OK?')

            # ------------------------------------------------------------------
            binary_aus_roi = None

            _pertub_heatmap = heatmap.copy()

            _imh, _imw = img.shape[0:2]
            zz = Image.fromarray(_pertub_heatmap).resize((_imw, _imh),
                                                         resample=NEAREST)
            _pertub_heatmap = np.array(zz)

            otsu_thresh = threshold_otsu(_pertub_heatmap, nbins=256)
            binary_aus_roi = (_pertub_heatmap >= otsu_thresh).astype(np.float32)

            radius_dilate = 0

            if radius_dilate > 0:
                binary_aus_roi = binary_dilation(binary_aus_roi,
                                                 footprint=disk(
                                                     radius=radius_dilate,
                                                     dtype=np.float32,
                                                     strict_radius=True
                                                 )
                                                 ).astype(np.float32)

            img_msk_black = get_masked_img_black(img, binary_aus_roi)
            img_msk_avg = get_masked_img_avg(img, binary_aus_roi)
            img_mask_blur = get_masked_img_blur(img, binary_aus_roi,
                                                sigma=30)

            # ------------------------------------------------------------------

            if not show_local:
                fast_draw_landmarks(
                    img, _pertub_heatmap, x, y,
                    wfp=join(fdout, f'{reformat_id(im_id)}.jpg'),
                    binary_roi=binary_aus_roi,
                    img_msk_black=img_msk_black,
                    img_msk_avg=img_msk_avg,
                    img_msk_blur=img_mask_blur
                )

            # Show
            if show_local:
                plt.imshow(heatmap)
                plt.show()
                interp = interpolate_via_pytorch(heatmap, int(h / scale),
                                                 int(w / scale))
                plt.imshow(interp)
                plt.show()
                break

        jj += 1

        if jj >= 5:
            pass

    announce_msg(f"End building test ehatmaps of landmarks. "
                 f"Dataset: {ds}. Split: {split}. Show jaw: {show_jaw}")


def get_masked_img_black(img: np.ndarray, roi: np.ndarray) -> np.ndarray:
    assert img.ndim == 3, img.ndim
    assert roi.ndim == 2, roi.ndim

    _roi = np.expand_dims(roi.copy(), axis=2)

    return (img * _roi).astype(img.dtype)


def get_masked_img_avg(img: np.ndarray, roi: np.ndarray) -> np.ndarray:
    assert img.ndim == 3, img.ndim
    assert roi.ndim == 2, roi.ndim

    _roi = np.expand_dims(roi.copy(), axis=2)

    avg = img.mean(0).mean(0).reshape((1, 1, 3))
    avg = avg.clip(0, 255).astype(img.dtype)
    _im_avg = img * 0.0 + avg

    new_img = img * _roi + _im_avg * (1. - _roi)

    return new_img.astype(img.dtype)


def get_masked_img_blur(img: np.ndarray, roi: np.ndarray,
                        sigma: float) -> np.ndarray:
    assert img.ndim == 3, img.ndim
    assert roi.ndim == 2, roi.ndim
    assert sigma > 0, sigma

    _roi = np.expand_dims(roi.copy(), axis=2)

    _blurred_img = gaussian(img, sigma=sigma, preserve_range=True,
                            channel_axis=-1)

    new_img = img * _roi + _blurred_img * (1. - _roi)

    return new_img.astype(img.dtype)


def find_nan_cause():
    resize_size = 256
    crop_size = 224
    scale_img = True
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    a  = torch.zeros(1, 256, 256) + torch.inf
    im = np.random.rand(resize_size, resize_size, 3) * 255
    im = im.astype(np.uint8)
    im = Image.fromarray(im)

    tr = Compose([
            Resize((resize_size, resize_size)),
            RandomRotation(degrees=[-6, 6]),
            RandomResizedCrop(size=crop_size, scale=(0.8, 1.0),
                              ratio=(1. / 1., 1. / 1.)),
            RandomHorizontalFlip(),
            RandomGrayscale(p=0.2),
            RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                              hue=0.1, p=0.8),
            ToTensor(scale=scale_img),
            # todo: erasing may affect localization.
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33),
                                     ratio=(0.3, 3.3), value='random'),
            transforms.Normalize(img_mean, img_std)
        ])
    img_, raw_im, cam, lnd, au = tr(im, im, None, a, a)
    print(lnd)
    print(au)


def store_heatmaps_of_lndmks(ds: str, split: str, show_jaw: bool, clean: bool):
    from dlib.utils.tools import get_root_wsol_dataset
    from dlib.utils.tools import get_heatmap_tag
    from dlib.configure.config import get_config
    from dlib.utils.tools import Dict2Obj
    from dlib.utils.shared import reformat_id

    announce_msg(f"Start storing heatmaps from landmarks. "
                 f"DS: {ds}, Split: {split}. Show jaw: {show_jaw}...")

    metadata_root = join(root_dir, f"folds/wsol-done-right-splits/{ds}/{split}")
    metadata = configure_metadata(metadata_root)
    landmarks = get_landmarks(metadata)
    n = len(list(landmarks.keys()))

    # config
    variance = 64.
    normalize = True
    args = get_config(ds)
    args = Dict2Obj(args)
    args.align_atten_to_heatmap_type_heatmap = constants.HEATMAP_LNDMKS
    args.align_atten_to_heatmap_normalize = normalize
    args.align_atten_to_heatmap_lndmk_variance = variance
    args.align_atten_to_heatmap_jaw = show_jaw
    tag = get_heatmap_tag(args, key=constants.ALIGN_ATTEN_HEATMAP)

    baseurl = get_root_wsol_dataset()
    outdir = join(baseurl, tag)
    if os.path.isdir(outdir) and clean:
        print(f" deleting {outdir}")
        os.system(f"rm -r {outdir}")

    os.makedirs(outdir, exist_ok=True)
    print(f"Destination: {outdir}")

    h, w = constants.SZ224, constants.SZ224
    cov = np.zeros((2, 2), dtype=np.float32)
    np.fill_diagonal(cov, variance)

    for im_id in tqdm.tqdm(landmarks, total=n, ncols=80):
        lndmks = landmarks[im_id]

        if not show_jaw:
            lndmks = lndmks[17:]

        path_img = join(baseurl, ds, im_id)
        assert os.path.isfile(path_img)

        img = cv2.imread(path_img)

        _h, _w = img.shape[:2]
        h = _h
        w = _w

        heatmap = landmarks_to_heatmap(h=h, w=w, lndmks=lndmks,
                                       cov=cov, normalize=normalize)
        path_out = join(outdir, f"{reformat_id(im_id)}.npy")

        # allow to overwrite.
        # assert not os.path.isfile(path_out), path_out

        np.save(path_out, heatmap, allow_pickle=False, fix_imports=True)

        # t0 = time.perf_counter()
        # z = np.load(path_out,allow_pickle=False, fix_imports=True)
        # t1 = time.perf_counter()
        # print(f"Loading time: {t1 - t0} (s)")
        # print((heatmap - z).sum())

    announce_msg(f"End storing heatmaps from landmarks. "
                 f"DS: {ds}, Split: {split}. Show jaw: {show_jaw}...")


def build_heatmaps_from_landmarks(ds: str):

    # store heatmaps of landmarks.

    for show_jaw in [True, False]:
        i = 0
        for split in [constants.TRAINSET,
                      constants.VALIDSET,
                      constants.TESTSET
                      ]:
            clean = (i == 0)
            print(f"Split: {split} / Jaw: {show_jaw}:")
            store_heatmaps_of_lndmks(ds=ds, split=split, show_jaw=show_jaw,
                                     clean=clean)

            i += 1

    # LANDMARKS ----------------------------------------------------------------


if __name__ == "__main__":
    _ACTION_VIS = 'visualize'
    _ACTION_STORE = 'store'
    _ACTIONS = [_ACTION_STORE, _ACTION_VIS]

    import argparse

    from dlib.utils.shared import str2bool
    from dlib.utils.shared import announce_msg

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default=_ACTION_VIS,
                        required=True, help="Action: visualize or store "
                                            "landmarks heatmaps.")
    parser.add_argument("--dataset", type=str,
                        default=constants.RAFDB,
                        required=True,
                        help="Dataset name: raf-db, affectnet.")

    args = parser.parse_args()
    ds_name = args.dataset
    action = args.action
    assert ds_name in [constants.RAFDB, constants.AFFECTNET], ds_name
    assert action in _ACTIONS, action

    ds = ds_name

    # find_nan_cause()

    # LANDMARKS ------------------------------------------------------------
    if action == _ACTION_VIS:
        # 1. visualize
        print(f"Processing dataset: {action} ): {ds_name}")
        for show_jaw in [True, False]:
            for split in [constants.TESTSET,
                          constants.VALIDSET,
                          constants.TRAINSET
                          ]:

                test_heatmap_from_lndmks(ds=ds, split=split, show_jaw=show_jaw)

    elif action == _ACTION_STORE:
        # 2. store maps.
        print(f"Processing dataset: {action}): {ds_name}")
        build_heatmaps_from_landmarks(ds=ds)

    else:
        raise NotImplementedError(action)


