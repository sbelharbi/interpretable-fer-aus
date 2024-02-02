import time
from pathlib import Path
import subprocess

import kornia.morphology
import numpy as np
import os
import sys
from os.path import dirname, abspath, join
import datetime as dt
import pickle as pkl
from typing import Tuple, Union


import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torch.cuda.amp import autocast
import torch.distributed as dist

from tqdm import tqdm as tqdm

from skimage.filters import threshold_otsu
from skimage import filters

from PIL import Image

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.metrics.wsol_metrics import BoxEvaluator
from dlib.metrics.wsol_metrics import MaskEvaluator
from dlib.metrics.wsol_metrics import compute_bboxes_from_scoremaps
from dlib.metrics.wsol_metrics import calculate_multiple_iou
from dlib.metrics.wsol_metrics import get_mask
from dlib.metrics.wsol_metrics import load_mask_image
from dlib.metrics.wsol_metrics import compute_cosine_one_sample

from dlib.datasets.wsol_loader import configure_metadata
from dlib.visualization.vision_wsol import Viz_WSOL

from dlib.utils.tools import t2n
from dlib.utils.tools import check_scoremap_validity
from dlib.configure import constants
from dlib.cams import build_std_cam_extractor
from dlib.cams import build_fcam_extractor
from dlib.utils.shared import reformat_id
from dlib.utils.shared import gpu_memory_stats

from dlib.parallel import sync_tensor_across_gpus

from dlib.utils.reproducibility import set_seed
import dlib.dllogger as DLLogger


_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = constants.SZ224  # 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float).
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()

    return cam


def max_normalize(cam):
    max_val = cam.max()
    if max_val == 0.:
        return cam

    return cam / max_val


def entropy_cam(cam: torch.Tensor) -> torch.Tensor:
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == 2

    ops = 1. - cam
    entrop = - cam * torch.log2(cam) - ops * torch.log2(ops)
    assert ((entrop > 1.) + (entrop < 0.)).sum() == 0.

    return entrop


class CAMComputer(object):
    def __init__(self,
                 args,
                 model,
                 loader: DataLoader,
                 metadata_root,
                 mask_root,
                 iou_threshold_list,
                 dataset_name,
                 split,
                 multi_contour_eval,
                 cam_curve_interval: float = .001,
                 out_folder=None,
                 fcam_argmax: bool = False):
        self.args = args
        self.model = model
        self.model.eval()
        self.loader = loader
        self.dataset_name = dataset_name
        self.split = split
        self.out_folder = out_folder
        self.fcam_argmax = fcam_argmax

        if args.task == constants.F_CL:
            self.req_grad = False
        elif args.task == constants.STD_CL:
            self.req_grad = constants.METHOD_REQU_GRAD[args.method]
        else:
            raise NotImplementedError

        metadata = configure_metadata(metadata_root)
        cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {constants.OpenImages: MaskEvaluator,
                          constants.CUB: BoxEvaluator,
                          constants.ILSVRC: BoxEvaluator,
                          constants.RAFDB: BoxEvaluator,
                          constants.AFFECTNET: BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval,
                                          args=args)

        if dataset_name in [constants.RAFDB, constants.AFFECTNET,
                            constants.CUB, constants.ILSVRC]:
            self.bbox = True
        elif dataset_name in [constants.OpenImages]:
            self.bbox = False
        else:
            raise NotImplementedError

        self.viz = Viz_WSOL()
        self.default_seed = int(os.environ["MYSEED"])

        self.std_cam_extractor = None
        self.fcam_extractor = None

        if args.task == constants.STD_CL:
            self.std_cam_extractor = self._build_std_cam_extractor(
                classifier=self.model, args=self.args)
        elif args.task == constants.F_CL:
            self.fcam_extractor = self._build_fcam_extractor(
                model=self.model, args=self.args)
            # useful for drawing side-by-side.
            # todo: build classifier from scratch and create its cam extractor.
        else:
            raise NotImplementedError

        self.special1 = self.args.method in [constants.METHOD_TSCAM]

        folds_path = join(root_dir, self.args.metadata_root)
        path_class_id = join(folds_path, 'class_id.yaml')
        with open(path_class_id, 'r') as fcl:
            cl_int = yaml.safe_load(fcl)

        self.cl_to_int: dict = cl_int
        self.int_to_cl: dict = self.switch_key_val_dict(cl_int)
        self.avg_per_cl_cams = None
        self.avg_per_cl_aus_maps = None  # g. truth
        self.avg_per_cl_att_maps = None  # attention map at the requested
        # layer for AU alignment. if many layers, we consider the last one only.
        self.init_avg_per_cl_cams()
        self.init_avg_per_cl_aus_maps()
        self.init_avg_per_cl_att_maps()

        self.avg_per_cl_au_cosine = None
        self.init_avg_per_cl_au_cosine()
        self.is_avail_avg_per_cl_au_cosine = False

    def get_matrix_avg_per_cl_au_cosine(self) -> Union[np.ndarray, None]:

        if not self.is_avail_avg_per_cl_au_cosine:
            return None

        # ordered classes from low (0) to high.
        ord_int_cls = sorted(list(self.int_to_cl.keys()), reverse=False)
        ord_str_cls = [self.int_to_cl[x] for x in ord_int_cls]
        cosines = [self.avg_per_cl_au_cosine[k][constants.AU_COSINE_MTR] for
                   k in ord_str_cls
                  ]

        default = None
        for i, z in enumerate(cosines):
            if z is not None:
                default = z * 0. - 1  # for neutral, ...
                break

        for i, z in enumerate(cosines):
            if z is None:
                cosines[i] = default

        cosines = [x.contiguous().view(1, -1) for x in cosines]

        tens_cos = torch.cat(cosines, dim=0)  # n_classes, n_layer + 1.

        out = tens_cos.detach().cpu().numpy()

        return out

    def init_avg_per_cl_cams(self):
        self.avg_per_cl_cams = dict()

        for k in self.cl_to_int:
            self.avg_per_cl_cams[k] = {'cam': None,
                                       'nbr': 0.
                                       }

    def init_avg_per_cl_att_maps(self):
        self.avg_per_cl_att_maps = dict()

        for k in self.cl_to_int:
            self.avg_per_cl_att_maps[k] = {'cam': None,
                                           'nbr': 0.
                                           }

    def init_avg_per_cl_aus_maps(self):
        self.avg_per_cl_aus_maps = dict()

        for k in self.cl_to_int:
            self.avg_per_cl_aus_maps[k] = {'cam': None,
                                           'nbr': 0.
                                          }

    def init_avg_per_cl_au_cosine(self):
        self.avg_per_cl_au_cosine = dict()

        for k in self.cl_to_int:
            self.avg_per_cl_au_cosine[k] = {constants.AU_COSINE_MTR: None,
                                            'nbr': 0.
                                           }

    def avg_cams(self):
        for k in self.cl_to_int:
            nbr = float(self.avg_per_cl_cams[k]['nbr'])

            if nbr > 0:
                _v = self.avg_per_cl_cams[k]['cam']
                self.avg_per_cl_cams[k]['cam'] = _v / nbr

    def avg_aus_maps(self):
        for k in self.cl_to_int:
            nbr = float(self.avg_per_cl_aus_maps[k]['nbr'])

            if nbr > 0:
                _v = self.avg_per_cl_aus_maps[k]['cam']
                self.avg_per_cl_aus_maps[k]['cam'] = _v / nbr


    def avg_att_maps(self):
        for k in self.cl_to_int:
            nbr = float(self.avg_per_cl_att_maps[k]['nbr'])

            if nbr > 0:
                _v = self.avg_per_cl_att_maps[k]['cam']
                self.avg_per_cl_att_maps[k]['cam'] = _v / nbr

    def synch_cams(self):
        null_cam = None
        for k in self.cl_to_int:
            cam = self.avg_per_cl_cams[k]['cam']
            if cam is not None:
                null_cam = cam * 0.0
                break

        assert null_cam is not None, 'Didnt find a valid cam.'

        for k in self.cl_to_int:
            nbr = float(self.avg_per_cl_cams[k]['nbr'])

            nx = torch.tensor([nbr], dtype=torch.float,
                              requires_grad=False,
                              device=torch.device(self.args.c_cudaid)).view(1, )
            nbr = sync_tensor_across_gpus(nx).sum().item()

            cam = self.avg_per_cl_cams[k]['cam']
            if cam is None:
                cam = null_cam

            cam = cam.unsqueeze(0)  # 1, h, w
            cam = sync_tensor_across_gpus(cam).sum(dim=0).squeeze()  # h, w
            assert cam.ndim == 2, cam.ndim

            self.avg_per_cl_cams[k]['nbr'] = nbr
            self.avg_per_cl_cams[k]['cam'] = cam

    def synch_aus_maps(self):
        null_map = None
        for k in self.cl_to_int:
            map = self.avg_per_cl_aus_maps[k]['cam']
            if map is not None:
                null_map = map * 0.0
                break

        assert null_map is not None, 'Didnt find a valid map.'

        for k in self.cl_to_int:
            nbr = float(self.avg_per_cl_aus_maps[k]['nbr'])

            nx = torch.tensor([nbr], dtype=torch.float,
                              requires_grad=False,
                              device=torch.device(self.args.c_cudaid)).view(1, )
            nbr = sync_tensor_across_gpus(nx).sum().item()

            map = self.avg_per_cl_aus_maps[k]['cam']
            if map is None:
                map = null_map

            map = map.unsqueeze(0)  # 1, h, w
            map = sync_tensor_across_gpus(map).sum(dim=0).squeeze()  # h, w
            assert map.ndim == 2, map.ndim

            self.avg_per_cl_aus_maps[k]['nbr'] = nbr
            self.avg_per_cl_aus_maps[k]['cam'] = map

    def synch_att_maps(self):
        null_map = None
        for k in self.cl_to_int:
            map = self.avg_per_cl_att_maps[k]['cam']
            if map is not None:
                null_map = map * 0.0
                break

        assert null_map is not None, 'Didnt find a valid map.'

        for k in self.cl_to_int:
            nbr = float(self.avg_per_cl_att_maps[k]['nbr'])

            nx = torch.tensor([nbr], dtype=torch.float,
                              requires_grad=False,
                              device=torch.device(self.args.c_cudaid)).view(1, )
            nbr = sync_tensor_across_gpus(nx).sum().item()

            map = self.avg_per_cl_att_maps[k]['cam']
            if map is None:
                map = null_map

            map = map.unsqueeze(0)  # 1, h, w
            map = sync_tensor_across_gpus(map).sum(dim=0).squeeze()  # h, w
            assert map.ndim == 2, map.ndim

            self.avg_per_cl_att_maps[k]['nbr'] = nbr
            self.avg_per_cl_att_maps[k]['cam'] = map

    def synch_au_cosine(self):

        assert self.is_avail_avg_per_cl_au_cosine

        null_cos = None
        for k in self.cl_to_int:
            cos = self.avg_per_cl_au_cosine[k][constants.AU_COSINE_MTR]
            if cos is not None:
                null_cos = cos * 0.0
                break

        assert null_cos is not None, 'Didnt find a valid cosine.'

        for k in self.cl_to_int:
            nbr = float(self.avg_per_cl_au_cosine[k]['nbr'])

            nx = torch.tensor([nbr], dtype=torch.float,
                              requires_grad=False,
                              device=torch.device(self.args.c_cudaid)).view(1, )
            nbr = sync_tensor_across_gpus(nx).sum().item()

            cos = self.avg_per_cl_au_cosine[k][constants.AU_COSINE_MTR]
            if cos is None:
                cos = null_cos

            cos = cos.contiguous().view(1, -1)  # 1, n_ly + 1
            cos = sync_tensor_across_gpus(cos).sum(dim=0).squeeze()  # m_ly + 1

            self.avg_per_cl_au_cosine[k]['nbr'] = nbr
            self.avg_per_cl_au_cosine[k][constants.AU_COSINE_MTR] = cos


    def avg_au_cosine(self):

        assert self.is_avail_avg_per_cl_au_cosine

        for k in self.cl_to_int:
            nbr = float(self.avg_per_cl_au_cosine[k]['nbr'])

            if nbr > 0:
                _v = self.avg_per_cl_au_cosine[k][constants.AU_COSINE_MTR]
                self.avg_per_cl_au_cosine[k][constants.AU_COSINE_MTR] = _v / nbr

    def _build_std_cam_extractor(self, classifier, args):
        return build_std_cam_extractor(classifier=classifier, args=args)

    def _build_fcam_extractor(self, model, args):
        return build_fcam_extractor(model=model, args=args)

    def get_cam_one_sample(self, image: torch.Tensor, target: int,
                           ) -> Tuple[torch.Tensor, torch.Tensor]:

        img_shape = image.shape[2:]

        with autocast(enabled=self.args.amp_eval):
            output = self.model(image, target)

        if self.args.task == constants.STD_CL:

            if self.args.amp_eval:
                output = output.float()

            cl_logits = output
            cam = self.std_cam_extractor(class_idx=target,
                                         scores=cl_logits,
                                         normalized=True,
                                         reshape=img_shape if self.special1
                                         else None)

            # (h`, w`)

        elif self.args.task == constants.F_CL:

            if self.args.amp_eval:
                tmp = []
                for term in output:
                    tmp.append(term.float() if term is not None else None)
                output = tmp

            cl_logits, fcams, im_recon = output
            cam = self.fcam_extractor(argmax=self.fcam_argmax)
            # (h`, w`)

        else:
            raise NotImplementedError

        if self.args.amp_eval:
            cam = cam.float()

        # Quick fix: todo...
        cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)
        # cl_logits: 1, nc.

        return cam, cl_logits

    def layers_str_2_int(self):

        assert self.args.align_atten_to_heatmap

        align_atten_to_heatmap_layers = self.args.align_atten_to_heatmap_layers
        layers = align_atten_to_heatmap_layers
        assert isinstance(layers, str), type(layers)

        assert isinstance(layers, str), type(str)
        z = layers.split('-')
        z = [int(s) for s in z]
        assert len(z) > 0, f"{len(z)} | {layers}"

        return z

    def build_att(self, layer_id: int) -> torch.Tensor:

        assert self.args.align_atten_to_heatmap

        features = self.model.features
        assert features != []
        if layer_id == -1:
            layer_id = len(features) - 1
            for i, ft in enumerate(features):
                if ft.ndim == 4:
                    layer_id = i

        assert layer_id < len(self.model.features), f"{layer_id} " \
                                                    f"{len(features)}"

        with torch.no_grad():
            f = features[layer_id]
            attention = f.mean(dim=1, keepdim=True)  # b, 1, h, w

        return attention

    def minibatch_accum(self,
                        images: torch.Tensor,
                        targets: torch.Tensor,
                        image_ids: list,
                        image_size: list,
                        au_heatmaps: Union[torch.Tensor, None]
                        ) -> None:

        i = 0

        for image, target, image_id in zip(images, targets, image_ids):

            trg = target.item()
            au_heatmap = None

            if au_heatmaps is not None:
                au_heatmap = au_heatmaps[i].squeeze()  # h, w:same as image

                # check if valid au heatmap
                _valid = (torch.isinf(au_heatmap).sum() == 0)

                self.is_avail_avg_per_cl_au_cosine |= _valid

                if not _valid:
                    au_heatmap = None

            with torch.set_grad_enabled(self.req_grad):
                cam, cl_logits = self.get_cam_one_sample(
                    image=image.unsqueeze(0), target=trg)
                cl_logits = cl_logits.detach()

            with torch.no_grad():
                cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                    image_size,
                                    mode='bilinear',
                                    align_corners=False).squeeze(0).squeeze(0)
                cam = cam.detach()
                # todo:
                cam = torch.clamp(cam, min=0.0, max=1.0)

                trg_cl = self.int_to_cl[trg]
                # accumulate au cosine metric.
                self.update_avg_per_cl_au_cosine(cl=trg_cl,
                                                 au_heatmap=au_heatmap,
                                                 cam=cam
                                                 )

                # accumulate cams per class.
                if self.avg_per_cl_cams[trg_cl]['cam'] is None:
                    self.avg_per_cl_cams[trg_cl]['cam'] = cam
                else:
                    _v = self.avg_per_cl_cams[trg_cl]['cam']
                    self.avg_per_cl_cams[trg_cl]['cam'] = _v + cam

                _s = self.avg_per_cl_cams[trg_cl]['nbr']
                self.avg_per_cl_cams[trg_cl]['nbr'] = _s + 1.

                # accumulate heatmaps per class
                if au_heatmap is not None:
                    if self.avg_per_cl_aus_maps[trg_cl]['cam'] is None:
                        self.avg_per_cl_aus_maps[trg_cl]['cam'] = au_heatmap
                    else:
                        _v = self.avg_per_cl_aus_maps[trg_cl]['cam']
                        self.avg_per_cl_aus_maps[trg_cl]['cam'] = _v + au_heatmap

                    _s = self.avg_per_cl_aus_maps[trg_cl]['nbr']
                    self.avg_per_cl_aus_maps[trg_cl]['nbr'] = _s + 1.


                # accumulate attention map per class.
                if self.args.align_atten_to_heatmap:
                    idx_layer_att = self.layers_str_2_int()[-1]
                    attention = self.build_att(layer_id=idx_layer_att)
                    # h, w
                    # todo: generalize for other layers.

                    attention = F.interpolate(input=attention,
                                              size=image_size,
                                              mode='bilinear',
                                              align_corners=True,
                                              antialias=True
                                              )
                    attention = torch.clamp(attention, min=0.0, max=1.0)
                    attention = attention.detach().squeeze()

                    if self.avg_per_cl_att_maps[trg_cl]['cam'] is None:
                        self.avg_per_cl_att_maps[trg_cl]['cam'] = attention
                    else:
                        _v = self.avg_per_cl_att_maps[trg_cl]['cam']
                        self.avg_per_cl_att_maps[trg_cl]['cam'] = _v + attention

                    _s = self.avg_per_cl_att_maps[trg_cl]['nbr']
                    self.avg_per_cl_att_maps[trg_cl]['nbr'] = _s + 1.

                # cam: (h, w)
                cam = t2n(cam)
                assert cl_logits.ndim == 2
                _, preds_ordered = torch.sort(input=cl_logits.cpu().squeeze(0),
                                              descending=True,
                                              stable=True
                                              )

                self.evaluator.accumulate(cam,
                                          image_id,
                                          trg,
                                          preds_ordered.numpy()
                                          )


            i += 1

    def update_avg_per_cl_au_cosine(self,
                                    cl: str,
                                    au_heatmap: Union[torch.Tensor, None],
                                    cam: torch.Tensor
                                    ):
        # order: attention_layer0, attention_layer1, ...,
        # attention_last_layer, cam
        if au_heatmap is None:  # invalid action unit map.
            return None

        au_heatmap = au_heatmap.squeeze()  # h, w
        h, w = au_heatmap.shape

        au_heatmap = au_heatmap.contiguous().view(-1)

        # layer_attentions
        layerwise_features: list = self.model.features

        # keep only 2d features, case by case.
        if self.args.method == constants.METHOD_APVIT:
            assert len(layerwise_features) == 3, len(layerwise_features)
            # img, 2d, dense
            assert layerwise_features[0].ndim == 4, layerwise_features[0].ndim
            assert layerwise_features[1].ndim == 4, layerwise_features[1].ndim
            assert layerwise_features[2].ndim == 2, layerwise_features[2].ndim
            layerwise_features.pop(-1)

        if self.args.method == constants.METHOD_TSCAM:
            # x, attention ft maps, semantic agnostric_att map.
            pass


        l_attentions = []
        # todo: self.model.features may contain dense features at the last
        #  layers...
        for att in layerwise_features:
            assert att.ndim == 4, att.ndim  # 1, nfeatures, h', w'
            assert att.shape[0] == 1, att.shape[0]

            attention = att.mean(dim=1, keepdim=True).squeeze()  # h', w'
            _min = torch.min(attention)
            _max = torch.max(attention)
            attention = F.interpolate(input=attention.unsqueeze(0).unsqueeze(0),
                                      size=(h, w),
                                      mode='bilinear',
                                      align_corners=True,
                                      antialias=True
                                      ).squeeze()
            attention = torch.clamp(attention, min=_min, max=_max)
            if _min < 0:
                # shift to yield always positive vals to compute consistent
                # cosine similarity.
                attention = attention + (-_min)

            l_attentions.append(attention)

        _min = torch.min(cam)
        _max = torch.max(cam)

        cam = F.interpolate(input=cam.unsqueeze(0).unsqueeze(0),
                            size=(h, w),
                            mode='bilinear',
                            align_corners=True,
                            antialias=True
                            ).squeeze()
        cam = torch.clamp(cam, min=_min, max=_max)

        mapx = l_attentions + [cam]
        mapx = [m.contiguous().view(-1) for m in mapx]
        cosines = []
        for m in mapx:
            cosines.append(compute_cosine_one_sample(m, au_heatmap).detach())

        cosines = torch.tensor([cosines], dtype=torch.float32,
                               requires_grad=False, device=cam.device
                               )
        cosines = torch.nan_to_num(cosines, nan=0.0, posinf=1.0, neginf=0.0)

        if self.avg_per_cl_au_cosine[cl][constants.AU_COSINE_MTR] is None:
            self.avg_per_cl_au_cosine[cl][constants.AU_COSINE_MTR] = cosines

        else:
            v = self.avg_per_cl_au_cosine[cl][constants.AU_COSINE_MTR]
            self.avg_per_cl_au_cosine[cl][constants.AU_COSINE_MTR] = v + cosines

        s = self.avg_per_cl_au_cosine[cl]['nbr']
        self.avg_per_cl_au_cosine[cl]['nbr'] = s + 1

    def normalizecam(self, cam):
        if self.args.task == constants.STD_CL:
            cam_normalized = normalize_scoremap(cam)
        elif self.args.task == constants.F_CL:
            cam_normalized = cam
        else:
            raise NotImplementedError
        return cam_normalized

    def fix_random(self):
        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = True

        torch.backends.cudnn.deterministic = True

    def compute_and_evaluate_cams(self):
        # flush.
        self.init_avg_per_cl_cams()
        self.init_avg_per_cl_aus_maps()
        self.init_avg_per_cl_att_maps()
        self.init_avg_per_cl_au_cosine()
        self.is_avail_avg_per_cl_au_cosine = False

        print("Computing and evaluating cams.")
        for batch_idx, (images, _, targets, image_ids, _, _, _, au_heatmap, _,
                        _, _) in tqdm(enumerate(self.loader),
                                      ncols=constants.NCOLS,
                                      total=len(self.loader)
                                      ):

            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda(self.args.c_cudaid)

            # todo: load au_heatmap here.

            if au_heatmap.ndim == 1:
                au_heatmap = None

            else:
                assert au_heatmap.ndim == 4, au_heatmap.ndim
                assert au_heatmap.shape[1] == 1, au_heatmap.shape[1]
                au_heatmap = au_heatmap.cuda(self.args.c_cudaid)

            self.minibatch_accum(images=images,
                                 targets=targets,
                                 au_heatmaps=au_heatmap,
                                 image_ids=image_ids,
                                 image_size=image_size
                                 )

            # # cams shape (batchsize, h, w)..
            # for cam, image_id in zip(cams, image_ids):
            #     # cams shape (h, w).
            #     assert cam.shape == image_size
            #
            #     # cam_resized = cv2.resize(cam, image_size,
            #     #                          interpolation=cv2.INTER_CUBIC)
            #
            #     cam_resized = cam
            #     cam_normalized = self.normalizecam(cam_resized)
            #     self.evaluator.accumulate(cam_normalized, image_id)

        if self.args.distributed:
            self.evaluator._synch_across_gpus()
            dist.barrier()

            self.synch_cams()
            self.synch_aus_maps()

        self.avg_cams()

        self.avg_aus_maps()

        if self.is_avail_avg_per_cl_au_cosine:
            if self.args.distributed:
                self.synch_au_cosine()

            self.avg_au_cosine()

        return self.evaluator.compute()

    def build_bbox(self, scoremap, image_id, tau: float):
        cam_threshold_list = [tau]

        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=cam_threshold_list,
            multi_contour_eval=self.evaluator.multi_contour_eval)

        assert len(boxes_at_thresholds) == 1
        assert len(number_of_box_list) == 1

        # nbrbox, 4
        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.evaluator.gt_bboxes[image_id]))  # (nbrbox, 1)

        multiple_iou = multiple_iou.flatten()
        idx = np.argmax(multiple_iou)
        bbox_iou = multiple_iou[idx]
        best_bbox = boxes_at_thresholds[idx]  # shape: (4,)

        return best_bbox, bbox_iou

    def build_mask(self):
        pass

    def assert_datatset_bbx(self):
        assert self.dataset_name in [constants.RAFDB, constants.AFFECTNET,
                                     constants.CUB,
                                     constants.ILSVRC]

    def assert_dataset_mask(self):
        assert self.dataset_name == constants.OpenImages

    def assert_tau_list(self):
        iou_threshold_list = self.evaluator.iou_threshold_list
        best_tau_list = self.evaluator.best_tau_list

        if isinstance(self.evaluator, BoxEvaluator):
            assert len(best_tau_list) == len(iou_threshold_list)
        elif isinstance(self.evaluator, MaskEvaluator):
            assert len(best_tau_list) == 1
        else:
            raise NotImplementedError

    def create_folder(self, fd):
        if not os.path.isdir(fd):
            os.makedirs(fd, exist_ok=True)

    def reformat_id(self, img_id):
        tmp = str(Path(img_id).with_suffix(''))
        return tmp.replace('/', '_')

    def get_ids_with_zero_ignore_mask(self):
        ids = self.loader.dataset.image_ids

        out = []
        for id in ids:
            ignore_file = os.path.join(self.evaluator.mask_root,
                                       self.evaluator.ignore_paths[id])
            ignore_box_mask = load_mask_image(ignore_file,
                                              (_RESIZE_LENGTH, _RESIZE_LENGTH))
            if ignore_box_mask.sum() == 0:
                out.append(id)

        return out

    def get_ids_with_one_bbx(self):

        ids = self.loader.dataset.image_ids

        out = []
        for id in ids:
            gt_bbx = self.evaluator.gt_bboxes[id]

            if len(gt_bbx) == 1:
                out.append(id)

        return out

    @staticmethod
    def switch_key_val_dict(d: dict) -> dict:
        out = dict()
        for k in d:
            assert d[k] not in out, 'more than 1 key with same value. wrong.'
            out[d[k]] = k

        return out

    @staticmethod
    def get_str_trg_prd_cl(pred_cl: int,
                           trg_cl: int,
                           int_cl: dict = None) -> str:
        if int_cl:
            return f'[CL] Trg: {int_cl[trg_cl]} - Prd: {int_cl[pred_cl]}'
        else:
            return f'[CL] Trg: {trg_cl} - Prd: {pred_cl}'

    def select_random_ids_to_draw(self, nbr: int) -> list:
        self.fix_random()
        if isinstance(self.evaluator, BoxEvaluator):
            ids = self.get_ids_with_one_bbx()
        elif isinstance(self.evaluator, MaskEvaluator):
            ids = self.get_ids_with_zero_ignore_mask()
        else:
            raise NotImplementedError

        total_s = len(ids)
        n = min(nbr, total_s)
        idx = np.random.choice(a=total_s, size=n, replace=False).flatten()

        selected_ids = [ids[z] for z in idx]
        self.fix_random()

        return selected_ids

    def stitch_horiz_n_imgs(self, l_paths: list, out_file: str):

        assert len(l_paths) > 0, len(l_paths)

        imgs = [Image.open(f).convert('RGB') for f in l_paths]
        w, h = imgs[0].size
        for i, im in enumerate(imgs):
            _w, _h = im.size
            assert _w == w, f"{_w} | {w} | {l_paths[i]}"
            assert _h == h, f"{_h} | {h} | {l_paths[i]}"

        space = 2
        n_imgs = len(imgs)
        all_w = w * n_imgs + space * (n_imgs - 1)
        img_out = Image.new("RGB", (all_w, h), "orange")

        delta = 0
        for i, img in enumerate(imgs):
            img_out.paste(img, (delta, 0), None)
            delta += w + space

        img_out.save(out_file, quality=300)

    def plot_avg_cams_per_cl(self):
        outfd = join(self.out_folder, 'per_cl_avg_cams')
        os.makedirs(outfd, exist_ok=True)

        l_paths_imgs = []

        for k in self.cl_to_int:
            nbr = self.avg_per_cl_cams[k]['nbr']

            img_path = join(outfd, f"{k}.png")

            if nbr > 0:
                cam = self.avg_per_cl_cams[k]['cam'].cpu().numpy()
                self.viz.plot_fer_avg_cam(cam=cam,
                                          class_name=k,
                                          outf=img_path
                                          )

                if self.args.dataset in [constants.RAFDB, constants.AFFECTNET]:
                    if k != constants.NEUTRAL:
                        l_paths_imgs.append(img_path)

                else:
                    raise NotImplementedError(self.args.dataset)

        if self.args.dataset in [constants.RAFDB, constants.AFFECTNET]:
            all_path = join(outfd, f"all-cams-{self.args.dataset}-{self.split}-"
                                   f"{self.args.method}.png")
            self.stitch_horiz_n_imgs(l_paths_imgs, all_path)
        else:
            raise NotImplementedError(self.args.dataset)

    def plot_avg_aus_maps(self):
        outfd = join(self.out_folder, 'per_cl_avg_aus_maps')
        os.makedirs(outfd, exist_ok=True)

        l_paths_imgs = []

        for k in self.cl_to_int:
            nbr = self.avg_per_cl_aus_maps[k]['nbr']

            img_path = join(outfd, f"{k}.png")

            if nbr > 0:
                map = self.avg_per_cl_aus_maps[k]['cam'].cpu().numpy()
                self.viz.plot_fer_avg_cam(cam=map,
                                          class_name=k,
                                          outf=img_path
                                          )

                if self.args.dataset in [constants.RAFDB, constants.AFFECTNET]:
                    if k != constants.NEUTRAL:
                        l_paths_imgs.append(img_path)

                else:
                    raise NotImplementedError(self.args.dataset)

        if self.args.dataset in [constants.RAFDB, constants.AFFECTNET]:
            all_path = join(outfd, f"all-aus_maps-{self.args.dataset}-"
                                   f"{self.split}.png")
            self.stitch_horiz_n_imgs(l_paths_imgs, all_path)
        else:
            raise NotImplementedError(self.args.dataset)

    def plot_avg_att_maps(self):
        idx_layer = self.layers_str_2_int()[-1]

        outfd = join(self.out_folder, f'per_cl_avg_att_maps-{idx_layer}')
        os.makedirs(outfd, exist_ok=True)

        l_paths_imgs = []

        for k in self.cl_to_int:
            nbr = self.avg_per_cl_att_maps[k]['nbr']

            img_path = join(outfd, f"{k}.png")

            if nbr > 0:
                map = self.avg_per_cl_att_maps[k]['cam'].cpu().numpy()
                self.viz.plot_fer_avg_cam(cam=map,
                                          class_name=k,
                                          outf=img_path
                                          )

                if self.args.dataset in [constants.RAFDB, constants.AFFECTNET]:
                    if k != constants.NEUTRAL:
                        l_paths_imgs.append(img_path)

                else:
                    raise NotImplementedError(self.args.dataset)

        if self.args.dataset in [constants.RAFDB, constants.AFFECTNET]:
            all_path = join(outfd, f"all-att_maps-{self.args.dataset}-"
                                   f"{self.split}-"
                                   f"{self.args.method}.png")
            self.stitch_horiz_n_imgs(l_paths_imgs, all_path)
        else:
            raise NotImplementedError(self.args.dataset)

    def draw_some_best_pred(self,
                            nbr=500,
                            separate=True,
                            compress=True,
                            store_imgs=False,
                            store_cams_alone=False,
                            cl_int=None,
                            dpi: int = None,
                            less_visual: bool = False
                            ):
        print('Drawing some pictures:')

        assert self.evaluator.best_tau_list != []

        iou_threshold_list = self.evaluator.iou_threshold_list
        best_tau_list = self.evaluator.best_tau_list
        self.assert_tau_list()

        int_cl = self.switch_key_val_dict(cl_int) if cl_int else self.int_to_cl

        ids_to_draw = self.select_random_ids_to_draw(nbr=nbr)

        true_plots = 0
        false_plots = 0
        # todo: optimize. unnecessary loading of useless samples.
        for idxb, (images, _, targets, image_ids, raw_imgs, _, lndmks_heatmap,
                   au_heatmap, _, _, _) in tqdm(
            enumerate(self.loader), ncols=constants.NCOLS,
            total=len(self.loader)):

            if false_plots > nbr:  # debug.
                break

            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda(self.args.c_cudaid)

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img, lnd_hmap, acu_hmap in zip(
                    images, targets, image_ids, raw_imgs, lndmks_heatmap,
                    au_heatmap):

                # if image_id not in ids_to_draw:
                #     continue

                if lnd_hmap.ndim == 0:
                    lnd_hmap = None

                if acu_hmap.ndim == 0:
                    acu_hmap = None

                assert not ((lnd_hmap is not None) and (acu_hmap is not None))
                heatmap = None
                tag_heatmap = ''

                if lnd_hmap is not None:
                    heatmap = lnd_hmap
                    tag_heatmap = 'Lndmks'

                elif acu_hmap is not None:
                    heatmap = acu_hmap
                    tag_heatmap = 'AUs'

                if heatmap is not None:  # 1, h, w.
                    if torch.isinf(heatmap[0, 0, 0]):  # invalid.
                        heatmap = None
                    else:
                        heatmap = heatmap.squeeze().numpy()

                # raw_img: 3, h, w
                raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
                raw_img = raw_img.astype(np.uint8)

                if store_imgs:
                    img_fd = join(self.out_folder, 'vizu/imgs')
                    os.makedirs(img_fd, exist_ok=True)
                    Image.fromarray(raw_img).save(join(img_fd, '{}.png'.format(
                        self.reformat_id(image_id))))

                with torch.set_grad_enabled(self.req_grad):
                    # todo: this is NOT low res.
                    low_cam, cl_logits = self.get_cam_one_sample(
                        image=image.unsqueeze(0), target=target.item())

                # Note: plot the attention of the layer used for alignment.
                #  if many, take the last one.
                idx_layer_att = self.layers_str_2_int()[-1]
                attention = self.build_att(layer_id=idx_layer_att)
                # h, w
                # todo: generalize for other layers.

                attention = F.interpolate(input=attention,
                                          size=raw_img.shape[:2],
                                          mode='bilinear',
                                          align_corners=True,
                                          antialias=True
                                          )
                attention = torch.clamp(attention, min=0.0, max=1.0)
                attention = attention.detach().squeeze().cpu().numpy()

                assert cl_logits.ndim == 2
                p_cl = cl_logits.argmax(dim=1).item()
                img_class_int = target.item()
                img_class_str = int_cl[img_class_int]

                tag_cl = self.get_str_trg_prd_cl(
                    pred_cl=p_cl, trg_cl=img_class_int, int_cl=int_cl)

                tag_pred = 'correct' if (p_cl == target.item()) else 'wrong'

                if (tag_pred == 'correct') and (true_plots > nbr):
                    continue

                if tag_pred == 'correct':
                    true_plots += 1

                else:
                    false_plots += 1


                with torch.no_grad():
                    cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                        image_size,
                                        mode='bilinear',
                                        align_corners=False
                                        ).squeeze(0).squeeze(0)

                    cam = torch.clamp(cam, min=0.0, max=1.)

                if store_cams_alone:
                    calone_fd = join(self.out_folder, 'vizu/cams_alone/low_res')
                    os.makedirs(calone_fd, exist_ok=True)

                    self.viz.plot_cam_raw(t2n(low_cam), outf=join(
                        calone_fd, '{}.png'.format(self.reformat_id(
                            image_id))), interpolation='none')

                    calone_fd = join(self.out_folder,
                                     'vizu/cams_alone/high_res')
                    os.makedirs(calone_fd, exist_ok=True)

                    self.viz.plot_cam_raw(t2n(cam), outf=join(
                        calone_fd, '{}.png'.format(self.reformat_id(
                            image_id))), interpolation='bilinear')

                cam = torch.clamp(cam, min=0.0, max=1.)
                cam = t2n(cam)

                # cams shape (h, w).
                assert cam.shape == image_size

                cam_resized = cam
                cam_normalized = cam_resized
                check_scoremap_validity(cam_normalized)

                if isinstance(self.evaluator, BoxEvaluator):
                    self.assert_datatset_bbx()
                    l_datum = []
                    for k, _THRESHOLD in enumerate(iou_threshold_list):
                        th_fd = join(self.out_folder, 'vizu',
                                     str(_THRESHOLD), tag_pred,
                                     int_cl[target.item()])

                        os.makedirs(th_fd, exist_ok=True)

                        tau = best_tau_list[k]
                        best_bbox, bbox_iou = self.build_bbox(
                            scoremap=cam_normalized, image_id=image_id,
                            tau=tau
                        )
                        gt_bbx = self.evaluator.gt_bboxes[image_id]
                        gt_bbx = np.array(gt_bbx)

                        datum = {'img': raw_img,
                                 'img_id': image_id,
                                 'gt_bbox': gt_bbx,
                                 'pred_bbox': best_bbox.reshape((1, 4)),
                                 'iou': bbox_iou,
                                 'tau': tau,
                                 'sigma': _THRESHOLD,
                                 'cam': cam_normalized,
                                 'tag_cl': tag_cl,
                                 'heatmap': heatmap,
                                 'tag_heatmap': tag_heatmap,
                                 'attention': attention,
                                 'img_class_str': img_class_str
                                 }

                        if separate:
                            outf = join(th_fd,
                                        f'{self.reformat_id(image_id)}.png')
                            show_cam = (self.args.method != constants.METHOD_APVIT)

                            if less_visual:
                                self.viz.plot_fer_single_less(
                                    datum=datum, outf=outf, dpi=dpi,
                                    show_cam=show_cam)
                            else:
                                self.viz.plot_fer_single(datum=datum, outf=outf)

                        l_datum.append(datum)

                    # th_fd = join(self.out_folder, 'vizu', 'all_taux')
                    # self.create_folder(th_fd)
                    # outf = join(th_fd, '{}.png'.format(self.reformat_id(
                    #     image_id)))
                    # self.viz.plot_multiple(data=l_datum, outf=outf)

                elif isinstance(self.evaluator, MaskEvaluator):
                    self.assert_dataset_mask()
                    tau = best_tau_list[0]
                    taux = sorted(list({0.5, 0.6, 0.7, 0.8, 0.9}))
                    gt_mask = get_mask(self.evaluator.mask_root,
                                       self.evaluator.mask_paths[image_id],
                                       self.evaluator.ignore_paths[image_id])
                    # gt_mask numpy.ndarray(size=(224, 224), dtype=np.uint8)

                    l_datum = []
                    for tau in taux:
                        th_fd = join(self.out_folder, 'vizu', str(tau))
                        self.create_folder(th_fd)
                        l_datum.append(
                            {'img': raw_img, 'img_id': image_id,
                             'gt_mask': gt_mask, 'tau': tau,
                             'best_tau': tau == best_tau_list[0],
                             'cam': cam_normalized}
                        )
                        # todo: plotting singles is not necessary for now.
                        # todo: control it latter for standalone inference.
                        if separate:
                            outf = join(th_fd, '{}.png'.format(self.reformat_id(
                                image_id)))
                            self.viz.plot_single(datum=l_datum[-1], outf=outf)

                    th_fd = join(self.out_folder, 'vizu', 'some_taux')
                    self.create_folder(th_fd)
                    outf = join(th_fd, '{}.png'.format(self.reformat_id(
                        image_id)))
                    self.viz.plot_multiple(data=l_datum, outf=outf)
                else:
                    raise NotImplementedError

        if compress:
            self.compress_fdout(self.out_folder, 'vizu')

    def compress_fdout(self, parent_fd, fd_trg):
        assert os.path.isdir(join(parent_fd, fd_trg))

        cmdx = [
            "cd {} ".format(parent_fd),
            "tar -cf {}.tar.gz {} ".format(fd_trg, fd_trg),
            "rm -r {} ".format(fd_trg)
        ]
        cmdx = " && ".join(cmdx)
        DLLogger.log("Running: {}".format(cmdx))
        try:
            subprocess.run(cmdx, shell=True, check=True)
        except subprocess.SubprocessError as e:
            DLLogger.log("Failed to run: {}. Error: {}".format(cmdx, e))

    def _watch_plot_perfs_meter(self, split: str, meters: dict, perfs: list,
                                fout: str):
        out = self.viz._watch_plot_perfs_meter(meters=meters, split=split,
                                               perfs=perfs, fout=fout)
        pklout = join(dirname(fout), '{}.pkl'.format(os.path.basename(
            fout).split('.')[0]))
        with open(pklout, 'wb') as fx:
            pkl.dump(out, file=fx, protocol=pkl.HIGHEST_PROTOCOL)

    def _watch_build_histogram_scores_cams(self, split):
        print('Building histogram of cams scores. ')
        threshs = list(np.arange(0, 1, 0.001)) + [1.]
        density = 0
        cnt = 0.
        for budx, (images, targets, image_ids, raw_imgs, _, _, _, _, _, _
                   ) in tqdm(enumerate(self.loader), ncols=constants.NCOLS,
                total=len(self.loader)):
            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda(self.args.c_cudaid)

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img in zip(
                    images, targets, image_ids, raw_imgs):

                # raw_img: 3, h, w
                raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
                raw_img = raw_img.astype(np.uint8)

                with torch.set_grad_enabled(self.req_grad):
                    low_cam, _ = self.get_cam_one_sample(
                        image=image.unsqueeze(0), target=target.item())

                with torch.no_grad():
                    cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                        image_size,
                                        mode='bilinear',
                                        align_corners=False
                                        ).squeeze(0).squeeze(0)

                    cam = torch.clamp(cam, min=0.0, max=1.)
                cam = t2n(cam)

                # cams shape (h, w).
                assert cam.shape == image_size

                cam_resized = cam
                cam_normalized = cam_resized
                check_scoremap_validity(cam_normalized)
                _density, bins = np.histogram(
                    cam_normalized,
                    bins=threshs,
                    density=False)

                _density = _density / _density.sum()
                density += _density

                cnt += 1.
        density /= cnt
        density *= 100.

        basename = 'histogram_normalized_cams-{}'.format(split)
        outf = join(self.out_folder, 'vizu/{}.png'.format(basename))
        self.viz._watch_plot_histogram_activations(density=density,
                                                   bins=bins,
                                                   outf=outf,
                                                   split=split)
        outf = join(self.out_folder, 'vizu/{}.pkl'.format(basename))
        with open(outf, 'wb') as fout:
            pkl.dump({'density': density, 'bins': bins}, file=fout,
                     protocol=pkl.HIGHEST_PROTOCOL)

    def _watch_build_store_std_cam_low(self, fdout):
        print('Building low res. cam and storing them.')
        for idx, (images, targets, image_ids, raw_imgs, _, _, _, _, _, _
                  ) in tqdm(enumerate(self.loader), ncols=constants.NCOLS,
                total=len(self.loader)):
            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda(self.args.c_cudaid)

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img in zip(
                    images, targets, image_ids, raw_imgs):

                with torch.set_grad_enabled(self.req_grad):
                    output = self.model(image.unsqueeze(0), target)

                    assert self.args.task == constants.STD_CL
                    cl_logits = output
                    cam = self.std_cam_extractor(class_idx=target.item(),
                                                 scores=cl_logits,
                                                 normalized=True,
                                                 reshape=image_size if
                                                 self.special1 else None)
                    # (h`, w`)

                    # Quick fix: todo...
                    cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)

                    cam = cam.detach().cpu()
                torch.save(cam, join(fdout, '{}.pt'.format(reformat_id(
                    image_id))))

    def _watch_get_std_cam_one_sample(self, image: torch.Tensor, target: int,
                                      ) -> torch.Tensor:

        output = self.model(image, target)
        img_sz = image.shape[2:]

        assert self.args.task == constants.STD_CL
        cl_logits = output
        cam = self.std_cam_extractor(class_idx=target,
                                     scores=cl_logits,
                                     normalized=False,
                                     reshape=img_sz if self.special1 else None)
        # (h`, w`)

        # Quick fix: todo...
        cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)
        return cam

    def _watch_analyze_entropy_std(self, nbr=200):
        assert self.args.task == constants.STD_CL

        ids_to_draw = self.select_random_ids_to_draw(nbr=nbr)
        vizu_cam_fd = join(self.out_folder, 'vizualization', 'cams')
        self.create_folder(vizu_cam_fd)

        for images, targets, image_ids, raw_imgs, _, _, _ in self.loader:
            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda(self.args.c_cudaid)

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img in zip(
                    images, targets, image_ids, raw_imgs):

                with torch.set_grad_enabled(self.req_grad):
                    low_cam = self._watch_get_std_cam_one_sample(
                        image=image.unsqueeze(0), target=target.item())

                cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                    image_size,
                                    mode='bilinear',
                                    align_corners=False).squeeze(0).squeeze(0)

                cam = cam.detach()

                cam_max_normed = max_normalize(cam)
                cam_entropy = entropy_cam(cam_max_normed)

                cam = t2n(cam)

                # raw_img: 3, h, w
                raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
                raw_img = raw_img.astype(np.uint8)

                # cams shape (h, w).
                assert cam.shape == image_size

                cam_resized = normalize_scoremap(cam.copy())
                cam_normalized = cam_resized
                check_scoremap_validity(cam_normalized)

                if image_id in ids_to_draw:

                    if isinstance(self.evaluator, BoxEvaluator):
                        self.assert_datatset_bbx()
                        gt_bbx = self.evaluator.gt_bboxes[image_id]
                        gt_bbx = np.array(gt_bbx)
                        datum = {
                            'visu': {
                                'cam': cam,
                                'cam_max_normed': t2n(cam_max_normed),
                                'cam_entropy': t2n(cam_entropy),
                                'cam_normalized': cam_normalized
                            },
                            'tags': {
                                'cam': 'Raw cam',
                                'cam_max_normed': 'Max-normed-cam',
                                'cam_entropy': 'Entropy-Cam',
                                'cam_normalized': 'Normed-cam'
                            },
                            'raw_img': raw_img,
                            'img_id': image_id,
                            'gt_bbox': gt_bbx
                        }
                        outf = join(vizu_cam_fd, '{}.png'.format(
                            self.reformat_id(image_id)))

                        self.viz._watch_plot_entropy(data=datum, outf=outf)

                    elif isinstance(self.evaluator, MaskEvaluator):
                       pass
                    else:
                        raise NotImplementedError

        # self.compress_fdout(self.out_folder, 'vizu')

    def _watch_analyze_thresh_std(self, nbr=200):
        assert self.args.task == constants.STD_CL

        ids_to_draw = self.select_random_ids_to_draw(nbr=nbr)
        vizu_cam_fd = join(self.out_folder, 'vizualization', 'cams-thresh')
        self.create_folder(vizu_cam_fd)

        threshs = list(np.arange(0, 1, 0.001))

        for images, targets, image_ids, raw_imgs, _, _, _ in self.loader:
            self.fix_random()

            image_size = images.shape[2:]
            images = images.cuda(self.args.c_cudaid)

            # cams shape (batchsize, h, w).
            for image, target, image_id, raw_img in zip(
                    images, targets, image_ids, raw_imgs):

                with torch.set_grad_enabled(self.req_grad):
                    low_cam = self._watch_get_std_cam_one_sample(
                        image=image.unsqueeze(0), target=target.item())

                with torch.no_grad():
                    cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                                        image_size,
                                        mode='bilinear',
                                        align_corners=False
                                        ).squeeze(0).squeeze(0)

                cam = cam.detach()
                cam = t2n(cam)

                # raw_img: 3, h, w
                raw_img = raw_img.permute(1, 2, 0).numpy()  # h, w, 3
                raw_img = raw_img.astype(np.uint8)

                # cams shape (h, w).
                assert cam.shape == image_size

                cam_resized = normalize_scoremap(cam.copy())
                cam_normalized = cam_resized
                check_scoremap_validity(cam_normalized)
                grey_cam = (cam_normalized * 255).astype(np.uint8)
                t0 = dt.datetime.now()
                otsu_thresh = threshold_otsu(grey_cam)
                print('otsu thresholding took: {}'.format(dt.datetime.now() -
                                                          t0))
                t0 = dt.datetime.now()
                li_thres = filters.threshold_li(grey_cam,
                                                initial_guess=otsu_thresh)
                print('li thresholding took: {}'.format(dt.datetime.now() -
                                                        t0))
                fg_otsu = (grey_cam > otsu_thresh).astype(np.uint8) * 255
                fg_li = (grey_cam > li_thres).astype(np.uint8) * 255

                fg_auto = (grey_cam > .2 * grey_cam.max()).astype(
                    np.uint8) * 255
                # erosion
                kernel_erose = torch.ones(11, 11).to(low_cam.device)
                fg_li_torch = torch.from_numpy(fg_li).to(
                    low_cam.device).unsqueeze(0).unsqueeze(0)
                li_eroded = kornia.morphology.erosion(fg_li_torch * 1.,
                                                      kernel_erose)
                li_eroded = li_eroded.cpu().squeeze().numpy().astype(
                    np.uint8) * 255
                fg_otsu_torch = torch.from_numpy(fg_otsu).to(
                    low_cam.device).unsqueeze(0).unsqueeze(0)
                ostu_eroded = fg_otsu_torch * 1.
                for kkk in range(2):
                    ostu_eroded = kornia.morphology.erosion(ostu_eroded,
                                                            kernel_erose)
                otsu_eroded = ostu_eroded.cpu().squeeze().numpy().astype(
                    np.uint8) * 255

                density, bins = np.histogram(
                    cam_normalized,
                    bins=threshs,
                    density=True)

                datum = {
                    'visu': {
                        'cam': cam,
                        'cam_normalized': cam_normalized,
                        'density': (density / density.sum(), bins),
                        'discrete_cam': (cam_normalized *
                                         255).astype(np.uint8),
                        'bin_otsu': fg_otsu,
                        'bin_li': fg_li,
                        'otsu_bin_eroded': otsu_eroded,
                        'li_bin_eroded':  li_eroded,
                        'fg_auto': fg_auto
                    },
                    'tags': {
                        'cam': 'Raw cam',
                        'cam_normalized': 'Cam normed',
                        'density': 'Cam-normed histo',
                        'discrete_cam': 'Discrete normed cam',
                        'bin_otsu': 'FG Otsu',
                        'bin_li': 'FG Li',
                        'otsu_bin_eroded': 'UTSU ERODED',
                        'li_bin_eroded': 'LI ERODED',
                        'fg_auto': 'FG AUTO'
                    },
                    'raw_img': raw_img,
                    'img_id': image_id,
                    'nbins': len(threshs),
                    'otsu_thresh': otsu_thresh / 255.,
                    'li_thres': li_thres / 255.
                }

                if image_id in ids_to_draw:

                    if isinstance(self.evaluator, BoxEvaluator):
                        self.assert_datatset_bbx()
                        gt_bbx = self.evaluator.gt_bboxes[image_id]
                        gt_bbx = np.array(gt_bbx)
                        datum['gt_bbox'] = gt_bbx

                    elif isinstance(self.evaluator, MaskEvaluator):
                        self.assert_dataset_mask()
                        gt_mask = get_mask(self.evaluator.mask_root,
                                           self.evaluator.mask_paths[image_id],
                                           self.evaluator.ignore_paths[image_id])
                        # gt_mask numpy.ndarray(size=(224, 224), dtype=np.uint8)
                        datum['gt_mask'] = gt_mask
                    else:
                        raise NotImplementedError

                    outf = join(vizu_cam_fd, '{}.png'.format(
                        self.reformat_id(image_id)))
                    self.viz._watch_plot_thresh(data=datum, outf=outf)
