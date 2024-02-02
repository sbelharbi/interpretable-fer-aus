import os
import random
import sys
import time
from os.path import dirname, abspath, join
from typing import Optional, Union, Tuple, List
from copy import deepcopy
import pickle as pkl
import math
import datetime as dt


import numpy as np
import torch
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import confusion_matrix
import yaml
from texttable import Texttable
import torch.nn.functional as F
import torch.distributed as dist

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.datasets.wsol_loader import get_data_loader

from dlib.utils.reproducibility import set_seed
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.shared import move_state_dict_to_device
from dlib.utils.shared import gpu_memory_stats
from dlib.utils.tools import get_cpu_device
from dlib.utils.tools import get_tag

from dlib.utils.shared import configure_metadata
from dlib.utils.shared import get_image_ids
from dlib.utils.shared import get_class_labels

from dlib.cams.selflearning import GetFastSeederSLFCAMS
from dlib.cams.selflearning import MBSeederSLFCAMS
from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor
from dlib.utils.shared import is_cc
from dlib.datasets.ilsvrc_manager import prepare_next_bucket
from dlib.datasets.ilsvrc_manager import prepare_vl_tst_sets
from dlib.datasets.ilsvrc_manager import delete_train

from dlib.parallel import sync_tensor_across_gpus
from dlib.parallel import MyDDP as DDP

from dlib.div_classifiers.parts.cutmix import cutmix as wsol_cutmix

from dlib.metrics.wsol_metrics import FastEvalSegmentation

from dlib.utils.tools import get_root_wsol_dataset

from dlib import losses


__all__ = ['Basic', 'Trainer']

# metrics start with:
_IGNORE_METRICS_LOG = ['localization', 'top1', 'top5']


class PerformanceMeter(object):
    def __init__(self, name_meter: str, split: str, higher_is_better=True):
        self.name_meter = name_meter
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None  # todo: fix / redundant / best_iter is
        # better. how to set best_epoch
        self.best_iter = 0
        self.value_per_epoch = []
        self.traces = []
        self.sumall = []

    def is_current_the_best(self) -> bool:
        return self.best_value == self.current_value

    def update(self, new_value):
        # todo: get best value index using master metric.
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]

        if self.name_meter == constants.CL_CONFMTX_MTR:
            self.traces.append(np.trace(new_value))
            best_trace = self.best_function(self.traces)

            best_iter = self.traces.index(best_trace)

        elif self.name_meter == constants.SEG_PERCLMTX_MTR:
            self.sumall.append(new_value.sum())
            best_sum = self.best_function(self.sumall)

            best_iter = self.sumall.index(best_sum)


        elif self.name_meter == constants.AU_COSINE_MTR:
            # rows: classes. colons: layers(0, 1, ...)|CAM.

            # delete layer-0: it holds input image. irrelevant cosine similarity.
            _new_value = np.delete(new_value, 0, axis=1)
            # todo: delete neutral cosine (always 0). irrelevant. no impact
            #  on order.
            self.traces.append(_new_value.mean())

            best_trace = self.best_function(self.traces)
            best_iter = self.traces.index(best_trace)

        else:
            best_value = self.best_function(self.value_per_epoch)
            best_iter = self.value_per_epoch.index(best_value)

        self.best_epoch = best_iter
        self.best_iter = best_iter
        self.best_value = self.value_per_epoch[best_iter]

        # todo: change to
        # idx = [i for i, x in enumerate(
        #  self.value_per_epoch) if x == self.best_value]
        # assert len(idx) > 0
        # self.best_epoch = idx[-1]

def compute_cnf_mtx(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred_ = pred.detach().cpu().numpy()
    target_ = target.detach().cpu().numpy()
    conf_mtx = confusion_matrix(y_true=target_,
                                y_pred=pred_,
                                sample_weight=None,
                                normalize='true'
                                )

    return conf_mtx

class Basic(object):
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _SPLITS = (constants.TRAINSET, constants.VALIDSET, constants.TESTSET)
    _EVAL_METRICS = ['loss',
                     constants.CL_ACCURACY_MTR,
                     constants.MAE_MTR,
                     constants.MSE_MTR,
                     constants.CL_CONFMTX_MTR,
                     constants.AU_COSINE_MTR,
                     constants.LOCALIZATION_MTR,
                     constants.DICE_MTR,
                     constants.IOU_MTR,
                     constants.SEG_PERCLMTX_MTR
                     ]
    # _BEST_CRITERION_METRIC = 'localization'
    _NUM_CLASSES_MAPPING = {
        constants.CUB: constants.NUMBER_CLASSES[constants.CUB],
        constants.RAFDB: constants.NUMBER_CLASSES[constants.RAFDB],
        constants.AFFECTNET: constants.NUMBER_CLASSES[constants.AFFECTNET],
        constants.ILSVRC: constants.NUMBER_CLASSES[constants.ILSVRC],
        constants.OpenImages: constants.NUMBER_CLASSES[constants.OpenImages],
    }

    @property
    def _BEST_CRITERION_METRIC(self):
        assert self.inited
        assert self.args is not None

        return self.master_selection_metric

    def __init__(self, args):
        self.args = args
        self.inited = False
        
    def _set_performance_meters(self):
        self._EVAL_METRICS += ['localization_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        self._EVAL_METRICS += ['top1_loc_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        self._EVAL_METRICS += ['top5_loc_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(metric,
                                         split,
                                         higher_is_better=False
                                         if metric in [
                                             'loss',
                                             constants.MAE_MTR,
                                             constants.MSE_MTR] else True
                                         )
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict


class TrainSetCurriculumLearning(object):
    def __init__(self,
                 use_curriculum_l: bool,
                 cl_type: str,
                 epoch_rate: int,
                 metadata_root: str,
                 proxy_training_set: bool,
                 max_epochs: int,
                 cl_class_order: Optional[str] = None,
                 classes: Optional[List] = None,
                 random_p: Optional[float] = 1.
                 ):

        assert use_curriculum_l
        self.use_curriculum_l = use_curriculum_l

        split = constants.TRAINSET
        self.split = split
        self.max_epochs = max_epochs

        msg = f"{cl_type} | {constants.CURRICULUM_TYPES}"
        assert cl_type in constants.CURRICULUM_TYPES, msg
        assert isinstance(epoch_rate, int), type(epoch_rate)
        assert epoch_rate > 0, epoch_rate

        if cl_type == constants.CURRICULUM_CLASS:

            assert isinstance(classes, list), type(classes)
            for c in classes:
                assert isinstance(c, int), c

            assert len(set(classes)) == len(classes)  # unique elements.


            cl_class_order = self._prepare_curriculum_l_cl_class_order(
                str_cl_class_order=cl_class_order, classes=classes)

            assert isinstance(cl_class_order, list), type(cl_class_order)

            # sanity check.
            tmp = []
            for up in cl_class_order:
                assert isinstance(up, list), f"{up}: {type(up)}"
                tmp += up

                for c in up:
                    assert isinstance(c, int), type(c)

            nc = len(classes)
            _nc = len(tmp)
            msg = f"Curriculum learning: " \
                  f"total nbr_cls: {nc}, " \
                  f"provided updated of CUR-L classes: {_nc}"
            assert nc == _nc, msg

            n = max_epochs / float(epoch_rate)
            msg = f"Curriculum learning: " \
                  f"total number of updates: {len(cl_class_order) - 1}, " \
                  f"possible updates during epochs: {n}"
            assert n > len(cl_class_order) - 1, msg


        if cl_type == constants.CURRICULUM_RANDOM:
            assert isinstance(random_p, float), type(random_p)
            assert 0 < random_p < 1., random_p
        # CURRICULUM_CLASS
        self.cl_class_order = cl_class_order
        self.cl_class_counter = 0
        self.classes = classes

        # CURRICULUM_RANDOM
        self.random_p = random_p


        self.cl_type = cl_type
        self.epoch_rate = epoch_rate

        self.metadata_root = join(metadata_root, split)
        self.metadata = configure_metadata(self.metadata_root)
        self.proxy_training_set = proxy_training_set
        self.all_image_ids: list = get_image_ids(self.metadata,
                                                 proxy=proxy_training_set)
        self.all_image_labels: dict = get_class_labels(self.metadata)
        self.n_all = len(self.all_image_ids)


        self.current_image_ids: List = []
        self.left_over_ids: List = self.all_image_ids.copy()

        self.already_init = False

    @property
    def current_n(self):
        return 100. * len(self.current_image_ids) / float(self.n_all)

    @staticmethod
    def _prepare_curriculum_l_cl_class_order(str_cl_class_order: str,
                                             classes: List) -> List:
        """
        curriculum_l_cl_class_order: Indicates the order of classes.
        separators: * to separate between two updates, - to separate between
        classes in the same update. e.g.:
        str_cl_class_order = '1-2*6-5-4*3' means: [[1, 2], [6, 5, 4],
        [3]]. initial CL starts with [1, 2]. then, in the next update,
        we add [6, 5, 4]. The last update will add [3].

        :param curriculum_l_cl_class_order: str.
        :param classes: list.
        :return:
        """
        cl_class_order = []
        str_order = str_cl_class_order
        ups = str_order.split('*')
        assert len(ups) > 0, len(ups)

        for up in ups:
            cls = up.split('-')
            assert len(cls) > 0, len(cls)
            _tmp = [int(c) for c in cls]
            assert len(_tmp) > 0, len(_tmp)
            for c in _tmp:
                assert c in classes, f"{c} | {classes}"

            cl_class_order.append(_tmp)

        assert len(cl_class_order) > 0, len(cl_class_order)

        return cl_class_order

    def init_curriculum_class(self):
        assert self.cl_type == constants.CURRICULUM_CLASS

        init_cls = self.cl_class_order[0]
        assert isinstance(init_cls, list), type(init_cls)
        assert len(init_cls) > 0, len(init_cls)



        for e in init_cls:
            assert e in self.classes, f"{e} | {self.classes}"

        self.current_image_ids: List = []
        self.cl_class_counter = 0

        for k_id in self.left_over_ids:

            assert k_id not in self.current_image_ids, k_id

            if self.all_image_labels[k_id] in init_cls:
                self.current_image_ids.append(k_id)

        DLLogger.log(fmsg(f'Init. curriculum learning '
                          f'[{self.cl_type}] with classes: '
                          f'{init_cls} [{len(init_cls)} classes. '
                          f'N.samples: {self.current_n:.4f}%]'
                          )
                     )

    def update_curriculum_class(self):
        self.cl_class_counter += 1

        new_cls = []

        if self.cl_class_counter < len(self.cl_class_order):
            new_cls = self.cl_class_order[self.cl_class_counter]

        assert isinstance(new_cls, list), type(new_cls)

        if not new_cls:
            return 0

        for e in new_cls:
            assert e in self.classes, f"{e} | {self.classes}"

        for k_id in self.left_over_ids:

            assert k_id not in self.current_image_ids, k_id

            if self.all_image_labels[k_id] in new_cls:
                self.current_image_ids.append(k_id)

        DLLogger.log(fmsg(f'Update curriculum learning '
                          f'[{self.cl_type}] with classes {new_cls}'
                          f' [{len(new_cls)} classes. '
                          f'N.samples: {self.current_n:.4f}%]'
                          )
                     )


    def init_curriculum(self):

        assert not self.already_init

        assert self.use_curriculum_l

        if self.cl_type == constants.CURRICULUM_CLASS:
            self.init_curriculum_class()

        else:
            raise NotImplementedError(self.cl_type)

        self.shuffle_current_samples()

        self.update_leftover()
        self.already_init = True


    def shuffle_current_samples(self):

        for _ in range(1000):
            random.shuffle(self.current_image_ids)

    def update_curriculum(self, epoch: int):
        assert self.use_curriculum_l
        assert self.already_init

        if (epoch % self.epoch_rate) != 0:
            return 0

        if self.cl_type == constants.CURRICULUM_CLASS:
            self.update_curriculum_class()

        else:
            raise NotImplementedError(self.cl_type)

        self.shuffle_current_samples()
        self.update_leftover()

    def update_leftover(self):

        for im_id in self.current_image_ids:

            if im_id in self.left_over_ids:
                self.left_over_ids.remove(im_id)


    def __str__(self):
        return f"Curriculum learning: " \
               f"Used: {self.use_curriculum_l}, " \
               f"Type: {self.cl_type}, " \
               f"Epoch rate: {self.epoch_rate}, " \
               f"cl_class_order: {self.cl_class_order}."


class Trainer(Basic):

    def __init__(self,
                 args,
                 model,
                 optimizer,
                 lr_scheduler,
                 loss: losses.MasterLoss,
                 classifier=None):
        super(Trainer, self).__init__(args=args)

        self.device = torch.device(args.c_cudaid)
        self.args = args
        self.performance_meters = self._set_performance_meters()
        self.model = model

        self.master_selection_metric = args.master_selection_metric
        assert self.master_selection_metric in constants.METRICS, \
            self.master_selection_metric

        if isinstance(model, DDP):
            self._pytorch_model = self.model.module
        else:
            self._pytorch_model = self.model

        self.loss: losses.MasterLoss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        folds_path = join(root_dir, self.args.metadata_root)
        path_class_id = join(folds_path, 'class_id.yaml')
        with open(path_class_id, 'r') as fcl:
            cl_int = yaml.safe_load(fcl)

        self.cl_to_int: dict = cl_int
        self.int_to_cl: dict = self.switch_key_val_dict(cl_int)

        if is_cc() and self.args.ds_chunkable:
            dist.barrier()
            print(f'PROCESS --------> {os.getpid()}')
            if self.args.is_node_master:
                print(f'NODE MASTER {os.getpid()}')
                status1 = prepare_vl_tst_sets(dataset=self.args.dataset)
                if (status1[0] == -1) and self.args.is_master:
                    DLLogger.log(f'Error in preparing valid/test. '
                                 f'{status1[1]}. Exiting.')

                status2 = prepare_next_bucket(bucket=0,
                                              dataset=self.args.dataset)
                if (status2[0] == -1) and self.args.is_master:
                    DLLogger.log(f'Error in preparing bucket '
                                 f'{0}. {status2[1]}. Exiting.')

                if (status1[0] == -1) or (status2[0] == -1):
                    sys.exit()
            dist.barrier()

        # curriculum learning --------------------------------------------------
        self.curriculum_mnger = None
        curriculum_tr_ids = None

        if self.args.curriculum_l:
            assert not self.args.distributed, 'not supported with curriculum l.'
            classes = sorted(list(self.int_to_cl.keys()))

            cl_class_order = None

            if self.args.curriculum_l_type == constants.CURRICULUM_CLASS:
                cl_class_order = self.args.curriculum_l_cl_class_order


            self.curriculum_mnger = TrainSetCurriculumLearning(
                use_curriculum_l=self.args.curriculum_l,
                cl_type=self.args.curriculum_l_type,
                epoch_rate=self.args.curriculum_l_epoch_rate,
                metadata_root=self.args.metadata_root,
                proxy_training_set=self.args.proxy_training_set,
                max_epochs=self.args.max_epochs,
                cl_class_order=cl_class_order,
                classes=classes
            )

            DLLogger.log(fmsg(str(self.curriculum_mnger)))

            self.curriculum_mnger.init_curriculum()
            curriculum_tr_ids = self.curriculum_mnger.current_image_ids.copy()
        # ----------------------------------------------------------------------


        self.loaders, self.train_sampler = get_data_loader(
            args=self.args,
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder,
            tr_bucket=0,
            curriculum_tr_ids=curriculum_tr_ids
        )

        self.sl_mask_builder: MBSeederSLFCAMS = self._get_sl(args)

        self.epoch = 0
        self.counter = 0
        self.seed = int(args.MYSEED)
        self.default_seed = int(args.MYSEED)
        self.max_seed = (2 ** 32) - 1
        msg = f"seed must be: 0 <= {self.seed} <= {self.max_seed}"
        assert 0 <= self.seed <= self.max_seed, msg

        self.best_model = deepcopy(self._pytorch_model).to(self.cpu_device
                                                           ).eval()
        self.last_model = deepcopy(self._pytorch_model).to(self.cpu_device
                                                           ).eval()

        self.perf_meters_backup = None
        self.inited = True

        self.classifier = classifier
        self.std_cam_extractor = None
        if args.task == constants.F_CL:
            assert classifier is not None
            self.std_cam_extractor = self._build_std_cam_extractor(
                classifier=classifier, args=args)

        self.fcam_argmax = False
        self.fcam_argmax_previous = False

        self.t_init_epoch = dt.datetime.now()
        self.t_end_epoch = dt.datetime.now()


    @staticmethod
    def switch_key_val_dict(d: dict) -> dict:
        out = dict()
        for k in d:
            assert d[k] not in out, 'more than 1 key with same value. wrong.'
            out[d[k]] = k

        return out

    @staticmethod
    def _build_std_cam_extractor(classifier, args):
        classifier.eval()
        return build_std_cam_extractor(classifier=classifier, args=args)

    def _get_sl(self, args):
        return MBSeederSLFCAMS(
                min_=args.sl_min,
                max_=args.sl_max,
                ksz=args.sl_ksz,
                min_p=args.sl_min_p,
                fg_erode_k=args.sl_fg_erode_k,
                fg_erode_iter=args.sl_fg_erode_iter,
                support_background=args.model['support_background'],
                multi_label_flag=args.multi_label_flag,
                seg_ignore_idx=args.seg_ignore_idx)

    def prepare_std_cams_disq(self, std_cams: torch.Tensor,
                              image_size: Tuple) -> torch.Tensor:

        assert std_cams.ndim == 4
        cams = std_cams.detach()

        # cams: (bsz, 1, h, w)
        assert cams.ndim == 4
        # Quick fix: todo...
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        return cams

    def get_std_cams_minibatch(self, images, targets) -> torch.Tensor:
        # used only for task f_cl
        assert self.args.task == constants.F_CL
        assert images.ndim == 4
        image_size = images.shape[2:]

        cams = None
        for idx, (image, target) in enumerate(zip(images, targets)):
            cl_logits = self.classifier(image.unsqueeze(0))
            cam = self.std_cam_extractor(
                class_idx=target.item(), scores=cl_logits, normalized=True)
            # h`, w`
            # todo: set to false (normalize).

            cam = cam.detach().unsqueeze(0).unsqueeze(0)

            if cams is None:
                cams = cam
            else:
                cams = torch.vstack((cams, cam))

        # cams: (bsz, 1, h, w)
        assert cams.ndim == 4
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)
        cams = F.interpolate(cams,
                             image_size,
                             mode='bilinear',
                             align_corners=False)  # (bsz, 1, h, w)
        cams = torch.nan_to_num(cams, nan=0.0, posinf=1., neginf=0.0)

        return cams

    def is_seed_required(self, _epoch):
        cmd = (self.args.task == constants.F_CL)
        cmd &= ('self_learning_fcams' in self.loss.n_holder)
        cmd2 = False
        for _l in self.loss.losses:
            if isinstance(_l, losses.SelfLearningFcams):
                cmd2 = _l.is_on(_epoch=_epoch)

        return cmd and cmd2

    def _do_cutmix(self):
        return (self.args.method == constants.METHOD_CUTMIX and
                self.args.cutmix_prob > np.random.rand(1).item() and
                self.args.cutmix_beta > 0)

    def _wsol_training(self,
                       images,
                       raw_imgs,
                       targets,
                       std_cams,
                       lndmks_heatmap,
                       au_heatmap,
                       heatmap_seg,
                       bin_heatmap_seg
                       ):
        y_global = targets

        cutmix_holder = None
        if self.args.method == constants.METHOD_CUTMIX:
            if self._do_cutmix():
                cutmix_data = wsol_cutmix(
                    x=images,
                    target=targets,
                    beta=self.args.cutmix_beta,
                    std_cams=std_cams,
                    lndmks_heatmap=lndmks_heatmap,
                    au_heatmap=au_heatmap,
                    heatmap_seg=heatmap_seg,
                    bin_heatmap_seg=bin_heatmap_seg
                )
                images, std_cams, lndmks_heatmap, au_heatmap, heatmap_seg, \
                bin_heatmap_seg, target_a, target_b, lam = cutmix_data
                # todo: warning: raw_images is not done. not needed for this prj
                cutmix_holder = [target_a, target_b, lam]  # for ce loss.

                self.loss.set_cutmix_holder(cutmix_holder)

            else:
                self.loss.reset_cutmix_holder()  # double security. already
                # reset after loss.forward.


        output = self.model(images, y_global)

        if self.args.task == constants.STD_CL:
            cl_logits = output
            heatmap = None

            cnd1 = self.args.align_atten_to_heatmap
            cnd1 &= (self.args.align_atten_to_heatmap_type_heatmap ==
                     constants.HEATMAP_LNDMKS)

            cnd2 = self.args.align_atten_to_heatmap
            cnd2 &= (self.args.align_atten_to_heatmap_type_heatmap in
                     [constants.HEATMAP_AUNITS_LNMKS,
                      constants.HEATMAP_GENERIC_AUNITS_LNMKS])

            if cnd1:
                heatmap = lndmks_heatmap

            elif cnd2:
                heatmap = au_heatmap

            assert not (cnd1 and cnd2), f"{cnd1} {cnd2}"


            loss = self.loss(model=self.model,
                             epoch=self.epoch,
                             cl_logits=cl_logits,
                             glabel=y_global,
                             heatmap=heatmap,
                             seg_map=heatmap_seg,
                             bin_seg=bin_heatmap_seg
                             )
            logits = cl_logits

        elif self.args.task == constants.F_CL:
            cl_logits, fcams, im_recon = output

            if self.is_seed_required(_epoch=self.epoch):
                if std_cams is None:
                    cams_inter = self.get_std_cams_minibatch(images=images,
                                                             targets=targets)
                else:
                    cams_inter = std_cams

                with torch.no_grad():
                    seeds = self.sl_mask_builder(cams_inter)
            else:
                cams_inter, seeds = None, None

            loss = self.loss(
                epoch=self.epoch,
                cams_inter=cams_inter,
                fcams=fcams,
                cl_logits=cl_logits,
                glabel=y_global,
                raw_img=raw_imgs,
                x_in=self.model.x_in,
                im_recon=im_recon,
                seeds=seeds
            )
            logits = cl_logits
        else:
            raise NotImplementedError

        return logits, loss

    def set_model_to_train_mode(self):
        self.model.train()

    def on_epoch_start(self):
        self.t_init_epoch = dt.datetime.now()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        self.set_model_to_train_mode()

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)

    def on_epoch_end(self):
        self.loss.update_t()
        # todo: temp. delete later.
        self.loss.check_losses_status()

        self.t_end_epoch = dt.datetime.now()
        delta_t = self.t_end_epoch - self.t_init_epoch
        DLLogger.log(fmsg('Train epoch runtime: {}'.format(delta_t)))

    def random(self):
        self.counter = self.counter + 1
        seed = self.default_seed + self.counter
        self.seed = int(seed % self.max_seed)
        set_seed(seed=self.seed, verbose=False)

    def reload_data_bucket(self, tr_bucket: int):

        loaders, train_sampler = get_data_loader(
            args=self.args,
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder,
            tr_bucket=tr_bucket
        )

        self.train_sampler = train_sampler
        self.loaders[constants.TRAINSET] = loaders[constants.TRAINSET]

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch + tr_bucket)

    def reload_trainset_curriculum_learning(self):

        assert self.epoch > 1, self.epoch  # init should have been used. the
        # first epoch is 1.

        assert self.args.curriculum_l
        assert not self.args.ds_chunkable
        assert self.curriculum_mnger is not None
        assert self.curriculum_mnger.already_init

        n1 = len(self.curriculum_mnger.current_image_ids)

        self.curriculum_mnger.update_curriculum(epoch=self.epoch)

        images_id = self.curriculum_mnger.current_image_ids.copy()

        n2 = len(images_id)

        loaders, train_sampler = get_data_loader(
            args=self.args,
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.eval_batch_size,
            workers=self.args.num_workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
            std_cams_folder=self.args.std_cams_folder,
            tr_bucket=0,
            curriculum_tr_ids=images_id
        )

        self.train_sampler = train_sampler
        self.loaders[constants.TRAINSET] = loaders[constants.TRAINSET]

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)

        DLLogger.log(
            fmsg(f"Updated curriculum learning on trainset: "
                 f"{n1} --> {n2} samples "
                 f"[epoch {self.epoch}, "
                 f"type CL: {self.curriculum_mnger.cl_type}."
                 f'N.samples: {self.curriculum_mnger.current_n:.4f}%]'
                 )
        )

    def is_time_to_intern_validate(self,
                                   batch_index: int,
                                   max_nb_batches:int
                                   ) -> bool:
        p = self.args.valid_freq_mb

        assert isinstance(p, float), type(p)
        assert (p == -1) or (0 < p < 1), p

        if p == -1:
            return False

        if batch_index == 0:
            return False

        n = max(int(p * max_nb_batches), 1)

        if (batch_index % n) == 0:
            return True

    def intern_valid_model_select(self, batch_idx: int, max_nb_batches: int):

        self.evaluate(self.epoch,
                      split=constants.VALIDSET,
                      eval_loc=False,
                      plot_do_segmentation=False,
                      batch_idx=batch_idx,
                      max_nb_batches=max_nb_batches
                      )
        self.model_selection(split=constants.VALIDSET)

        if self.args.is_master:
            self.report(self.epoch, split=constants.VALIDSET, show_epoch=False)

    def train(self, split, epoch):
        assert split == constants.TRAINSET

        self.epoch = epoch
        self.random()
        self.on_epoch_start()

        nbr_tr_bucket = self.args.nbr_buckets
        if not self.args.ds_chunkable:
            nbr_tr_bucket = 1

        # curriculum learning
        if (epoch > 1) and (self.curriculum_mnger is not None):
            self.reload_trainset_curriculum_learning()

        loader = self.loaders[split]

        total_loss = 0.0
        num_correct = 0
        num_images = 0

        scaler = GradScaler(enabled=self.args.amp)

        for bucket in range(nbr_tr_bucket):

            status = 0
            if self.args.ds_chunkable:
                if is_cc():
                    dist.barrier()
                    if self.args.is_node_master:
                        if bucket > 0:
                            delete_train(bucket=bucket - 1,
                                         dataset=self.args.dataset)

                        status = prepare_next_bucket(bucket=bucket,
                                                     dataset=self.args.dataset)
                        if (status == -1) and self.args.is_master:
                            DLLogger.log(f'Error in preparing bucket '
                                         f'{bucket}. Exiting.')

                dist.barrier()
                if status == -1:
                    sys.exit()
                self.reload_data_bucket(tr_bucket=bucket)
                loader = self.loaders[split]

            dist.barrier()

            for batch_idx, (images, _, targets, _, raw_imgs,
                            std_cams,
                            lndmks_heatmap, au_heatmap, heatmap_seg,
                            bin_heatmap_seg) in tqdm(
                    enumerate(loader), ncols=constants.NCOLS,
                    total=len(loader), desc=f'BUCKET {bucket}/{nbr_tr_bucket}'):

                self.random()

                self.set_model_to_train_mode()

                images = images.cuda(self.args.c_cudaid)
                targets = targets.cuda(self.args.c_cudaid)

                if std_cams.ndim == 1:
                    std_cams = None
                else:
                    assert std_cams.ndim == 4
                    std_cams = std_cams.cuda(self.args.c_cudaid)

                    with autocast(enabled=self.args.amp):
                        with torch.no_grad():
                            std_cams = self.prepare_std_cams_disq(
                                std_cams=std_cams, image_size=images.shape[2:])

                if lndmks_heatmap.ndim == 1:
                    lndmks_heatmap = None
                else:
                    assert lndmks_heatmap.ndim == 4, lndmks_heatmap.ndim
                    assert lndmks_heatmap.shape[1] == 1, lndmks_heatmap.shape[1]
                    lndmks_heatmap = lndmks_heatmap.cuda(self.args.c_cudaid)

                if au_heatmap.ndim == 1:
                    au_heatmap = None

                else:
                    assert au_heatmap.ndim == 4, au_heatmap.ndim
                    assert au_heatmap.shape[1] == 1, au_heatmap.shape[1]
                    au_heatmap = au_heatmap.cuda(self.args.c_cudaid)

                if heatmap_seg.ndim == 1:
                    heatmap_seg = None

                else:
                    assert heatmap_seg.ndim == 4, heatmap_seg.ndim
                    assert heatmap_seg.shape[1] == 1, heatmap_seg.shape[1]
                    heatmap_seg = heatmap_seg.cuda(self.args.c_cudaid)

                if bin_heatmap_seg.ndim == 1:
                    bin_heatmap_seg = None

                else:
                    assert bin_heatmap_seg.ndim == 4, bin_heatmap_seg.ndim
                    assert bin_heatmap_seg.shape[1] == 1, bin_heatmap_seg.shape[1]
                    bin_heatmap_seg = bin_heatmap_seg.cuda(self.args.c_cudaid)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.args.amp):
                    logits, loss = self._wsol_training(images,
                                                       raw_imgs,
                                                       targets,
                                                       std_cams,
                                                       lndmks_heatmap,
                                                       au_heatmap,
                                                       heatmap_seg,
                                                       bin_heatmap_seg
                                                       )

                with torch.no_grad():
                    pred = logits.argmax(dim=1).detach()

                    total_loss += loss.detach() * images.size(0)
                    num_correct += (pred == targets).sum()
                    num_images += images.size(0)

                scaler.scale(loss).backward()
                # --------------------------------------------------------------
                grad_norm = self.args.optimizer['opt__clipgrad']
                assert grad_norm >= 0., grad_norm
                if grad_norm > 0:
                    scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=grad_norm,
                        norm_type=2
                    )
                # --------------------------------------------------------------


                scaler.step(self.optimizer)
                scaler.update()

                if self.is_time_to_intern_validate(batch_index=batch_idx,
                                                   max_nb_batches=len(loader)
                                                   ):
                    self.intern_valid_model_select(batch_idx, len(loader))

        num_correct = sync_tensor_across_gpus(num_correct.view(1, )).sum()
        nxx = torch.tensor([num_images], dtype=torch.float,
                           requires_grad=False, device=torch.device(
                self.args.c_cudaid)).view(1, )
        num_images = sync_tensor_across_gpus(nxx).sum().item()
        total_loss = sync_tensor_across_gpus(total_loss.view(1, )).sum()

        loss_average = total_loss.item() / float(num_images)
        classification_acc = num_correct.item() / float(num_images) * 100
        dist.barrier()

        self.performance_meters[split][constants.CL_ACCURACY_MTR].update(
            classification_acc)
        self.performance_meters[split]['loss'].update(loss_average)

        self.on_epoch_end()

        return dict(classification_acc=classification_acc,
                    loss=loss_average)

    def print_performances(self, checkpoint_type=None):
        # todo: REMOVE. RETIRED. use self.report().
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        for split in self._SPLITS:
            for metric in self._EVAL_METRICS:
                current_performance = \
                    self.performance_meters[split][metric].current_value
                if current_performance is not None:
                    DLLogger.log(
                        "Split {}, metric {}, current value: {}".format(
                         split, metric, current_performance))
                    if split != constants.TESTSET:
                        DLLogger.log(
                            "Split {}, metric {}, best value: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_value))
                        DLLogger.log(
                            "Split {}, metric {}, best epoch: {}".format(
                             split, metric,
                             self.performance_meters[split][metric].best_epoch))

    def serialize_perf_meter(self) -> dict:
        return {
            split: {
                metric: vars(self.performance_meters[split][metric])
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }

    def save_performances(self, epoch=None, checkpoint_type=None):
        tag = '' if checkpoint_type is None else '_{}'.format(checkpoint_type)

        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = '_Argmax_True'

        log_path = join(self.args.outd, 'performance_log{}{}.pickle'.format(
            tag, tagargmax))
        with open(log_path, 'wb') as f:
            pkl.dump(self.serialize_perf_meter(), f)

        log_path = join(self.args.outd, 'performance_log{}{}.txt'.format(
            tag, tagargmax))
        with open(log_path, 'w') as f:
            f.write("PERF - CHECKPOINT {}  - EPOCH {}  {} \n".format(
                checkpoint_type, epoch, tagargmax))

            for split in self._SPLITS:
                for metric in self._EVAL_METRICS:

                    if metric.startswith(tuple(_IGNORE_METRICS_LOG)):
                        continue

                    c_val = self.performance_meters[split][metric].current_value
                    b_val = self.performance_meters[split][metric].best_value

                    tagx = ''
                    if metric == self.master_selection_metric:
                        tagx = '(MASTER)'

                    if (c_val is not None) and (
                            metric not in [constants.CL_CONFMTX_MTR,
                                           constants.AU_COSINE_MTR,
                                           constants.SEG_PERCLMTX_MTR
                                           ]
                    ):
                        f.write(f"split: {split}. {metric}{tagx}: {c_val} \n")
                        f.write(f"split: {split}. {metric}: {b_val}_best \n")
                    elif (c_val is not None) and (
                            metric == constants.CL_CONFMTX_MTR):
                        f.write(f"BEST:{metric}{tagx} \n")
                        f.write(f"{self.print_confusion_mtx(b_val)} \n")

                        # plot confusion mtx.
                        fdout = join(self.args.outd, str(checkpoint_type),
                                     split)
                        self.plot_save_confusion_mtx(
                            mtx=b_val, fdout=fdout,
                            name=f'confusion-matrix-{split}')

                    elif (c_val is not None) and (
                            metric == constants.SEG_PERCLMTX_MTR):
                        f.write(f"BEST:{metric}{tagx} \n")
                        f.write(f"{self.print_avg_per_cl_seg(b_val)} \n")

                        # plot confusion mtx.
                        fdout = join(self.args.outd, str(checkpoint_type),
                                     split)
                        self.plot_save_avg_per_cl_seg(
                            mtx=b_val, fdout=fdout,
                            name=f'seg-matrix-{split}')

                    elif (c_val is not None) and (
                            metric == constants.AU_COSINE_MTR):
                        f.write(f"BEST:{metric}{tagx} \n")
                        f.write(f"{self.print_avg_per_cl_au_cosine(b_val)} \n")

                        # plot cosine per (per_layer + cam) per expression.
                        fdout = join(self.args.outd, str(checkpoint_type),
                                     split)
                        self.plot_save_avg_per_cl_au_cosine(
                            mtx=b_val,
                            fdout=fdout,
                            name=f'action-unit-cosine-{split}'
                        )

    def cl_forward(self, images, targets):
        output = self.model(images, targets)

        if self.args.task == constants.STD_CL:
            cl_logits = output

        elif self.args.task == constants.F_CL:
            cl_logits, fcams, im_recon = output

        else:
            raise NotImplementedError

        return cl_logits

    def _compute_cl_perf(self,
                         loader,
                         split: str,
                         checkpoint_type: str,
                         plot_do_segmentation: bool,
                         plot_n_per_cl: int = 5
                         ) -> dict:

        do_segmentation = self.args.model['do_segmentation']

        if do_segmentation:
            fdout = None
            if plot_do_segmentation and isinstance(checkpoint_type, str):
                fdout = join(self.args.outd, checkpoint_type, split, 'visu',
                             'do_segmentation')

            seg_eval = FastEvalSegmentation(ds=self.args.dataset,
                                            fdout=fdout,
                                            plotit=plot_do_segmentation,
                                            distributed=self.args.distributed,
                                            plot_n_per_cl=plot_n_per_cl
                                            )

            DLLogger.log(fmsg(f"Evaluate classification/segmentation, "
                              f"split: {split}..."))
        else:
            DLLogger.log(fmsg(f"Evaluate classification, split: {split}..."))

        t0 = dt.datetime.now()

        num_correct = 0
        num_images = 0
        all_pred = None
        all_y = None

        # not supported.
        assert self.args.eval_daug_mask_img_heatmap_type != \
               constants.HEATMAP_PER_CLASS_AUNITS_LNMKS, self.args.eval_daug_mask_img_heatmap_type

        for i, (images, _, targets, image_ids, raw_imgs, _, _, _, _,
                heatmap_seg, bin_heatmap_seg) in enumerate(loader):
            images = images.cuda(self.args.c_cudaid)
            targets = targets.cuda(self.args.c_cudaid)
            with torch.no_grad():
                cl_logits = self.cl_forward(images, targets).detach()

                pred = cl_logits.argmax(dim=1)
                num_correct += (pred == targets).sum()
                num_images += images.size(0)

                if all_pred is None:
                    all_pred = pred
                    all_y = targets

                else:
                    all_pred = torch.cat((all_pred, pred))
                    all_y = torch.cat((all_y, targets))

                if do_segmentation:
                    pred_logit = self.model.segmentation_head.segment_logits

                    seg_eval.track_on_the_fly(pred_logit=pred_logit,
                                              true_seg=bin_heatmap_seg,
                                              true_heatmap=heatmap_seg,
                                              cl_label=targets,
                                              cl_pred=pred,
                                              raw_img=raw_imgs,
                                              image_id=image_ids
                                              )

        # sync
        num_correct = sync_tensor_across_gpus(num_correct.view(1, )).sum()
        nx = torch.tensor([num_images], dtype=torch.float,
                          requires_grad=False, device=torch.device(
                self.args.c_cudaid)).view(1, )
        num_images = sync_tensor_across_gpus(nx).sum().item()

        all_pred = sync_tensor_across_gpus(all_pred.view(-1, ))
        all_y = sync_tensor_across_gpus(all_y.view(-1, ))

        conf_mtx = compute_cnf_mtx(pred=all_pred, target=all_y)

        # classification_acc = num_correct / float(num_images) * 100
        classification_acc = ((all_pred == all_y).float().mean() * 100.).item()
        diff = (all_pred - all_y).float()
        mse = ((diff ** 2).mean()).item()
        mae = (diff.abs().mean()).item()

        if do_segmentation:
            seg_eval.compute_avg_metrics()

            if self.args.is_master:
                seg_eval.compress_visu()


        dist.barrier()

        DLLogger.log(fmsg(f"Classification evaluation time for split: {split}, "
                          f"{dt.datetime.now() - t0}"))

        out = {
            constants.CL_ACCURACY_MTR: classification_acc,
            constants.MSE_MTR: mse,
            constants.MAE_MTR: mae,
            constants.CL_CONFMTX_MTR: conf_mtx
        }

        if do_segmentation:
            out[constants.IOU_MTR] = seg_eval.avg_iou
            out[constants.DICE_MTR] = seg_eval.avg_dice
            out[constants.SEG_PERCLMTX_MTR] = seg_eval.get_seg_perclmtx()

        return out

    def evaluate(self,
                 epoch,
                 split,
                 checkpoint_type=None,
                 fcam_argmax=False,
                 eval_loc=True,
                 plot_do_segmentation=False,
                 plot_n_per_cl: int = 5,
                 batch_idx: int = -1,
                 max_nb_batches: int = -1
                 ):

        if fcam_argmax:
            assert self.args.task == constants.F_CL

        self.fcam_argmax_previous = self.fcam_argmax
        self.fcam_argmax = fcam_argmax
        tagargmax = ''
        if self.args.task == constants.F_CL:
            tagargmax = 'Argmax {}'.format(fcam_argmax)

        if batch_idx == -1:
            DLLogger.log(fmsg(f"Evaluate: Epoch {epoch} "
                              f"Split {split} {tagargmax}"))

        else:
            DLLogger.log(fmsg(f"Evaluate: Epoch {epoch} "
                              f"[MBatch: {batch_idx} / {max_nb_batches}] "
                              f"Split {split} {tagargmax}"))

        outd = None
        if split == constants.TESTSET:
            assert checkpoint_type is not None
            if fcam_argmax:
                outd = join(self.args.outd, checkpoint_type, 'argmax-true',
                            split)
            else:
                outd = join(self.args.outd, checkpoint_type, split)
            if not os.path.isdir(outd):
                os.makedirs(outd, exist_ok=True)

        set_seed(seed=self.default_seed, verbose=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.model.eval()

        # ======================================================================
        # classification.
        # ======================================================================

        cl_perf = self._compute_cl_perf(loader=self.loaders[split],
                                        split=split,
                                        checkpoint_type=checkpoint_type,
                                        plot_do_segmentation=plot_do_segmentation,
                                        plot_n_per_cl=plot_n_per_cl
                                        )

        self.performance_meters[split][constants.CL_ACCURACY_MTR].update(
            cl_perf[constants.CL_ACCURACY_MTR]
        )
        self.performance_meters[split][constants.MSE_MTR].update(
            cl_perf[constants.MSE_MTR]
        )
        self.performance_meters[split][constants.MAE_MTR].update(
            cl_perf[constants.MAE_MTR]
        )
        self.performance_meters[split][constants.CL_CONFMTX_MTR].update(
            cl_perf[constants.CL_CONFMTX_MTR]
        )

        # segmentation
        if self.args.model['do_segmentation']:
            self.performance_meters[split][constants.DICE_MTR].update(
                cl_perf[constants.DICE_MTR]
            )
            self.performance_meters[split][constants.IOU_MTR].update(
                cl_perf[constants.IOU_MTR]
            )
            self.performance_meters[split][constants.SEG_PERCLMTX_MTR].update(
                cl_perf[constants.SEG_PERCLMTX_MTR]
            )

        # ======================================================================
        # localization
        # ======================================================================

        if not eval_loc:
            return 0

        cam_curve_interval = self.args.cam_curve_interval
        cmdx = (split == constants.VALIDSET)
        cmdx &= self.args.dataset in [constants.RAFDB, constants.AFFECTNET,
                                      constants.CUB,
                                      constants.ILSVRC]
        if cmdx:
            cam_curve_interval = constants.VALID_FAST_CAM_CURVE_INTERVAL

        cam_computer = CAMComputer(
            args=deepcopy(self.args),
            model=self._pytorch_model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset,
            split=split,
            cam_curve_interval=cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            out_folder=outd,
            fcam_argmax=fcam_argmax
        )

        t0 = dt.datetime.now()

        cam_performance = cam_computer.compute_and_evaluate_cams()

        DLLogger.log(fmsg(f"CAM EVALUATE TIME of slit {split} :"
                          f" {dt.datetime.now() - t0}"))

        cosine_au = cam_computer.get_matrix_avg_per_cl_au_cosine()

        if cosine_au is not None:
            self.performance_meters[split][constants.AU_COSINE_MTR].update(
                cosine_au
            )

        if split == constants.TESTSET and self.args.is_master:
            folds_path = join(root_dir, self.args.metadata_root)
            path_class_id = join(folds_path, 'class_id.yaml')
            with open(path_class_id, 'r') as fcl:
                cl_int = yaml.safe_load(fcl)

            cam_computer.draw_some_best_pred(cl_int=cl_int)

            cam_computer.plot_avg_cams_per_cl()

            if self.args.align_atten_to_heatmap:
                cam_computer.plot_avg_aus_maps()
                cam_computer.plot_avg_att_maps()

        if self.args.multi_iou_eval or (self.args.dataset ==
                                        constants.OpenImages):
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

        self.performance_meters[split][constants.LOCALIZATION_MTR].update(
            loc_score)

        if self.args.dataset in (constants.RAFDB, constants.AFFECTNET,
                                 constants.CUB,
                                 constants.ILSVRC):
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    'localization_IOU_{}'.format(IOU_THRESHOLD)].update(
                    cam_performance[idx])

                self.performance_meters[split][
                    'top1_loc_{}'.format(IOU_THRESHOLD)].update(
                    cam_computer.evaluator.top1[idx])

                self.performance_meters[split][
                    'top5_loc_{}'.format(IOU_THRESHOLD)].update(
                    cam_computer.evaluator.top5[idx])

            if split == constants.TESTSET and self.args.is_master:
                curve_top_1_5 = cam_computer.evaluator.curve_top_1_5
                with open(join(outd, 'curves_top_1_5.pkl'), 'wb') as fc:
                    pkl.dump(curve_top_1_5, fc, protocol=pkl.HIGHEST_PROTOCOL)

                title = get_tag(self.args, checkpoint_type=checkpoint_type)
                title = 'Top1/5: {}'.format(title)

                if fcam_argmax:
                    title += '_argmax_true'
                else:
                    title += '_argmax_false'
                self.plot_perf_curves_top_1_5(curves=curve_top_1_5, fdout=outd,
                                              title=title)

        if split == constants.TESTSET and self.args.is_master:

            curves = cam_computer.evaluator.curve_s
            with open(join(outd, 'curves.pkl'), 'wb') as fc:
                pkl.dump(curves, fc, protocol=pkl.HIGHEST_PROTOCOL)

            title = get_tag(self.args, checkpoint_type=checkpoint_type)

            if fcam_argmax:
                title += '_argmax_true'
            else:
                title += '_argmax_false'
            self.plot_perf_curves(curves=curves, fdout=outd, title=title)

        # # todo: find memory leak.
        # torch.cuda.empty_cache()

    def plot_perf_curves_top_1_5(self, curves: dict, fdout: str, title: str):

        x_label = r'$\tau$'
        y_label = 'BoxAcc'

        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False)

        for i, top in enumerate(['top1', 'top5']):

            iouthres = sorted(list(curves[top].keys()))
            for iout in iouthres:
                axes[0, i].plot(curves['x'], curves[top][iout],
                                label=r'{}: $\sigma$={}'.format(top, iout))

            axes[0, i].xaxis.set_tick_params(labelsize=5)
            axes[0, i].yaxis.set_tick_params(labelsize=5)
            axes[0, i].set_xlabel(x_label, fontsize=8)
            axes[0, i].set_ylabel(y_label, fontsize=8)
            axes[0, i].grid(True)
            axes[0, i].legend(loc='best')
            axes[0, i].set_title(top)

        fig.suptitle(title, fontsize=8)
        plt.tight_layout()
        plt.show()
        fig.savefig(join(fdout, 'curves_top1_5.png'), bbox_inches='tight',
                    dpi=300)

    def plot_perf_curves(self, curves: dict, fdout: str, title: str):

        bbox = True
        x_label = r'$\tau$'
        y_label = 'BoxAcc'
        if 'y' in curves:
            bbox = False
            x_label = 'Recall'
            y_label = 'Precision'

        fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=True)

        if bbox:
            iouthres = sorted([kk for kk in curves.keys() if kk != 'x'])
            for iout in iouthres:
                ax.plot(curves['x'], curves[iout],
                        label=r'$\sigma$={}'.format(iout))
        else:
            ax.plot(curves['x'], curves['y'], color='tab:orange',
                    label='Precision/Recall')

        ax.xaxis.set_tick_params(labelsize=5)
        ax.yaxis.set_tick_params(labelsize=5)
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.grid(True)
        plt.legend(loc='best')
        fig.suptitle(title, fontsize=8)
        plt.tight_layout()
        plt.show()
        fig.savefig(join(fdout, 'curves_perf.png'), bbox_inches='tight',
                    dpi=300)

    def capture_perf_meters(self):
        self.perf_meters_backup = deepcopy(self.performance_meters)

    def switch_perf_meter_to_captured(self):
        self.performance_meters = deepcopy(self.perf_meters_backup)
        self.fcam_argmax = self.fcam_argmax_previous

    def save_args(self):
        self._save_args(path=join(self.args.outd, 'config_obj_final.yaml'))

    def _save_args(self, path):
        _path = path
        with open(_path, 'w') as f:
            self.args.tend = dt.datetime.now()
            yaml.dump(vars(self.args), f)

    @property
    def cpu_device(self):
        return get_cpu_device()

    def save_best_epoch(self, split):
        best_epoch = self.performance_meters[split][
            self._BEST_CRITERION_METRIC].best_epoch
        self.args.best_epoch = best_epoch

    def save_checkpoints(self, split):
        best_epoch = self.performance_meters[split][
            self._BEST_CRITERION_METRIC].best_epoch

        self._save_model(checkpoint_type=constants.BEST, epoch=best_epoch)

        max_epoch = self.args.max_epochs
        self._save_model(checkpoint_type=constants.LAST, epoch=max_epoch)

    def _save_model(self, checkpoint_type, epoch):
        assert checkpoint_type in [constants.BEST, constants.LAST]

        if checkpoint_type == constants.BEST:
            _model = deepcopy(self.best_model).to(self.cpu_device).eval()
        elif checkpoint_type == constants.LAST:
            _model = deepcopy(self.last_model).to(self.cpu_device).eval()
        else:
            raise NotImplementedError

        _model.flush()

        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        path = join(self.args.outd, tag)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        if self.args.task == constants.STD_CL:
            if self.args.method in [constants.METHOD_TSCAM,
                                      constants.METHOD_APVIT]:
                to_save = _model.state_dict()

            else:
                to_save = {
                    'encoder': _model.encoder.state_dict(),
                    'classification_head': _model.classification_head.state_dict()
                }

                if _model.segmentation_head is not None:
                    to_save['segmentation_head'] = \
                        _model.segmentation_head.state_dict()

            torch.save(to_save, join(path, 'model.pt'))

            # torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            # torch.save(_model.classification_head.state_dict(),
            #            join(path, 'classification_head.pt'))

        elif self.args.task == constants.F_CL:
            torch.save(_model.encoder.state_dict(), join(path, 'encoder.pt'))
            torch.save(_model.decoder.state_dict(), join(path, 'decoder.pt'))
            torch.save(_model.segmentation_head.state_dict(),
                       join(path, 'segmentation_head.pt'))
            if _model.reconstruction_head is not None:
                torch.save(_model.reconstruction_head.state_dict(),
                           join(path, 'reconstruction_head.pt'))

        else:
            raise NotImplementedError

        self._save_args(path=join(path, 'config_model.yaml'))
        DLLogger.log(message="Stored Model [CP: {} \t EPOCH: {} \t TAG: {}]:"
                             " {}".format(checkpoint_type, epoch, tag, path))

    def model_selection(self, split):

        self.model.flush()

        if self.performance_meters[split][
            self._BEST_CRITERION_METRIC].is_current_the_best():
            self.best_model = deepcopy(self._pytorch_model).to(self.cpu_device
                                                               ).eval()

    def on_end_training(self):
        self.last_model = deepcopy(self._pytorch_model).to(self.cpu_device
                                                           ).eval()


    def load_checkpoint(self, checkpoint_type):
        assert checkpoint_type in [constants.BEST, constants.LAST]
        tag = get_tag(self.args, checkpoint_type=checkpoint_type)
        path = join(self.args.outd, tag)

        self._pytorch_model.flush()

        if self.args.task == constants.STD_CL:
            all_w = torch.load(join(path, 'model.pt'),
                               map_location=self.cpu_device)

            if self.args.method in [constants.METHOD_TSCAM,
                                      constants.METHOD_APVIT]:
                w = move_state_dict_to_device(all_w, self.device)
                self._pytorch_model.load_state_dict(w, strict=True)

            else:

                encoder_w = all_w['encoder']
                classification_head_w = all_w['classification_head']


                encoder_w = move_state_dict_to_device(encoder_w, self.device)
                self._pytorch_model.encoder.super_load_state_dict(
                    encoder_w, strict=True)

                classification_head_w = move_state_dict_to_device(
                    classification_head_w, self.device)
                self._pytorch_model.classification_head.load_state_dict(
                    classification_head_w, strict=True)

                if self._pytorch_model.segmentation_head is not None:
                    segmentation_head_w = all_w['segmentation_head']
                    segmentation_head_w = move_state_dict_to_device(
                        segmentation_head_w, self.device)
                    self._pytorch_model.segmentation_head.load_state_dict(
                        segmentation_head_w, strict=True)

            # weights = torch.load(join(path, 'encoder.pt'),
            #                      map_location=self.device)
            # self.model.encoder.super_load_state_dict(weights, strict=True)
            #
            # weights = torch.load(join(path, 'classification_head.pt'),
            #                      map_location=self.device)
            # self.model.classification_head.load_state_dict(weights,
            #                                                strict=True)

        elif self.args.task == constants.F_CL:
            weights = torch.load(join(path, 'encoder.pt'),
                                 map_location=self.device)
            self.model.encoder.super_load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'decoder.pt'),
                                 map_location=self.device)
            self.model.decoder.load_state_dict(weights, strict=True)

            weights = torch.load(join(path, 'segmentation_head.pt'),
                                 map_location=self.device)
            self.model.segmentation_head.load_state_dict(weights, strict=True)

            if self.model.reconstruction_head is not None:
                weights = torch.load(join(path, 'reconstruction_head.pt'),
                                     map_location=self.device)
                self.model.reconstruction_head.load_state_dict(weights,
                                                               strict=True)
        else:
            raise NotImplementedError

        DLLogger.log("Checkpoint {} loaded.".format(path))

    def report_train(self, train_performance, epoch, split=constants.TRAINSET):
        DLLogger.log('REPORT EPOCH/{}: {}/classification: {}'.format(
            epoch, split, train_performance['classification_acc']))
        DLLogger.log('REPORT EPOCH/{}: {}/loss: {}'.format(
            epoch, split, train_performance['loss']))

    def report(self, epoch, split, checkpoint_type=None, show_epoch=True):
        tagargmax = ''
        if self.fcam_argmax:
            tagargmax = ' Argmax: True'
        if checkpoint_type is not None:
            DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
                checkpoint_type, tagargmax)))

        if show_epoch:
            DLLogger.log(f"EPOCH/{epoch}.")

        for metric in self._EVAL_METRICS:

            if metric.startswith(tuple(_IGNORE_METRICS_LOG)):
                continue

            c_val = self.performance_meters[split][metric].current_value
            best_val = self.performance_meters[split][metric].best_value
            best_ep = self.performance_meters[split][metric].best_epoch

            if (c_val is not None) and (
                    metric not in [constants.CL_CONFMTX_MTR,
                                   constants.AU_COSINE_MTR,
                                   constants.SEG_PERCLMTX_MTR
                                   ]
            ):
                tagx = ''
                if metric == self.master_selection_metric:
                    tagx = '(MASTER)'

                DLLogger.log(f"split: {split}. {metric}{tagx}: {c_val}")
                cnd = (metric == self._BEST_CRITERION_METRIC)
                cnd &= (split == constants.VALIDSET)

                if cnd:
                    DLLogger.log(f"split: {split}. {metric}{tagx}: "
                                 f"{best_val} [BEST] [BEST-EPOCH: {best_ep}] ")

            elif (c_val is not None) and (
                    metric in [constants.CL_CONFMTX_MTR,
                               constants.AU_COSINE_MTR,
                               constants.SEG_PERCLMTX_MTR
                               ]
            ):
                # todo. not necessary. overload logs.
                pass

    def print_confusion_mtx(self, cmtx: np.ndarray) -> str:
        header_type = ['t']
        keys = list(self.int_to_cl.keys())
        h, w = cmtx.shape
        assert len(keys) == h, f"{len(keys)} {h}"
        assert len(keys) == w, f"{len(keys)} {w}"

        keys = sorted(keys, reverse=False)
        t = Texttable()
        t.set_max_width(400)
        header = ['*']
        for k in keys:
            header_type.append('f')
            header.append(self.int_to_cl[k])

        t.header(header)
        t.set_cols_dtype(header_type)
        t.set_precision(6)

        for i in range(h):
            row = [self.int_to_cl[i]]
            for j in range(w):
                row.append(cmtx[i, j])

            t.add_row(row)

        return t.draw()

    def print_avg_per_cl_au_cosine(self, mtx: np.ndarray) -> str:
        ord_int_cls = sorted(list(self.int_to_cl.keys()), reverse=False)
        ord_str_cls = [self.int_to_cl[x] for x in ord_int_cls]

        # ----------------------------------------------------------------------
        # TODO: FIX when adding AU to neutral.
        # Delete neutral: no action units.
        neutral = constants.NEUTRAL
        assert neutral in ord_str_cls, f"{neutral} -- {ord_str_cls}"
        idx_neutral = ord_str_cls.index(neutral)

        # remove neutral: no action units.
        ord_str_cls.pop(idx_neutral)
        mtx = np.delete(mtx, idx_neutral, axis=0)

        # delete layer-0: it holds input image. irrelevant cosine similarity.
        mtx = np.delete(mtx, 0, axis=1)

        # ----------------------------------------------------------------------

        header_type = ['t']
        keys = ord_str_cls
        h, w = mtx.shape
        assert len(keys) == h, f"{len(keys)} {h}"

        t = Texttable()
        t.set_max_width(400)
        header = ['*']
        for i in range(w - 1):
            header_type.append('f')
            header.append(f"layer-{i + 1}")  # + 1: removed layer0.

        # cam
        header_type.append('f')
        header.append("CAM")

        t.header(header)
        t.set_cols_dtype(header_type)
        t.set_precision(6)

        for i in range(h):
            row = [ord_str_cls[i]]
            for j in range(w):
                row.append(mtx[i, j])

            t.add_row(row)

        row = ['Average']
        _avg = mtx.mean(axis=0)
        for j in range(w):
            row.append(_avg[j])

        t.add_row(row)

        return t.draw()

    def print_avg_per_cl_seg(self, mtx: np.ndarray) -> str:
        assert mtx.ndim == 2, mtx.ndim  # ncls, [iou, dice]
        assert mtx.shape[1] == 2, mtx.shape[1]  # iou, dice

        ord_int_cls = sorted(list(self.int_to_cl.keys()), reverse=False)
        ord_str_cls = [self.int_to_cl[x] for x in ord_int_cls]

        header_type = ['t']
        keys = ord_str_cls
        h, w = mtx.shape
        assert len(keys) == h, f"{len(keys)} {h}"

        t = Texttable()
        t.set_max_width(400)
        header = ['*']
        metrics = [constants.IOU_MTR, constants.DICE_MTR]
        for i in range(w):
            header_type.append('f')
            header.append(metrics[i])

        t.header(header)
        t.set_cols_dtype(header_type)
        t.set_precision(6)

        for i in range(h):
            row = [ord_str_cls[i]]
            for j in range(w):
                row.append(mtx[i, j])

            t.add_row(row)

        return t.draw()

    def adjust_learning_rate(self):
        self.lr_scheduler.step()

    def plot_save_confusion_mtx(self, mtx: np.ndarray, fdout: str, name: str):
        if not os.path.isdir(fdout):
            os.makedirs(fdout, exist_ok=True)

        keys = list(self.int_to_cl.keys())
        h, w = mtx.shape
        assert len(keys) == h, f"{len(keys)} {h}"
        assert len(keys) == w, f"{len(keys)} {w}"

        keys = sorted(keys, reverse=False)
        cls = [self.int_to_cl[k] for k in keys]

        plt.close('all')
        g = sns.heatmap(mtx, annot=True, cmap='Greens',
                        xticklabels=1, yticklabels=1)
        g.set_xticklabels(cls, fontsize=7)
        g.set_yticklabels(cls, rotation=0, fontsize=7)

        plt.title("Confusion matrix", fontsize=7)
        # plt.tight_layout()
        plt.ylabel("True class", fontsize=7),
        plt.xlabel("Predicted class", fontsize=7)

        # disp.plot()
        plt.savefig(join(fdout, f'{name}.png'), bbox_inches='tight', dpi=300)
        plt.close('all')

    def plot_save_avg_per_cl_au_cosine(self,
                                       mtx: np.ndarray,
                                       fdout: str,
                                       name: str
                                       ):
        if not os.path.isdir(fdout):
            os.makedirs(fdout, exist_ok=True)

        ord_int_cls = sorted(list(self.int_to_cl.keys()), reverse=False)
        ord_str_cls = [self.int_to_cl[x] for x in ord_int_cls]

        # ----------------------------------------------------------------------
        # TODO: remove when allowing neutral to have ROI.
        # Delete neutral: no action units.
        neutral = constants.NEUTRAL
        assert neutral in ord_str_cls, f"{neutral} -- {ord_str_cls}"
        idx_neutral = ord_str_cls.index(neutral)

        # remove neutral: no action units.
        ord_str_cls.pop(idx_neutral)
        mtx = np.delete(mtx, idx_neutral, axis=0)

        # delete layer-0: it holds input image. irrelevant cosine similarity.
        mtx = np.delete(mtx, 0, axis=1)

        # ----------------------------------------------------------------------

        h, w = mtx.shape
        assert len(ord_str_cls) == h, f"{len(ord_str_cls)} {h}"

        # include avg
        _avg = mtx.mean(axis=0).reshape((1, -1))  # 1, n
        mtx = np.concatenate((mtx, _avg), axis=0)  # h+1, w
        labels_x = []
        for i in range(w - 1):
            labels_x.append(f"layer-{i + 1}")  # +1: since we deleted layer-0.

        # cam
        labels_x.append("CAM")

        labels_y = ord_str_cls + ['Average']

        plt.close('all')
        g = sns.heatmap(mtx, annot=True, cmap='Greens',
                        xticklabels=1, yticklabels=1, fmt='.4f')
        g.set_xticklabels(labels_x, fontsize=7)
        g.set_yticklabels(labels_y, rotation=0, fontsize=7)

        plt.title("Cosine similarity with action units heatmap", fontsize=7)

        # disp.plot()
        plt.savefig(join(fdout, f'{name}.png'), bbox_inches='tight', dpi=300)
        plt.close('all')

    def plot_save_avg_per_cl_seg(self,
                                 mtx: np.ndarray,
                                 fdout: str,
                                 name: str
                                 ):

        assert mtx.ndim == 2, mtx.ndim  # ncls, [iou, dice]
        assert mtx.shape[1] == 2, mtx.shape[1]  # iou, dice

        if not os.path.isdir(fdout):
            os.makedirs(fdout, exist_ok=True)

        ord_int_cls = sorted(list(self.int_to_cl.keys()), reverse=False)
        ord_str_cls = [self.int_to_cl[x] for x in ord_int_cls]

        h, w = mtx.shape
        assert len(ord_str_cls) == h, f"{len(ord_str_cls)} {h}"

        labels_x = [constants.IOU_MTR, constants.DICE_MTR]
        labels_y = ord_str_cls

        plt.close('all')
        g = sns.heatmap(mtx, annot=True, cmap='Greens',
                        xticklabels=1, yticklabels=1, fmt='.4f')
        g.set_xticklabels(labels_x, fontsize=7)
        g.set_yticklabels(labels_y, rotation=0, fontsize=7)

        plt.title("Segmentation metrics", fontsize=7)

        # disp.plot()
        plt.savefig(join(fdout, f'{name}.png'), bbox_inches='tight', dpi=300)
        plt.close('all')

    def plot_meter(self, metrics: dict, filename: str, split: str,
                   title: str = '',
                   xlabel: str = '', best_iter: int = None):

        ncols = 4
        ks = list(metrics.keys())
        if len(ks) > ncols:
            nrows = math.ceil(len(ks) / float(ncols))
        else:
            nrows = 1
            ncols = len(ks)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False,
                                 sharey=False, squeeze=False)
        t = 0
        for i in range(nrows):
            for j in range(ncols):
                if t >= len(ks):
                    axes[i, j].set_visible(False)
                    t += 1
                    continue

                val = metrics[ks[t]]['value_per_epoch']
                x = list(range(len(val)))
                axes[i, j].plot(x, val, color='tab:orange')
                axes[i, j].set_title(ks[t], fontsize=4)
                axes[i, j].xaxis.set_tick_params(labelsize=4)
                axes[i, j].yaxis.set_tick_params(labelsize=4)
                axes[i, j].set_xlabel('#{}'.format(xlabel), fontsize=4)
                axes[i, j].grid(True)
                # axes[i, j].xaxis.set_major_locator(MaxNLocator(integer=True))

                if best_iter is not None:
                    bi = best_iter
                    if split == constants.TRAINSET:
                        bi = max(0, bi - 1)

                    try:
                        axes[i, j].plot([x[bi]],
                                        [val[bi]],
                                        marker='o',
                                        markersize=5,
                                        color="red")
                    except:
                        DLLogger.log(f'Failed: {split}. '
                                     f'Iter: {best_iter}. '
                                     f'Key: {ks[t]}. '
                                     f'N.Vals: {len(val)}. '
                                     f'Filename: {filename}. '
                                     f' Fixed it.')

                        DLLogger.flush()

                t += 1

        fig.suptitle(title, fontsize=4)
        plt.tight_layout()

        fig.savefig(join(self.args.outd, '{}.png'.format(filename)),
                    bbox_inches='tight', dpi=300)

    def clean_metrics(self, metric: dict) -> dict:
        _metric = deepcopy(metric)
        l = []
        for k in _metric.keys():
            cd = (_metric[k]['value_per_epoch'] == [])
            cd |= (_metric[k]['value_per_epoch'] == [np.inf])
            cd |= (_metric[k]['value_per_epoch'] == [-np.inf])
            cd |= (k == constants.CL_CONFMTX_MTR)
            cd |= (k == constants.AU_COSINE_MTR)
            cd |= (k == constants.SEG_PERCLMTX_MTR)

            if cd:
                l.append(k)

        for k in l:
            _metric.pop(k, None)

        return _metric

    def plot_perfs_meter(self):
        meters = self.serialize_perf_meter()
        xlabel = 'epochs'

        best_epoch = self.performance_meters[constants.VALIDSET][
            self._BEST_CRITERION_METRIC].best_epoch  # todo: fix.

        for split in [constants.TRAINSET, constants.VALIDSET]:
            title = f'DS: {self.args.dataset}, ' \
                    f'Split: {split}, ' \
                    f'box_v2_metric: {self.args.box_v2_metric}. ' \
                    f'Best iter.:' \
                    f'{best_epoch} {xlabel}. ' \
                    f'Master: {self._BEST_CRITERION_METRIC}'

            filename = '{}-{}-boxv2-{}'.format(
                self.args.dataset, split, self.args.box_v2_metric)
            self.plot_meter(
                self.clean_metrics(meters[split]), filename=filename,
                split=split, title=title, xlabel=xlabel, best_iter=best_epoch)

