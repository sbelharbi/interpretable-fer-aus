import os
import time
from copy import deepcopy
import sys
from os.path import dirname, abspath, join
import threading
from copy import deepcopy
from typing import Tuple
import subprocess

import cv2
import numpy as np

import torch.utils.data as torchdata
import torch
import torch.nn.functional as F

import yaml

import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.filters import gaussian

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


from dlib.datasets.wsol_loader import get_image_ids
from dlib.datasets.wsol_loader import get_bounding_boxes
from dlib.datasets.wsol_loader import get_image_sizes
from dlib.datasets.wsol_loader import get_mask_paths

from dlib.utils.shared import reformat_id

from dlib.utils.tools import check_scoremap_validity
from dlib.utils.tools import check_box_convention

from dlib.configure import constants
from dlib.parallel import sync_tensor_across_gpus

import dlib.dllogger as DLLogger


_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = constants.SZ224  # 224
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


__all__ = ['BoxEvaluator',
           'MaskEvaluator',
           'load_mask_image',
           'get_mask',
           'calculate_multiple_iou',
           'compute_bboxes_from_scoremaps',
           'compute_cosine_one_sample',
           'FastEvalSegmentation'
           ]

def compute_cosine_one_sample(x1: torch.Tensor,
                              x2: torch.Tensor
                              ) -> torch.Tensor:
    cosine = F.cosine_similarity(x1.contiguous().view(-1),
                                 x2.contiguous().view(-1),
                                 dim=0,
                                 eps=1e-8)

    return cosine


def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    check_box_convention(box_a, 'x0y0x1y1')
    check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)

        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


class CamDataset(torchdata.Dataset):
    def __init__(self, scoremap_path, image_ids):
        self.scoremap_path = scoremap_path
        self.image_ids = image_ids

    def _load_cam(self, image_id):
        scoremap_file = os.path.join(self.scoremap_path, image_id + '.npy')
        return np.load(scoremap_file)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        cam = self._load_cam(image_id)
        return cam, image_id

    def __len__(self):
        return len(self.image_ids)


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """

    def __init__(self,
                 metadata,
                 dataset_name,
                 split,
                 cam_threshold_list,
                 iou_threshold_list,
                 mask_root,
                 multi_contour_eval,
                 args):
        self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_name = dataset_name
        self.split = split
        self.mask_root = mask_root
        self.multi_contour_eval = multi_contour_eval

        self.best_tau_list = []
        self.curve_s = None

        self.top1 = None
        self.top5 = None
        self.curve_top_1_5 = None

        self.args = args

    def accumulate(self, scoremap, image_id, target, preds_ordered):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

    def _synch_across_gpus(self):
        raise NotImplementedError


class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = _RESIZE_LENGTH
        self.cnt = 0
        self.num_correct = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.original_bboxes = get_bounding_boxes(self.metadata)
        self.image_sizes = get_image_sizes(self.metadata)
        self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)

        self.num_correct_top1 = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        self.num_correct_top5 = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def accumulate(self, scoremap, image_id, target, preds_ordered):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
            target: int. longint. the true class label. for bbox evaluation (
            top1/5).
            preds_ordered: numpy.ndarray. vector of predicted labels ordered
            from from the most probable the least probable. for evaluation of
            bbox using top1/5.
        """

        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_threshold_list,
            multi_contour_eval=self.multi_contour_eval)

        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array(self.gt_bboxes[image_id]))  # (nbr_boxes_in_img, 1)

        idx = 0
        sliced_multiple_iou = []  # length == number tau thresh
        for nr_box in number_of_box_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            # pick the maximum score iou among all the boxes.
            idx += nr_box

        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou) >= (_THRESHOLD/100))[0]

            self.num_correct[_THRESHOLD][correct_threshold_indices] += 1

            if target == preds_ordered[0]:
                self.num_correct_top1[_THRESHOLD][
                    correct_threshold_indices] += 1

            if target in preds_ordered[:5]:
                self.num_correct_top5[_THRESHOLD][
                    correct_threshold_indices] += 1

        self.cnt += 1

    def _synch_across_gpus(self):

        for tracker in [self.num_correct, self.num_correct_top1,
                        self.num_correct_top5]:
            for k in tracker.keys():
                _k_val = torch.from_numpy(tracker[k]).cuda(
                    self.args.c_cudaid).view(1, -1)

                tracker[k] = sync_tensor_across_gpus(
                    _k_val).sum(dim=0).cpu().view(-1).numpy()

        cnt = torch.tensor([self.cnt], dtype=torch.float,
                           requires_grad=False, device=torch.device(
                self.args.c_cudaid))

        cnt = sync_tensor_across_gpus(cnt)
        self.cnt = cnt.sum().cpu().item()

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []
        self.best_tau_list = []
        self.curve_s = {
            'x': self.cam_threshold_list
        }

        self.top1 = []
        self.top5 = []
        self.curve_top_1_5 = {
            'x': self.cam_threshold_list,
            'top1': dict(),
            'top5': dict()
        }

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            max_box_acc.append(localization_accuracies.max())

            self.curve_s[_THRESHOLD] = localization_accuracies

            self.best_tau_list.append(
                self.cam_threshold_list[np.argmax(localization_accuracies)])

            loc_acc = self.num_correct_top1[_THRESHOLD] * 100. / float(self.cnt)
            self.top1.append(loc_acc.max())

            self.curve_top_1_5['top1'][_THRESHOLD] = deepcopy(loc_acc)

            loc_acc = self.num_correct_top5[_THRESHOLD] * 100. / float(self.cnt)
            self.top5.append(loc_acc.max())

            self.curve_top_1_5['top5'][_THRESHOLD] = deepcopy(loc_acc)

        return max_box_acc


def load_mask_image(file_path, resize_size):
    """
    Args:
        file_path: string.
        resize_size: tuple of ints (height, width)
    Returns:
        mask: numpy.ndarray(dtype=numpy.float32, shape=(height, width))
    """
    mask = np.float32(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE))
    mask = cv2.resize(mask, resize_size, interpolation=cv2.INTER_NEAREST)
    return mask


def get_mask(mask_root, mask_paths, ignore_path):
    """
    Ignore mask is set as the ignore box region \setminus the ground truth
    foreground region.

    Args:
        mask_root: string.
        mask_paths: iterable of strings.
        ignore_path: string.

    Returns:
        mask: numpy.ndarray(size=(224, 224), dtype=np.uint8)
    """
    mask_all_instances = []
    for mask_path in mask_paths:
        mask_file = os.path.join(mask_root, mask_path)
        mask = load_mask_image(mask_file, (_RESIZE_LENGTH, _RESIZE_LENGTH))
        mask_all_instances.append(mask > 0.5)
    mask_all_instances = np.stack(mask_all_instances, axis=0).any(axis=0)

    ignore_file = os.path.join(mask_root, ignore_path)
    ignore_box_mask = load_mask_image(ignore_file,
                                      (_RESIZE_LENGTH, _RESIZE_LENGTH))
    ignore_box_mask = ignore_box_mask > 0.5

    ignore_mask = np.logical_and(ignore_box_mask,
                                 np.logical_not(mask_all_instances))

    if np.logical_and(ignore_mask, mask_all_instances).any():
        raise RuntimeError("Ignore and foreground masks intersect.")

    return (mask_all_instances.astype(np.uint8) +
            255 * ignore_mask.astype(np.uint8))


class MaskEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(MaskEvaluator, self).__init__(**kwargs)

        if self.dataset_name != constants.OpenImages:
            raise ValueError("Mask evaluation must be performed on OpenImages.")

        self.mask_paths, self.ignore_paths = get_mask_paths(self.metadata)

        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.num_bins = len(self.cam_threshold_list) + 2
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,
                                                   [1.0, 2.0, 3.0])

        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=float)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=float)

    def accumulate(self, scoremap, image_id, target=None, preds_ordered=None):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
            target and preds_ordered are not used in this case.
        """
        check_scoremap_validity(scoremap)
        gt_mask = get_mask(self.mask_root,
                           self.mask_paths[image_id],
                           self.ignore_paths[image_id])

        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(float)

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(float)

    def get_best_operating_point(self, recall, precision):
        # todo delete.
        tmp = precision[:-3] + recall[:-3]
        idx = np.argmax(tmp[1:]) + 1

        return self.cam_threshold_list[idx]

    def _synch_across_gpus(self):

        for tracker in [self.gt_true_score_hist, self.gt_false_score_hist]:
            _k_val = torch.from_numpy(tracker).cuda(self.args.c_cudaid)
            tracker = sync_tensor_across_gpus(_k_val).sum(dim=0).cpu().numpy()

    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        self.curve_s = {
            'x': recall,
            'y': precision
        }

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        self.best_tau_list = [self.get_best_operating_point(recall, precision)]

        # print("Mask AUC on split {}: {}".format(self.split, auc))
        return auc


class FastEvalSegmentation(object):
    def __init__(self,
                 ds: str,
                 fdout:str,
                 plotit: bool,
                 distributed: bool,
                 plot_n_per_cl: int = -1
                 ):
        super(FastEvalSegmentation, self).__init__()

        assert ds in constants.DATASETS, ds
        self.ds = ds
        folds_path = join(root_dir,
                          join(root_dir, f"folds/wsol-done-right-splits/{ds}")
                          )
        path_class_id = join(folds_path, 'class_id.yaml')
        with open(path_class_id, 'r') as fcl:
            cl_int = yaml.safe_load(fcl)

        int_cl = self.switch_key_val_dict(cl_int)

        self.cl_int = cl_int
        self.int_cl = int_cl

        assert isinstance(plotit, bool), type(plotit)
        self.plotit = plotit

        if plotit:
            os.makedirs(fdout, exist_ok=True)

        self.fdout = fdout

        self.per_cl_tracker: dict = {}
        self.avg_per_cl: dict = {}
        self.avg_dice = 0.0
        self.avg_iou = 0.0

        assert isinstance(plot_n_per_cl, int), type(plot_n_per_cl)
        assert plot_n_per_cl >= -1, plot_n_per_cl
        self.plot_n_per_cl = plot_n_per_cl
        self.plot_n: dict = {}

        assert isinstance(distributed, bool), distributed
        self.distributed = distributed

        self._reset_stats()

    @staticmethod
    def switch_key_val_dict(d: dict) -> dict:
        out = dict()
        for k in d:
            assert d[k] not in out, 'more than 1 key with same value. wrong.'
            out[d[k]] = k

        return out

    def _reset_stats(self):

        self.per_cl_tracker = {
            i: {
                constants.DICE_MTR: [],
                constants.IOU_MTR: []
            } for i in range(constants.NUMBER_CLASSES[self.ds])
        }

        self.avg_per_cl = {
            i: {
                constants.DICE_MTR: 0.0,
                constants.IOU_MTR: 0.0
            } for i in range(constants.NUMBER_CLASSES[self.ds])
        }

        self.plot_n = {
            i: 0 for i in range(constants.NUMBER_CLASSES[self.ds])
        }

    def _synch_tracker_across_gpus(self):

        for k in self.per_cl_tracker:
            for m in self.per_cl_tracker[k]:
                val: list = self.per_cl_tracker[k][m]
                val_tensore = torch.tensor(
                    val, dtype=torch.float,
                    requires_grad=False,
                    device=torch.device(torch.cuda.current_device())
                ).view(-1, )
                val_tensore = sync_tensor_across_gpus(val_tensore)
                val = val_tensore.cpu().numpy().tolist()
                self.per_cl_tracker[k][m] = val

    def compute_avg_metrics(self):

        if self.distributed:
            self._synch_tracker_across_gpus()

        dice = 0.0
        iou = 0.0
        n = 0.0

        for cl in self.per_cl_tracker:

            z = self.per_cl_tracker[cl]

            _dice = z[constants.DICE_MTR]
            _iou = z[constants.IOU_MTR]
            _n = float(len(_dice))

            sum_dice = sum(_dice)
            sum_iou = sum(_iou)

            dice += sum_dice
            iou += sum_iou
            n += _n

            if _n > 0:
                self.avg_per_cl[cl][constants.DICE_MTR] = float(sum_dice) / _n
                self.avg_per_cl[cl][constants.IOU_MTR] = float(sum_iou) / _n
            else:
                self.avg_per_cl[cl][constants.DICE_MTR] = 0.0
                self.avg_per_cl[cl][constants.IOU_MTR] = 0.0

        self.avg_dice = float(dice) / float(n)
        self.avg_iou = float(iou) / float(n)

    def get_seg_perclmtx(self):
        cls = sorted(list(self.avg_per_cl.keys()), reverse=False)
        n_cls = len(cls)
        out = np.zeros((n_cls, 2), dtype=float)  # ncls, [iou, dice]
        for k in cls:
            out[k, 0] = self.avg_per_cl[k][constants.IOU_MTR]
            out[k, 1] = self.avg_per_cl[k][constants.DICE_MTR]

            msg = f"{k} | {len(list(self.avg_per_cl[k].keys()))}"
            assert len(list(self.avg_per_cl[k].keys())) == 2, msg

        return out


    def track_on_the_fly(self,
                         pred_logit: torch.Tensor,
                         true_seg: torch.Tensor,
                         true_heatmap: torch.Tensor,
                         cl_label: torch.Tensor,
                         cl_pred: torch.Tensor,
                         raw_img: torch.Tensor,
                         image_id: list
                         ):

        assert pred_logit.ndim == 4, pred_logit.ndim  # bsz, 1, h', w'
        assert true_seg.ndim == 4, true_seg.ndim  # bsz, 1, h, w

        bsz = cl_label.shape[0]

        assert pred_logit.shape[1] == 1, pred_logit.shape[1]
        assert cl_label.ndim == 1, cl_label.ndim
        msg = f"{cl_label.shape[0]} | {pred_logit.shape[0]}"
        assert cl_label.shape[0] == pred_logit.shape[0], msg
        assert cl_label.shape == cl_pred.shape, f"{cl_label.shape} | " \
                                                f"{cl_pred.shape}"

        assert true_heatmap.shape == true_seg.shape, f"{true_heatmap.shape} |" \
                                                     f" {true_seg.shape}"

        assert len(image_id) == bsz, f"{len(image_id)} | {bsz}"
        assert raw_img.ndim == 4, raw_img.ndim  # bsz, 3, h, w

        pred_logit = pred_logit.detach()
        pred_logit = F.interpolate(pred_logit,
                                   size=true_seg.shape[2:],
                                   mode='bicubic',
                                   align_corners=True
                                  )

        assert pred_logit.shape == true_seg.shape, f"{pred_logit.shape} | " \
                                                   f"{true_seg.shape}"

        bsz = pred_logit.shape[0]
        cl_label = cl_label.cpu().numpy()

        pred_seg = torch.sigmoid(pred_logit).cpu().numpy()
        true_seg = true_seg.cpu().float().numpy()
        cl_pred = cl_pred.detach().cpu().numpy()

        for i in range(bsz):

            t_heatmap = true_heatmap[i].squeeze(0).numpy()  # h, w

            cl: int = cl_label[i]

            assert cl in list(self.per_cl_tracker.keys()), cl

            seg = true_seg[i].squeeze(0)  # h, w
            pred = pred_seg[i].squeeze(0)  # h, w
            assert seg.shape == pred.shape, f"{seg.shape} | {pred.shape}"
            assert pred.ndim == 2, pred.ndim

            th = threshold_otsu(pred, nbins=256)
            pred_bin = (pred >= th).astype(np.float32)

            if not np.isinf(t_heatmap[0, 0]):
                dice, iou = self._compute_dice_iou(seg, pred_bin)
                self.per_cl_tracker[cl][constants.DICE_MTR].append(dice)
                self.per_cl_tracker[cl][constants.IOU_MTR].append(iou)

            else:
                seg = None
                t_heatmap = None
                dice = None
                iou = None

            do_plot = self.plotit
            do_plot &= (self.plot_n_per_cl == -1) or (self.plot_n[cl] <
                                                      self.plot_n_per_cl)
            if do_plot:
                r_img = raw_img[i]  # 3, h, w
                r_img = r_img.permute(1, 2, 0).numpy()  # h, w, 3
                r_img = r_img.astype(np.uint8)

                true_masked_img = None

                if seg is not None:
                    true_masked_img = self.get_masked_img_black(r_img, seg)

                pred_masked_img = self.get_masked_img_black(r_img, pred_bin)

                im_id = image_id[i]
                wfp = join(self.fdout, f'{reformat_id(im_id)}.jpg')

                self.fast_plot_results(img=r_img,
                                       true_heatmap=t_heatmap,
                                       true_bin_mask=seg,
                                       true_cl=self.int_cl[cl],
                                       true_masked_img=true_masked_img,
                                       pred_heatmap=pred,
                                       pred_bin_mask=pred_bin,
                                       pred_cl=self.int_cl[cl_pred[i]],
                                       pred_masked_img=pred_masked_img,
                                       dice=dice,
                                       iou=iou,
                                       wfp=wfp
                                       )

                self.plot_n[cl] += 1

    def compress_visu(self):

        if not self.plotit:
            return 0

        fd = self.fdout.split(os.sep)[-1]
        if fd == '':
            fd = self.fdout.split(os.sep)[-2]
        cmdx = [
            "cd {} ".format(self.fdout),
            "cd .. ",
            "tar -cf {}.tar.gz {} ".format(fd, fd),
            "rm -r {} ".format(fd)
        ]
        cmdx = " && ".join(cmdx)
        DLLogger.log("Running: {}".format(cmdx))
        try:
            subprocess.run(cmdx, shell=True, check=True)
        except subprocess.SubprocessError as e:
            DLLogger.log("Failed to run: {}. Error: {}".format(cmdx, e))

    @staticmethod
    def _compute_dice_iou(seg1: np.ndarray,
                          seg2: np.ndarray
                          ) -> Tuple[float, float]:

        assert seg1.ndim == 2, seg1.ndim  # h, w
        assert seg1.shape == seg2.shape, f"{seg1.shape} | {seg2.shape}"

        seg1 = seg1.astype(bool)
        seg2 = seg2.astype(bool)

        ims_sum = float(seg1.sum() + seg2.sum())
        intersection = float(np.logical_and(seg1, seg2).sum())

        dice = 1.
        if ims_sum != 0.0:
            dice = 2. * intersection / ims_sum

        _union = ims_sum - intersection
        iou = 1.
        if _union != 0.0:
            iou = intersection / _union

        dice = dice * 100.
        iou = iou * 100.

        return dice, iou

    def fast_plot_results(self,
                          img: np.ndarray,
                          true_heatmap: np.ndarray,
                          true_bin_mask: np.ndarray,
                          true_cl: str,
                          true_masked_img: np.ndarray,
                          pred_heatmap: np.ndarray,
                          pred_bin_mask: np.ndarray,
                          pred_cl: str,
                          pred_masked_img: np.ndarray,
                          dice: float,
                          iou: float,
                          wfp: str
                          ):

        height, width = img.shape[:2]

        ncols = 5
        nrows = 2

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                                 figsize=(5, 2))
        fontsize = 3

        data = [
            [true_heatmap, true_bin_mask, true_cl, true_masked_img],
            [pred_heatmap, pred_bin_mask, pred_cl, pred_masked_img]
        ]

        if iou is not None:
            perf = f"IOU: {iou:.2f}%"
        else:
            perf = 'IOU: None'

        if dice is not None:
            perf = f"{perf}, Dice: {dice:.2f}%."
        else:
            perf = f"{perf}, Dice: None."


        for i in [0, 1]:
            tag = 'True' if i == 0 else 'Pred'

            heatmap, bin_mask, cl, masked_img = data[i]

            axes[i, 0].imshow(img)
            axes[i, 0].text(
                3, 40, f"{tag}: {cl}",
                fontsize=fontsize,
                bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                      'edgecolor': 'none'}
            )

            if heatmap is not None:
                axes[i, 1].imshow(img)
                axes[i, 1].imshow(heatmap, alpha=0.7)
                axes[i, 1].text(
                    3, 40, f"{tag}: {cl}",
                    fontsize=fontsize,
                    bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                          'edgecolor': 'none'}
                )

                axes[i, 2].imshow(heatmap, alpha=1.0)
                axes[i, 2].text(
                    3, 40, f"{tag}: {cl}",
                    fontsize=fontsize,
                    bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                          'edgecolor': 'none'}
                )

            if bin_mask is not None:
                axes[i, 3].imshow(bin_mask.astype(np.uint8) * 255, cmap='gray')
                axes[i, 3].text(
                    3, 40, f'{tag} ROI ({perf})',
                    fontsize=fontsize,
                    bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                          'edgecolor': 'none'}
                )

            if masked_img is not None:
                axes[i, 4].imshow(masked_img)
                axes[i, 4].text(
                    3, 40, f'{tag} masked image',
                    fontsize=fontsize,
                    bbox={'facecolor': 'white', 'pad': 1, 'alpha': 0.8,
                          'edgecolor': 'none'}
                )

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0,
                            wspace=0)
        for ax in fig.axes:
            ax.axis('off')
            ax.margins(0, 0)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        plt.savefig(wfp, pad_inches=0, bbox_inches='tight', dpi=300)
        plt.close()

    def get_masked_img_black(self,
                             img: np.ndarray,
                             roi: np.ndarray) -> np.ndarray:

        assert img.ndim == 3, img.ndim
        assert roi.ndim == 2, roi.ndim

        _roi = np.expand_dims(roi.copy(), axis=2)

        return (img * _roi).astype(img.dtype)

    def get_masked_img_avg(self,
                           img: np.ndarray,
                           roi: np.ndarray,
                           avg_train_pixel: np.ndarray
                           ) -> np.ndarray:

        assert img.ndim == 3, img.ndim
        assert avg_train_pixel.ndim == 3, avg_train_pixel.ndim  # 1, 1, 3.
        assert avg_train_pixel.shape == (1, 1, 3), avg_train_pixel.shape
        assert roi.ndim == 2, roi.ndim

        _roi = np.expand_dims(roi.copy(), axis=2)

        # avg = img.mean(0).mean(0).reshape((1, 1, 3))
        # avg = avg.clip(0, 255).astype(img.dtype)
        avg = avg_train_pixel.astype(img.dtype)
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


