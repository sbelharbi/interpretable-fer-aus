from copy import deepcopy
import os
import sys
from os.path import join, dirname, abspath, basename
import datetime as dt
import argparse

import numpy as np
import yaml
import munch
from texttable import Texttable
from tqdm import tqdm as tqdm

import torch
import matplotlib.pyplot as plt

import seaborn as sns

root_dir = dirname(abspath(__file__))
sys.path.append(root_dir)

from  dlib.configure import constants

from dlib.dllogger import ArbJSONStreamBackend
from dlib.dllogger import Verbosity
from dlib.dllogger import ArbStdOutBackend
from dlib.dllogger import ArbTextStreamBackend
import dlib.dllogger as DLLogger
from dlib.utils.shared import fmsg
from dlib.utils.tools import get_tag
from dlib.utils.tools import Dict2Obj
from dlib.utils.tools import log_device
from dlib.configure import config
from dlib.utils.tools import get_cpu_device


from dlib.learning.inference_wsol import CAMComputer
from dlib.utils.reproducibility import set_seed
from dlib.process.instantiators import get_model

from dlib.datasets.wsol_loader import get_data_loader
from dlib.learning.train_wsol import Basic
from dlib.learning.train_wsol import compute_cnf_mtx


def load_weights(args, model, path: str):
    assert args.task == constants.STD_CL, args.task
    all_w = torch.load(path, map_location=get_cpu_device())

    if args.method in [constants.METHOD_TSCAM,
                       constants.METHOD_APVIT]:
        model.load_state_dict(all_w, strict=True)

    else:
        encoder_w = all_w['encoder']
        classification_head_w = all_w['classification_head']

        model.encoder.super_load_state_dict(encoder_w, strict=True)
        model.classification_head.load_state_dict(
            classification_head_w, strict=True)

    DLLogger.log(f"Loaded weights from {path}.")
    return model


def cl_forward(model, images, targets, task) -> torch.Tensor:

    output = model(images, targets)

    if task == constants.STD_CL:
        cl_logits = output

    elif task == constants.F_CL:
        cl_logits, fcams, im_recon = output

    else:
        raise NotImplementedError

    return cl_logits


def classify(model,
             loader,
             split: str,
             args
             ) -> dict:

    DLLogger.log(fmsg(f"Evaluate classification, split: {split}..."))

    t0 = dt.datetime.now()

    num_correct = 0
    num_images = 0
    all_pred = None
    all_y = None

    for i, (images, _, targets, image_ids, raw_imgs, _, _, _, _,
            _, _) in tqdm(enumerate(loader),
                                                  total=len(loader), ncols=80):

        images = images.cuda()
        targets = targets.cuda()
        with torch.no_grad():
            cl_logits = cl_forward(model, images, targets, args.task).detach()

            pred = cl_logits.argmax(dim=1)
            num_correct += (pred == targets).sum()
            num_images += images.size(0)

            if all_pred is None:
                all_pred = pred
                all_y = targets

            else:
                all_pred = torch.cat((all_pred, pred))
                all_y = torch.cat((all_y, targets))

    conf_mtx = compute_cnf_mtx(pred=all_pred, target=all_y)

    classification_acc = ((all_pred == all_y).float().mean() * 100.).item()
    diff = (all_pred - all_y).float()
    mse = ((diff ** 2).mean()).item()
    mae = (diff.abs().mean()).item()

    DLLogger.log(fmsg(f"Classification evaluation time for split: {split}, "
                      f"{dt.datetime.now() - t0}"))

    out = {
        constants.CL_ACCURACY_MTR: classification_acc,
        constants.MSE_MTR: mse,
        constants.MAE_MTR: mae,
        constants.CL_CONFMTX_MTR: conf_mtx
    }
    return out


def print_confusion_mtx(cmtx: np.ndarray, int_to_cl: dict) -> str:
    header_type = ['t']
    keys = list(int_to_cl.keys())
    h, w = cmtx.shape
    assert len(keys) == h, f"{len(keys)} {h}"
    assert len(keys) == w, f"{len(keys)} {w}"

    keys = sorted(keys, reverse=False)
    t = Texttable()
    t.set_max_width(400)
    header = ['*']
    for k in keys:
        header_type.append('f')
        header.append(int_to_cl[k])

    t.header(header)
    t.set_cols_dtype(header_type)
    t.set_precision(6)

    for i in range(h):
        row = [int_to_cl[i]]
        for j in range(w):
            row.append(cmtx[i, j])

        t.add_row(row)

    return t.draw()


def plot_save_confusion_mtx(mtx: np.ndarray, fdout: str, name: str,
                            int_to_cl: dict):
    if not os.path.isdir(fdout):
        os.makedirs(fdout, exist_ok=True)

    keys = list(int_to_cl.keys())
    h, w = mtx.shape
    assert len(keys) == h, f"{len(keys)} {h}"
    assert len(keys) == w, f"{len(keys)} {w}"

    keys = sorted(keys, reverse=False)
    cls = [int_to_cl[k] for k in keys]

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


def print_avg_per_cl_au_cosine(mtx: np.ndarray, int_to_cl: dict) -> str:
    ord_int_cls = sorted(list(int_to_cl.keys()), reverse=False)
    ord_str_cls = [int_to_cl[x] for x in ord_int_cls]

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


def plot_save_avg_per_cl_au_cosine(mtx: np.ndarray,
                                   fdout: str,
                                   name: str,
                                   int_to_cl: dict,
                                   method: str
                                   ):
    if not os.path.isdir(fdout):
        os.makedirs(fdout, exist_ok=True)

    ord_int_cls = sorted(list(int_to_cl.keys()), reverse=False)
    ord_str_cls = [int_to_cl[x] for x in ord_int_cls]

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
    n_layers = len(labels_x)  # todo: change to w.
    if method == constants.METHOD_APVIT:
        labels_x.append(f"layer-{n_layers + 1}")

    else:
        labels_x.append("CAM")

    labels_y = ord_str_cls + ['Average']

    plt.close('all')
    # todo: set vmin, vmax to be to compare 2 figures.
    g = sns.heatmap(mtx, annot=True, cmap='Greens',
                    xticklabels=1, yticklabels=1, fmt='.4f')
    g.set_xticklabels(labels_x, fontsize=7)
    g.set_yticklabels(labels_y, rotation=0, fontsize=7)

    plt.title("Cosine similarity with action units heatmap", fontsize=7)

    # disp.plot()
    plt.savefig(join(fdout, f'{name}.png'), bbox_inches='tight', dpi=300)
    plt.close('all')


def switch_key_val_dict(d: dict) -> dict:
    out = dict()
    for k in d:
        assert d[k] not in out, 'more than 1 key with same value. wrong.'
        out[d[k]] = k

    return out


def evaluate():
    t0 = dt.datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--cudaid", type=str, default=None, help="cuda id.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--checkpoint_type", type=str, default=None)
    parser.add_argument("--exp_path", type=str, default=None)

    parsedargs = parser.parse_args()
    split = parsedargs.split
    exp_path = parsedargs.exp_path
    checkpoint_type = parsedargs.checkpoint_type

    assert os.path.isdir(exp_path)
    assert split in constants.SPLITS

    assert checkpoint_type in [constants.BEST, constants.LAST]

    _CODE_FUNCTION = 'eval_loc_{}'.format(split)

    _VERBOSE = True
    with open(join(exp_path, 'config_model.yaml'), 'r') as fy:
        args = yaml.safe_load(fy)

        args_dict = deepcopy(args)

        org_align_atten_to_heatmap = args['align_atten_to_heatmap']
        if not args['align_atten_to_heatmap']:

            args['align_atten_to_heatmap'] = True

            args['align_atten_to_heatmap_type_heatmap'] = constants.HEATMAP_AUNITS_LNMKS
            args['align_atten_to_heatmap_normalize'] = True
            args['align_atten_to_heatmap_jaw'] = False
            args['align_atten_to_heatmap_lndmk_variance'] = 64.
            args['align_atten_to_heatmap_aus_seg_full'] = True

            if args['method'] in [constants.METHOD_APVIT, constants.METHOD_TSCAM]:
                args['align_atten_to_heatmap_layers'] = '1'

            elif args['method'] == constants.METHOD_CUTMIX:
                args['align_atten_to_heatmap_layers'] = '4'

            else:
                args['align_atten_to_heatmap_layers'] = '5'

            args['align_atten_to_heatmap_align_type'] = constants.ALIGN_AVG
            args['align_atten_to_heatmap_norm_att'] = constants.NORM_NONE
            args['align_atten_to_heatmap_p'] = 1.
            args['align_atten_to_heatmap_q'] = 1.
            args['align_atten_to_heatmap_loss'] = constants.A_COSINE
            args['align_atten_to_heatmap_elb'] = False
            args['align_atten_to_heatmap_lambda'] = 9.
            args['align_atten_to_heatmap_scale_to'] = constants.SCALE_TO_ATTEN

            args['align_atten_to_heatmap_use_self_atten'] = False

            if args['dataset'] == constants.RAFDB:
                args['align_atten_to_heatmap_use_precomputed'] = False
                args['align_atten_to_heatmap_folder'] = ''

            elif args['dataset'] == constants.AFFECTNET:
                args['align_atten_to_heatmap_use_precomputed'] = False
                args['align_atten_to_heatmap_folder'] = ''

            else:
                raise NotImplementedError

        else:
            if args['dataset'] == constants.RAFDB:
                args['align_atten_to_heatmap_use_precomputed'] = False
                args['align_atten_to_heatmap_folder'] = ''

            elif args['dataset'] == constants.AFFECTNET:
                args['align_atten_to_heatmap_use_precomputed'] = False
                args['align_atten_to_heatmap_folder'] = ''

            else:
                raise NotImplementedError

        args['distributed'] = False
        args = Dict2Obj(args)

        args.outd = exp_path

        args.eval_checkpoint_type = checkpoint_type

    _DEFAULT_SEED = args.MYSEED
    os.environ['MYSEED'] = str(args.MYSEED)

    if checkpoint_type == constants.BEST:
        epoch = args.best_epoch
    elif checkpoint_type == constants.LAST:
        epoch = args.max_epochs
    else:
        raise NotImplementedError

    outd = join(exp_path, _CODE_FUNCTION,
                f'split_{split}-checkpoint_{checkpoint_type}')

    os.makedirs(outd, exist_ok=True)

    msg = f'Task: {args.task}. Checkpoint {checkpoint_type} \t ' \
          f'Dataset: {args.dataset} \t Method: {args.method} \t ' \
          f'Encoder: {args.model["encoder_name"]} \t'

    log_backends = [
        ArbJSONStreamBackend(Verbosity.VERBOSE,
                             join(outd, "log.json")),
        ArbTextStreamBackend(Verbosity.VERBOSE,
                             join(outd, "log.txt")),
    ]

    if _VERBOSE:
        log_backends.append(ArbStdOutBackend(Verbosity.VERBOSE))
    DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())
    DLLogger.log(fmsg("Start time: {}".format(t0)))
    DLLogger.log(fmsg(msg))
    DLLogger.log(fmsg("Evaluate epoch {}, split {}".format(epoch, split)))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    tag = get_tag(args, checkpoint_type=checkpoint_type)

    model_weights_path = join(exp_path, 'model.pt')

    log_device(parsedargs)
    torch.cuda.set_device(0)
    model = get_model(args, eval=False)
    model = load_weights(args, model,  model_weights_path)
    model.cuda()
    model.eval()

    args.outd = outd  # must be after: get_model(args, eval=True)

    basic_config = config.get_config(ds=args.dataset)
    args.data_root = basic_config['data_root']
    args.data_paths = basic_config['data_paths']
    args.metadata_root = basic_config['metadata_root']
    args.mask_root = basic_config['mask_root']

    loaders, _ = get_data_loader(
        args=args,
        data_roots=basic_config['data_paths'],
        metadata_root=basic_config['metadata_root'],
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        workers=args.num_workers,
        resize_size=args.resize_size,
        crop_size=args.crop_size,
        proxy_training_set=args.proxy_training_set,
        num_val_sample_per_class=args.num_val_sample_per_class,
        std_cams_folder=None,
        get_splits_eval=[split],
        isdistributed=False
    )

    subtrainer: Basic = Basic(args=args)
    performance_meters = subtrainer._set_performance_meters()

    folds_path = join(root_dir, args.metadata_root)
    path_class_id = join(folds_path, 'class_id.yaml')
    with open(path_class_id, 'r') as fcl:
        cl_int = yaml.safe_load(fcl)

    cl_to_int: dict = cl_int
    int_to_cl: dict = switch_key_val_dict(cl_int)

    # eval classification
    out_cl = classify(model, loaders[split], split, args)
    log_cl_path = join(args.outd, f'eval_cl_perf_log_{tag}.txt')
    metric = constants.CL_CONFMTX_MTR
    with open(log_cl_path, 'w') as f:
        f.write(f"BEST:{metric} \n")
        f.write(f"{print_confusion_mtx(out_cl[constants.CL_CONFMTX_MTR],int_to_cl)}"
                f" \n")
        msg = f"{constants.CL_ACCURACY_MTR}: " \
              f"{out_cl[constants.CL_ACCURACY_MTR]:.4f}%."
        f.write(f"{msg}\n")
        DLLogger.log(msg)

    fdout = join(args.outd, str(checkpoint_type), split)
    plot_save_confusion_mtx(out_cl[constants.CL_CONFMTX_MTR],
                            fdout,
                            f'cl-conf-matrix-{split}',
                            int_to_cl)

    # eval localization
    cam_computer = CAMComputer(
        args=deepcopy(args),
        model=model,
        loader=loaders[split],
        metadata_root=os.path.join(args.metadata_root, split),
        mask_root=args.mask_root,
        iou_threshold_list=args.iou_threshold_list,
        dataset_name=args.dataset,
        split=split,
        cam_curve_interval=args.cam_curve_interval,
        multi_contour_eval=args.multi_contour_eval,
        out_folder=outd
    )
    cam_performance = cam_computer.compute_and_evaluate_cams()

    DLLogger.log(fmsg(f"CAM EVALUATE TIME of split {split} :"
                      f" {dt.datetime.now() - t0}"))

    cosine_au = cam_computer.get_matrix_avg_per_cl_au_cosine()

    nbr_samples_to_plot = 1000
    cam_computer.draw_some_best_pred(cl_int=cl_int, nbr=nbr_samples_to_plot,
                                     less_visual=True)
    cam_computer.plot_avg_cams_per_cl()
    cam_computer.plot_avg_aus_maps()
    cam_computer.plot_avg_att_maps()

    # plot/ print
    metric = constants.AU_COSINE_MTR
    tagargmax = ''
    log_path = join(args.outd, 'eval_loc_perf_log{}{}.txt'.format(
        tag, tagargmax))
    with open(log_path, 'w') as f:
        align_layer = args.align_atten_to_heatmap_layers
        if org_align_atten_to_heatmap:
            f.write(f"BEST:{metric} / "
                    f"align with AUs at layer: {align_layer}\n")
        else:
            f.write(f"BEST:{metric} / No align with AUs. \n")
        f.write(f"{print_avg_per_cl_au_cosine(cosine_au, int_to_cl)} \n")

    # plot cosine per (per_layer + cam) per expression.
    fdout = join(args.outd, str(checkpoint_type), split)
    plot_save_avg_per_cl_au_cosine(
        mtx=cosine_au,
        fdout=fdout,
        name=f'action-unit-cosine-{split}',
        int_to_cl=int_to_cl,
        method=args.method
    )

    DLLogger.log(fmsg("Bye."))


if __name__ == '__main__':
    evaluate()

