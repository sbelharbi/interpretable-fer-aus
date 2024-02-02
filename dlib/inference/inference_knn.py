from copy import deepcopy
import os
import sys
from os.path import join, dirname, abspath, basename
import subprocess
from pathlib import Path
import datetime as dt
import argparse
import more_itertools as mit

import numpy as np
import pretrainedmodels.utils
import tqdm
import yaml
import munch
import pickle as pkl
from texttable import Texttable

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.utils.shared import find_files_pattern
from dlib.utils.shared import announce_msg
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
from dlib.knn_algo.knn import Knn

from dlib.utils.shared import is_gsys
from dlib.utils.shared import is_tay

from dlib.learning.inference_wsol import CAMComputer
from dlib.cams import build_std_cam_extractor
from dlib.utils.reproducibility import set_seed
from dlib.process.instantiators import get_model

from dlib.datasets.wsol_loader import get_data_loader
from dlib.learning.train_wsol import Basic
from dlib.losses.entropy import Entropy


_ENTROPY = 'entropy'
_PROBS = 'probs'
_LOGITS = 'logits'
_PRED = 'pred'
_Y = 'y'
_IDS = 'ids'

core_pattern = 'passed.txt'

_ENV_NAME = 'fer'

if is_gsys():
    virenv = "\nCONDA_BASE=$(conda info --base) \n" \
             "source $CONDA_BASE/etc/profile.d/conda.sh\n" \
             "conda activate {}\n".format(_ENV_NAME)

elif is_tay():
    virenv = f"\nsource /projets/AP92990/venvs" \
             f"/{_ENV_NAME}/bin/activate\n"
else:  # ???
    virenv = "\nCONDA_BASE=$(conda info --base) \n" \
             "source $CONDA_BASE/etc/profile.d/conda.sh\n" \
             "conda activate {}\n".format(_ENV_NAME)


PREAMBULE = "#!/usr/bin/env bash \n {}".format(virenv)
PREAMBULE += '\n# ' + '=' * 78 + '\n'
PREAMBULE += 'cudaid=$1\nexport CUDA_VISIBLE_DEVICES=$cudaid\n\n'

_IGNORE_METRICS_LOG = ['localization', 'top1', 'top5']

def mk_fd(fd):
    os.makedirs(fd, exist_ok=True)

def serialize_perf_meter(performance_meters, _EVAL_METRICS, _SPLITS) -> dict:
    return {
        split: {
            metric: vars(performance_meters[split][metric])
            for metric in _EVAL_METRICS
        }
        for split in _SPLITS
    }

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
                            int_to_cl: dict, title: str):
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

    plt.title(title, fontsize=7)
    # plt.tight_layout()
    plt.ylabel("True class", fontsize=7),
    plt.xlabel("Predicted class", fontsize=7)

    # disp.plot()
    plt.savefig(join(fdout, f'{name}.png'), bbox_inches='tight', dpi=300)
    plt.close('all')


def save_performances(outd, performance_meters, _EVAL_METRICS, _SPLITS,
                      int_to_cl,
                      epoch=None, checkpoint_type=None):
    tag = '' if checkpoint_type is None else '_{}'.format(checkpoint_type)

    tagargmax = ''

    log_path = join(outd, 'performance_log{}{}.pickle'.format(
        tag, tagargmax))
    with open(log_path, 'wb') as f:
        pkl.dump(
            serialize_perf_meter(performance_meters, _EVAL_METRICS, _SPLITS), f)

    log_path = join(outd, 'performance_log{}{}.txt'.format(
        tag, tagargmax))
    with open(log_path, 'w') as f:
        f.write("PERF - CHECKPOINT {}  - EPOCH {}  {} \n".format(
            checkpoint_type, epoch, tagargmax))

        for split in _SPLITS:
            for metric in _EVAL_METRICS:

                if metric.startswith(tuple(_IGNORE_METRICS_LOG)):
                    continue

                c_val = performance_meters[split][metric].current_value
                b_val = performance_meters[split][metric].best_value

                if (c_val is not None) and (
                        metric not in [constants.CL_CONFMTX_MTR,
                                       constants.AU_COSINE_MTR]
                ):
                    f.write(f"split: {split}. {metric}: {c_val} \n")
                    f.write(f"split: {split}. {metric}: {b_val}_best \n")
                elif (c_val is not None) and (
                        metric == constants.CL_CONFMTX_MTR):
                    f.write(f"BEST:{metric} \n\n")
                    f.write(f"{print_confusion_mtx(b_val, int_to_cl)} \n")

                    # plot confusion mtx.
                    fdout = join(outd, str(checkpoint_type),
                                 split)
                    plot_save_confusion_mtx(
                        mtx=b_val, fdout=fdout,
                        name=f'confusion-matrix-{split}',
                        int_to_cl=int_to_cl,
                        title='Classification confusion matrix'
                    )

                elif (c_val is not None) and (
                        metric == constants.AU_COSINE_MTR):
                    f.write(f"BEST:{metric} \n")


                    # plot cosine per (per_layer + cam) per expression.
                    fdout = join(outd, str(checkpoint_type),
                                 split)

def switch_key_val_dict(d: dict) -> dict:
    out = dict()
    for k in d:
        assert d[k] not in out, 'more than 1 key with same value. wrong.'
        out[d[k]] = k

    return out


def report(epoch, split, _EVAL_METRICS, _BEST_CRITERION_METRIC,
           performance_meters,
           checkpoint_type=None, show_epoch=True):
    tagargmax = ''

    if checkpoint_type is not None:
        DLLogger.log(fmsg('PERF - CHECKPOINT: {} {}'.format(
            checkpoint_type, tagargmax)))

    if show_epoch:
        DLLogger.log(f"EPOCH/{epoch}.")

    for metric in _EVAL_METRICS:

        if metric.startswith(tuple(_IGNORE_METRICS_LOG)):
            continue

        c_val = performance_meters[split][metric].current_value
        best_val = performance_meters[split][metric].best_value
        best_ep = performance_meters[split][metric].best_epoch

        if (c_val is not None) and (
                metric not in [constants.CL_CONFMTX_MTR,
                               constants.AU_COSINE_MTR]
        ):
            DLLogger.log(f"split: {split}. {metric}: {c_val}")
            cnd = (metric == _BEST_CRITERION_METRIC)
            cnd &= (split == constants.VALIDSET)
            if cnd:
                DLLogger.log(f"split: {split}. {metric}: "
                             f"{best_val} [BEST] [BEST-EPOCH: {best_ep}] ")

        elif (c_val is not None) and (
                metric in [constants.CL_CONFMTX_MTR,
                           constants.AU_COSINE_MTR]
        ):
            # todo. not necessary. overload logs.
            pass


def compute_cnf_mtx(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred_ = pred.detach().cpu().numpy()
    target_ = target.detach().cpu().numpy()
    conf_mtx = confusion_matrix(y_true=target_, y_pred=pred_,
                                sample_weight=None, normalize='true')

    return conf_mtx


def cl_forward(model, images, args):
    output = model(images)

    if args.task == constants.STD_CL:
        cl_logits = output

    elif args.task == constants.F_CL:
        cl_logits, fcams, im_recon = output

    else:
        raise NotImplementedError

    return cl_logits



def _compute_cl_perf(model, loader, split, args) -> dict:
    DLLogger.log(fmsg(f"Evaluate classification, split: {split}..."))
    t0 = dt.datetime.now()

    num_correct = 0
    num_images = 0
    all_pred = None
    all_y = None
    all_entropy = None
    all_prob = None
    all_logits = None
    all_ids = []
    entropy_fn = Entropy()

    for i, (images, targets, images_id, _, _, _, _) in tqdm.tqdm(
            enumerate(loader),  ncols=80, total=len(loader)):

        images = images.cuda(args.c_cudaid)
        targets = targets.cuda(args.c_cudaid)
        with torch.no_grad():
            cl_logits = cl_forward(model, images, args).detach()

            pred = cl_logits.argmax(dim=1)
            prob = F.softmax(cl_logits, dim=1)
            entropy = entropy_fn(prob)
            num_correct += (pred == targets).sum()
            num_images += images.size(0)

            if all_pred is None:
                all_pred = pred
                all_y = targets
                all_entropy = entropy
                all_prob = prob
                all_logits = cl_logits
                all_ids = list(images_id)

            else:
                all_pred = torch.cat((all_pred, pred))
                all_y = torch.cat((all_y, targets))
                all_entropy = torch.cat((all_entropy, entropy))
                all_prob = torch.cat((all_prob, prob))
                all_logits = torch.cat((all_logits, cl_logits))
                all_ids = all_ids + list(images_id)

    # sync

    conf_mtx = compute_cnf_mtx(pred=all_pred, target=all_y)

    classification_acc = ((all_pred == all_y).float().mean() * 100.).item()

    DLLogger.log(fmsg(f"Classification evaluation time for split: {split}, "
                      f"{dt.datetime.now() - t0}"))

    return {
        constants.CL_ACCURACY_MTR: classification_acc,
        constants.CL_CONFMTX_MTR: conf_mtx,
        _ENTROPY: all_entropy.cpu(),
        _PROBS: all_prob.cpu(),
        _LOGITS: all_logits.cpu(),
        _PRED: all_pred.cpu(),
        _Y: all_y.cpu(),
        _IDS: all_ids,
        "split": split
    }


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


def compute_cnf_mtx(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred_ = pred.detach().cpu().numpy()
    target_ = target.detach().cpu().numpy()
    conf_mtx = confusion_matrix(y_true=target_, y_pred=pred_,
                                sample_weight=None, normalize='true')

    return conf_mtx

def plot_pb_confusion_matrix(outd: str, results: dict, int_to_cl: dict):

    split = results['split']

    cls = sorted(list(int_to_cl.keys()), reverse=False)
    n = len(cls)
    logit_conf_mtx = torch.zeros((n, n), dtype=torch.float32,
                                 requires_grad=False)
    prob_conf_mtx = logit_conf_mtx * 0.0

    logits = results[_LOGITS]
    y = results[_Y]

    for cl in cls:
        _cl_logits = logits[y == cl]
        avg_logits = _cl_logits.mean(dim=0)
        logit_conf_mtx[cl] = avg_logits

        _cl_prob = F.softmax(_cl_logits, dim=1)
        avg_probs = _cl_prob.mean(dim=0)
        prob_conf_mtx[cl] = avg_probs

    logit_conf_mtx = logit_conf_mtx.numpy()
    prob_conf_mtx = prob_conf_mtx.numpy()

    DLLogger.log(f"Logit confusion matrix for split {split}")
    DLLogger.log(f"{print_confusion_mtx(logit_conf_mtx, int_to_cl)}")
    plot_save_confusion_mtx(
        mtx=logit_conf_mtx, fdout=outd,
        name=f'logits-confusion-matrix-{split}',
        int_to_cl=int_to_cl, title="Logits confusion matrix"
    )

    DLLogger.log(f"Probability confusion matrix for split {split}")
    DLLogger.log(f"{print_confusion_mtx(prob_conf_mtx, int_to_cl)}")
    plot_save_confusion_mtx(
        mtx=prob_conf_mtx, fdout=outd,
        name=f'prob-confusion-matrix-{split}',
        int_to_cl=int_to_cl, title="Probability confusion matrix"
    )

    cl_conf_mtx = compute_cnf_mtx(pred=results[_PRED], target=y)
    DLLogger.log(f"Classification confusion matrix for split {split}")
    DLLogger.log(f"{print_confusion_mtx(cl_conf_mtx, int_to_cl)}")
    plot_save_confusion_mtx(
        mtx=cl_conf_mtx, fdout=outd,
        name=f'classification-confusion-matrix-{split}',
        int_to_cl=int_to_cl, title="Classification confusion matrix"
    )

    DLLogger.flush()



def plot_entropy(outd: str, results: dict):
    entropy = results[_ENTROPY].numpy()
    align = results[_PRED].numpy() == results[_Y].numpy()

    fig, axes = plt.subplots(nrows=1, ncols=3, squeeze=False)
    p = 0.3

    w = np.ones_like(entropy) * (1. / entropy.size)
    n, bins, patches = axes[0, 0].hist(entropy, bins=1000, label='all samples',
                    weights=w)
    z = get_max_val(n, p)
    axes[0, 0].set_ylim([0, z])
    axes[0, 0].legend(fontsize="7", loc='best')


    if (align == 1).sum() > 0:
        v = entropy[align == 1]
        w = np.ones_like(v) * (1. / v.size)
        n, bins, patches = axes[0, 1].hist(v, bins=1000, label='Correct pred.',
                        weights=w)
        z = get_max_val(n, p)
        axes[0, 1].set_ylim([0, z])
        axes[0, 1].legend(fontsize="7", loc='best')

    if (align == 0).sum() > 0:
        v = entropy[align == 0]
        w = np.ones_like(v) * (1. / v.size)
        n, bins, patches = axes[0, 2].hist(v, bins=1000, label='Wrong pred.',
                        weights=w)
        z = get_max_val(n, p)
        axes[0, 2].set_ylim([0, z])
        axes[0, 2].legend(fontsize="7", loc='best')
        print(z)
    plt.suptitle(f"Entropy: split {results['split']}")
    fig.tight_layout()


    fig.savefig(join(outd, f"entropy-{results['split']}.png"),
                bbox_inches='tight', dpi=200)

def get_max_val(v: np.ndarray, p: float = 0.1):
    v.sort()
    a = v.size
    n = int(v.size * p)
    m = np.mean(v[a - n:])
    return m

def plot_probs(outd: str, results: dict):
    pass



def evaluate_knn():
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
    cudaid = parsedargs.cudaid

    assert os.path.isdir(exp_path)
    assert split in constants.SPLITS

    assert checkpoint_type in [constants.BEST, constants.LAST]

    eval_batch_size = 32

    _VERBOSE = True
    with open(join(exp_path, 'config_obj_final.yaml'), 'r') as fy:
        args: dict = yaml.safe_load(fy)
        args['c_cudaid'] = str(cudaid)
        args = Dict2Obj(args)
        args.outd = exp_path

    _CODE_FUNCTION = f'dense_dims_{args.dense_dims}_split_{split}'
    _DEFAULT_SEED = args.MYSEED
    os.environ['MYSEED'] = str(args.MYSEED)

    if checkpoint_type == constants.BEST:
        epoch = args.best_epoch
    elif checkpoint_type == constants.LAST:
        epoch = args.max_epochs
    else:
        raise NotImplementedError

    outd = join(exp_path, _CODE_FUNCTION)
    mk_fd(outd)
    tag = get_tag(args, checkpoint_type=checkpoint_type)
    tag += '_cams_{}'.format(split)

    msg = 'Task: {} \t box_v2_metric: {} \t' \
          'Dataset: {} \t Method: {} \t ' \
          'Encoder: {} \t'.format(args.task, args.box_v2_metric, args.dataset,
                                  args.method, args.model['encoder_name'])

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
    DLLogger.log(fmsg(f"Evaluate KNN, epoch {epoch}, split {split}"))

    set_seed(seed=_DEFAULT_SEED, verbose=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    log_device(parsedargs)
    model = get_model(args, eval=True)
    model.cuda()
    model.eval()

    basic_config = config.get_config(ds=args.dataset)
    args.data_paths = basic_config['data_paths']
    args.metadata_root = basic_config['metadata_root']
    args.mask_root = basic_config['mask_root']

    # standard evaluation. -----------------------------------------------------
    loaders, _ = get_data_loader(args=args,
                                 data_roots=basic_config['data_paths'],
                                 metadata_root=basic_config['metadata_root'],
                                 batch_size=args.batch_size,
                                 eval_batch_size=eval_batch_size,
                                 workers=args.num_workers,
                                 resize_size=args.resize_size,
                                 crop_size=args.crop_size,
                                 proxy_training_set=False,
                                 num_val_sample_per_class=0,
                                 std_cams_folder=None,
                                 get_splits_eval=[constants.TRAINSET],
                                 tr_bucket=None,
                                 curriculum_tr_ids=None,
                                 isdistributed=False
                                 )
    train_loader = loaders[constants.TRAINSET]

    loaders, _ = get_data_loader(args=args,
                                 data_roots=basic_config['data_paths'],
                                 metadata_root=basic_config['metadata_root'],
                                 batch_size=args.batch_size,
                                 eval_batch_size=eval_batch_size,
                                 workers=args.num_workers,
                                 resize_size=args.resize_size,
                                 crop_size=args.crop_size,
                                 proxy_training_set=False,
                                 num_val_sample_per_class=0,
                                 std_cams_folder=None,
                                 get_splits_eval=[split],
                                 tr_bucket=None,
                                 curriculum_tr_ids=None,
                                 isdistributed=False
                                 )
    eval_loader = loaders[split]

    folds_path = join(root_dir, args.metadata_root)
    path_class_id = join(folds_path, 'class_id.yaml')
    with open(path_class_id, 'r') as fcl:
        cl_int = yaml.safe_load(fcl)

    cl_to_int: dict = cl_int
    int_to_cl: dict = switch_key_val_dict(cl_int)


    # main performance
    performance_log = join(exp_path, f"performance_log_{checkpoint_type}.pickle")
    with open(performance_log, 'rb') as fx:
        stats = pkl.load(fx)

    main_acc = stats[split][constants.CL_ACCURACY_MTR]['value_per_epoch'][-1]

    # Knn.
    knn_model = Knn(k=1, cudaid=int(cudaid))
    knn_model.set_it(model=model, train_loader=train_loader)
    vl_fts, vl_lbls, vl_ids = knn_model.prepapre_eval_set(eval_loader)

    ks = list(range(1, 200, 1))
    ks = ks + list(range(200, 500, 10)) + [500]
    ks = list(range(1, 50, 1))

    tracker = {'main_acc': {constants.CL_ACCURACY_MTR: main_acc}}

    for k in tqdm.tqdm(ks, total=len(ks), ncols=80):
        DLLogger.log(fmsg(f"Evaluate KNN: k={k}, epoch {epoch}, split {split}"))

        subtrainer: Basic = Basic(args=args)
        subtrainer.inited = True
        performance_meters = subtrainer._set_performance_meters()

        knn_outd = join(outd, f"k-{k}")
        os.makedirs(knn_outd, exist_ok=True)

        knn_model.set_k(k=k)
        set_seed(seed=_DEFAULT_SEED, verbose=False)
        y_knn, _, _ = knn_model.evaluate_prepared_data(vl_fts, vl_lbls, vl_ids)


        # performance
        conf_mtx = compute_cnf_mtx(pred=y_knn, target=vl_lbls)
        classification_acc = ((y_knn == vl_lbls).float().mean() * 100.).item()

        cl_perf = {
            constants.CL_ACCURACY_MTR: classification_acc,
            constants.CL_CONFMTX_MTR: conf_mtx,
            _ENTROPY: None,
            _PROBS: None,
            _LOGITS: None,
            _PRED: y_knn.cpu(),
            _Y: vl_lbls.cpu(),
            _IDS: vl_ids,
            "split": split
        }

        knn_config = {'k': k,
                      constants.CL_ACCURACY_MTR: cl_perf[
                          constants.CL_ACCURACY_MTR],
                      'dense_dims': args.dense_dims
                      }

        with open(join(knn_outd, 'knn_config.yaml'), 'w') as fy:
            yaml.dump(knn_config, fy)

        tracker[k] = {constants.CL_ACCURACY_MTR: cl_perf[
                          constants.CL_ACCURACY_MTR],
                      'dense_dims': args.dense_dims
                      }

        with open(join(outd, 'knn_full_perf.yaml'), 'w') as fy:
            yaml.dump(tracker, fy)

        performance_meters[split][constants.CL_ACCURACY_MTR].update(
            cl_perf[constants.CL_ACCURACY_MTR]
        )
        performance_meters[split][constants.CL_CONFMTX_MTR].update(
            cl_perf[constants.CL_CONFMTX_MTR]
        )

        # plot
        cl_conf_mtx = compute_cnf_mtx(pred=cl_perf[_PRED], target=cl_perf[_Y])
        DLLogger.log(f"Classification confusion matrix for split {split}")
        DLLogger.log(f"{print_confusion_mtx(cl_conf_mtx, int_to_cl)}")
        plot_save_confusion_mtx(
            mtx=cl_conf_mtx, fdout=knn_outd,
            name=f'classification-confusion-matrix-{split}',
            int_to_cl=int_to_cl, title="Classification confusion matrix"
        )

        report(epoch,
               split=split,
               _EVAL_METRICS=subtrainer._EVAL_METRICS,
               _BEST_CRITERION_METRIC=subtrainer._BEST_CRITERION_METRIC,
               performance_meters=performance_meters,
               checkpoint_type=checkpoint_type
               )

        save_performances(outd=knn_outd,
                          performance_meters=performance_meters,
                          _EVAL_METRICS=subtrainer._EVAL_METRICS,
                          _SPLITS=subtrainer._SPLITS,
                          int_to_cl=int_to_cl,
                          epoch=epoch,
                          checkpoint_type=checkpoint_type
                          )

    DLLogger.log(fmsg('Time: {}'.format(dt.datetime.now() - t0)))
    plot_meter(tracker, outd, title=f"Split: {split}. "
                                    f"Dense_dims: {args.dense_dims}. "
                                    f"CL.ACC-wrt-KNN.")
    # end standard evaluation. -------------------------------------------------


def plot_meter(tracker: dict,
               outd: str,
               title: str = ''):

    vals = [(k, tracker[k][constants.CL_ACCURACY_MTR]) for k in tracker if
            k!='main_acc']

    vals.sort(key=lambda a: a[0])

    v = [item[1] for item in vals]
    ks = [item[0] for item in vals]
    max_v = max(v)
    j_max = v.index(max_v)
    main_acc = tracker['main_acc'][constants.CL_ACCURACY_MTR]

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False,
                             sharey=False, squeeze=False)


    x = list(range(len(v)))
    axes[0, 0].plot(x, v, color='tab:orange', label='KNN')
    axes[0, 0].plot(x, [main_acc for _ in x], color='tab:blue',
                    label='Model')
    # axes[0, 0].set_title('', fontsize=4)
    axes[0, 0].xaxis.set_tick_params(labelsize=4)
    axes[0, 0].yaxis.set_tick_params(labelsize=4)
    axes[0, 0].set_xlabel('k', fontsize=4)
    axes[0, 0].set_ylabel('CL. ACC.', fontsize=4)
    axes[0, 0].grid(True)
    axes[0, 0].legend(loc='best')
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))


    axes[0, 0].plot([x[j_max]],
                    [v[j_max]],
                    marker='o',
                    markersize=5,
                    color="red")

    plt.sca(axes[0, 0])
    plt.xticks(x, ks)

    fig.suptitle(title, fontsize=4)
    plt.tight_layout()

    os.makedirs(outd, exist_ok=True)
    fig.savefig(join(outd, 'acc-wrt-knn.png'), bbox_inches='tight', dpi=300)

def build_cmds():
    _NBRGPUS = 1
    # STD task.
    task = constants.STD_CL
    checkpoint = constants.BEST
    dataset = constants.RAFDB
    encoder = constants.APVIT
    search_dir = join(root_dir, constants.FULL_BEST_EXPS,
                      f"{dataset}-GMM", encoder, task)

    split = constants.TESTSET

    passed_files = find_files_pattern(fd_in_=search_dir, pattern_=core_pattern)
    # filter by task
    tmp_passed = []
    for pf in passed_files:
        exp_fd = dirname(pf)
        with open(join(exp_fd, 'config_obj_final.yaml'), 'r') as yl:
            args = yaml.safe_load(yl)

        cnd = True
        cnd &= (args['task'] == task)
        cnd &= (args['dataset'] == dataset)
        cnd &= (args['model']['encoder_name'] == encoder)

        if cnd:
            tmp_passed.append(pf)

    passed_files = tmp_passed
    splitted_files = [list(c) for c in mit.divide(_NBRGPUS, passed_files)]
    assert len(splitted_files) == _NBRGPUS

    for i in range(_NBRGPUS):
        _script_path = join(root_dir, 'knn_c_{}.sh'.format(i))
        script = open(_script_path, 'w')
        script.write(PREAMBULE)
        for file in splitted_files[i]:
            exp_dir = dirname(file)

            cmd = f'python dlib/inference/inference_knn.py ' \
                  f'--cudaid 0 ' \
                  f'--split {split} ' \
                  f'--checkpoint_type {checkpoint} ' \
                  f'--exp_path {exp_dir} ' \
                  f'\n'
            script.write(cmd)
            print(cmd)

        script.close()
        os.system('chmod +x {}'.format(_script_path))

    print('Passed files {}'.format(len(passed_files)))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        build_cmds()
    else:
        evaluate_knn()

