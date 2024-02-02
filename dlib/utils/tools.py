import sys
from os.path import dirname, abspath, join, basename, normpath
import os
import subprocess
import glob
import shutil
import subprocess
import datetime as dt
import math
from collections.abc import Iterable

import torch
import yaml
from sklearn.metrics import auc
import numpy as np

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib.dllogger as DLLogger

from dlib.utils.shared import fmsg
from dlib.configure import constants
from dlib.utils.shared import is_cc
from dlib.utils.shared import get_tag_device


def get_cpu_device():
    """
    Return CPU device.
    :return:
    """
    return torch.device("cpu")


def log_device(args):
    assert torch.cuda.is_available()

    tag = get_tag_device(args=args)

    DLLogger.log(message=tag)


def chunks_into_n(l: Iterable, n: int) -> Iterable:
    """
    Split iterable l into n chunks (iterables) with the same size.

    :param l: iterable.
    :param n: number of chunks.
    :return: iterable of length n.
    """
    chunksize = int(math.ceil(len(l) / n))
    return (l[i * chunksize:i * chunksize + chunksize] for i in range(n))


def chunk_it(l, n):
    """
    Create chunks with the same size (n) from the iterable l.
    :param l: iterable.
    :param n: int, size of the chunk.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


class Dict2Obj(object):
    """
    Convert a dictionary into a class where its attributes are the keys of
     the dictionary, and
    the values of the attributes are the values of the keys.
    """
    def __init__(self, dictionary):
        for key in dictionary.keys():
            setattr(self, key, dictionary[key])

    def __repr__(self):
        attrs = str([x for x in self.__dict__])
        return "<Dict2Obj: %s" % attrs


def count_nb_params(model):
    """
    Count the number of parameters within a model.

    :param model: nn.Module or None.
    :return: int, number of learnable parameters.
    """
    if model is None:
        return 0
    else:
        return sum([p.numel() for p in model.parameters()])


def create_folders_for_exp(exp_folder, name):
    """
    Create a set of folder for the current exp.
    :param exp_folder: str, the path to the current exp.
    :param name: str, name of the dataset (train, validation, test)
    :return: object, where each attribute is a folder.
    There is the following attributes:
        . folder: the name of the folder that will contain everything about
        this dataset.
        . prediction: for the image prediction.
    """
    l_dirs = dict()

    l_dirs["folder"] = join(exp_folder, name)
    l_dirs["prediction"] = join(exp_folder, "{}/prediction".format(name))

    for k in l_dirs:
        if not os.path.exists(l_dirs[k]):
            os.makedirs(l_dirs[k], exist_ok=True)

    return Dict2Obj(l_dirs)


def copy_code(dest,
              compress=False,
              verbose=False
              ):
    """Copy code to the exp folder for reproducibility.
    Input:
        dest: path to the destination folder (the exp folder).
        compress: bool. if true, we compress the destination folder and
        delete it.
        verbose: bool. if true, we show what is going on.
    """
    # extensions to copy.
    exts = tuple(["py", "sh", "yaml"])
    flds_files = ['.']

    for fld in flds_files:
        files = glob.iglob(os.path.join(root_dir, fld, "*"))
        subfd = join(dest, fld) if fld != "." else dest
        if not os.path.exists(subfd):
            os.makedirs(subfd, exist_ok=True)

        for file in files:
            if file.endswith(exts):
                if os.path.isfile(file):
                    shutil.copy(file, subfd)
    # cp dlib
    dirs = ["dlib", "cmds"]
    for dirx in dirs:
        cmds = [
            "cd {} && ".format(root_dir),
            "cp -r {} {} ".format(dirx, dest)
        ]
        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)

    if compress:
        head = dest.split(os.sep)[-1]
        if head == '':  # dest ends with '/'
            head = dest.split(os.sep)[-2]
        cmds = [
            "cd {} && ".format(dest),
            "cd .. && ",
            "tar -cf {}.tar.gz {}  && ".format(head, head),
            "rm -rf {}".format(head)
               ]

        cmd = "".join(cmds)
        if verbose:
            print("Running bash-cmds: \n{}".format(cmd.replace("&& ", "\n")))
        subprocess.run(cmd, shell=True, check=True)


def log_args(args_dict):
    DLLogger.log(fmsg("Configuration"))
    # todo


def save_model(model, args, outfd):
    model.eval()
    cpu_device = get_cpu_device()
    model.to(cpu_device)
    torch.save(model.state_dict(), join(outfd, "best_model.pt"))

    if args.task == constants.STD_CL:
        tag = "{}-{}-{}".format(
            args.dataset, args.model['encoder_name'], args.spatial_pooling)
        path = join(outfd, tag)
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        torch.save(model.encoder.state_dict(), join(path, 'encoder.pt'))
        torch.save(model.classification_head.state_dict(),
                   join(path, 'head.pt'))
        DLLogger.log(message="Stored classifier. TAG: {}".format(tag))


def save_config(config_dict, outfd):
    with open(join(outfd, 'config.yaml'), 'w') as fout:
        yaml.dump(config_dict, fout)


def get_best_epoch(fyaml):
    with open(fyaml, 'r') as f:
        config = yaml.safe_load(f)
        return config['best_epoch']


def compute_auc(vec: np.ndarray, nbr_p: int):
    """
    Compute the area under a curve.
    :param vec: vector contains values in [0, 100.].
    :param nbr_p: int. number of points in the x-axis. it is expected to be
    the same as the number of values in `vec`.
    :return: float in [0, 100]. percentage of the area from the perfect area.
    """
    if vec.size == 1:
        return float(vec[0])
    else:
        area_under_c = auc(x=np.array(list(range(vec.size))), y=vec)
        area_under_c /= (100. * (nbr_p - 1))
        area_under_c *= 100.  # (%)
        return area_under_c


# WSOL

def check_box_convention(boxes, convention, tolerate_neg=False):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
        tolerate_neg: bool. if true, we dont mind negative values.
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if not tolerate_neg:
        if (boxes < 0).any():
            raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError(f"Box array must have dimension (4) or "
                           f"(num_boxes, 4): {len(boxes.shape)}.")

    if boxes.shape[1] != 4:
        raise RuntimeError(f"Box array must have dimension (4) or "
                           f"(num_boxes, 4): {boxes.shape[1]}.")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))

def t2n(t):
    return t.detach().cpu().numpy().astype(float)


def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != float:
        raise TypeError("Scoremap must be of np.float type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def get_tag(args: object, checkpoint_type: str = None) -> str:
    if checkpoint_type is None:
        checkpoint_type = args.eval_checkpoint_type
    tag = "{}-{}-{}-{}-cp_{}-boxv2_{}".format(
        args.dataset, args.model['encoder_name'], args.method,
        args.spatial_pooling, checkpoint_type,
        args.box_v2_metric)

    return tag

def get_heatmap_tag(args: object, key: str) -> str:

    assert key in constants.HEATMAP_KEYS, key

    if key == constants.ALIGN_ATTEN_HEATMAP:

        type_heatmap = args.align_atten_to_heatmap_type_heatmap
        lndmk_variance = args.align_atten_to_heatmap_lndmk_variance
        jaw = args.align_atten_to_heatmap_jaw
        normalize = args.align_atten_to_heatmap_normalize
        aus_seg_full = args.align_atten_to_heatmap_aus_seg_full

    elif key == constants.TRAIN_HEATMAP:

        type_heatmap = args.train_daug_mask_img_heatmap_type
        lndmk_variance = args.train_daug_mask_img_heatmap_lndmk_variance
        jaw = args.train_daug_mask_img_heatmap_jaw
        normalize = args.train_daug_mask_img_heatmap_normalize
        aus_seg_full = args.train_daug_mask_img_heatmap_aus_seg_full

    elif key  == constants.EVAL_HEATMAP:

        type_heatmap = args.eval_daug_mask_img_heatmap_type
        lndmk_variance = args.eval_daug_mask_img_heatmap_lndmk_variance
        jaw = args.eval_daug_mask_img_heatmap_jaw
        normalize = args.eval_daug_mask_img_heatmap_normalize
        aus_seg_full = args.eval_daug_mask_img_heatmap_aus_seg_full

    elif key  == constants.AUS_SEGM:

        type_heatmap = args.aus_seg_heatmap_type
        lndmk_variance = args.aus_seg_lndmk_variance
        jaw = args.aus_seg_jaw
        normalize = args.aus_seg_normalize
        aus_seg_full = args.aus_seg_aus_seg_full


    else:
        raise NotImplementedError(key)


    tag = f"{args.dataset}-{type_heatmap}"

    if type_heatmap == constants.HEATMAP_LNDMKS:
        tag = f"{tag}-{normalize}-{lndmk_variance}-{jaw}"

    elif type_heatmap in [constants.HEATMAP_AUNITS_LNMKS,
                          constants.HEATMAP_GENERIC_AUNITS_LNMKS,
                          constants.HEATMAP_PER_CLASS_AUNITS_LNMKS
                          ]:
        tag = f"{tag}-{normalize}"

    elif type_heatmap == constants.HEATMAP_AUNITS_LEARNED_SEG:
        tag = f"{tag}-{normalize}-{aus_seg_full}"

    else:
        raise NotImplementedError(type_heatmap)

    return tag

def bye(args):
    DLLogger.log(fmsg("End time: {}".format(args.tend)))
    DLLogger.log(fmsg("Total time: {}".format(args.tend - args.t0)))

    with open(join(root_dir, 'LOG.txt'), 'a') as f:
        m = "{}: \t " \
            "Dataset: {} \t " \
            "Method: {} \t " \
            "Spatial pooling: {} \t " \
            "Encoder: {} \t " \
            "Check point: {} \t " \
            "Box_v2_metric: {} \t " \
            "SL: {} \t " \
            "CRF: {} \t " \
            "... Passed in [{}]. \n".format(
                dt.datetime.now(),
                args.dataset,
                args.method,
                args.spatial_pooling,
                args.model['encoder_name'],
                args.eval_checkpoint_type,
                args.box_v2_metric,
                args.sl_fc,
                args.crf_fc,
                args.tend - args.t0
            )
        f.write(m)

    with open(join(args.outd, 'passed.txt'), 'w') as fout:
        fout.write('Passed.')

    DLLogger.log(fmsg('bye.'))

    # clean cc
    if is_cc():
        scratch_exp_fd = join(os.environ["SCRATCH"], constants.SCRATCH_FOLDER,
                              args.subpath)
        scratch_tmp = dirname(normpath(scratch_exp_fd))  # parent
        _tag = basename(normpath(args.outd))
        cmdx = [
            "cd {} ".format(args.outd),
            "cd .. ",
            "tar -cf {}.tar.gz {}".format(_tag, _tag),
            'cp {}.tar.gz {}'.format(_tag, scratch_tmp),
            'cd {}'.format(scratch_tmp),
            'tar -xf {}.tar.gz -C {} --strip-components=1'.format(
                _tag, basename(normpath(scratch_exp_fd))),
            "rm {}.tar.gz".format(_tag)
        ]
        cmdx = " && ".join(cmdx)
        print("Running bash-cmds: \n{}".format(cmdx.replace("&& ", "\n")))
        subprocess.run(cmdx, shell=True, check=True)

def get_root_wsol_dataset():
    baseurl = None
    if "HOST_XXX" in os.environ.keys():
        if os.environ['HOST_XXX'] == 'laptop':
            baseurl = f"{os.environ['EXDRIVE']}/datasets"
        elif os.environ['HOST_XXX'] == 'lab':
            baseurl = f"{os.environ['DATASETSH']}/wsol-done-right"
        elif os.environ['HOST_XXX'] == 'gsys':
            baseurl = f"{os.environ['DATASETSH']}/wsol-done-right"
        elif os.environ['HOST_XXX'] == 'tay':
            baseurl = f"{os.environ['DATASETSH']}/wsol-done-right"
        elif os.environ['HOST_XXX'] == 'ESON':
            baseurl = f"{os.environ['DATASETSH']}/datasets"
        else:
            raise NotImplementedError(os.environ['HOST_XXX'])

    elif "CC_CLUSTER" in os.environ.keys():
        if "SLURM_TMPDIR" in os.environ.keys():
            # if we are running within a job use the node disc:  $SLURM_TMPDIR
            baseurl = "{}/datasets/wsol-done-right".format(
                os.environ["SLURM_TMPDIR"])
        else:
            # if we are not running within a job, use the scratch.
            # this cate my happen if someone calls this function outside a job.
            baseurl = "{}/datasets/wsol-done-right".format(os.environ["SCRATCH"])

    msg_unknown_host = "Sorry, it seems we are unable to recognize the " \
                       "host. You seem to be new to this code. " \
                       "We recommend you to add your baseurl on your own."
    if baseurl is None:
        raise ValueError(msg_unknown_host)

    return baseurl


def build_heatmap_folder(args: dict,
                         key: str,
                         force_it: bool = False
                         ) -> str:
    """
    Operates for:
    - train_daug_mask_img_heatmap
    - eval_daug_mask_img_heatmap
    - align_atten_to_heatmap
    - aus_seg
    :param args:
    :return:
    """

    assert key in constants.HEATMAP_KEYS, key

    pre_computed = args[constants.PRECOMPUTED[key]]
    folder = args[constants.FOLDER_HEATMAP[key]]

    _cnd_ = pre_computed
    _cnd_ &= ((folder == '') or force_it)

    if _cnd_:
        tag = get_heatmap_tag(Dict2Obj(args), key=key)

        if is_cc():
            baseurl_sc = "{}/datasets/wsol-done-right".format(
                os.environ["SCRATCH"])
            scratch_path = join(baseurl_sc, '{}.tar.gz'.format(tag))

            if os.path.isfile(scratch_path):
                slurm_dir = get_root_wsol_dataset()
                cmds = [
                    'cp {} {} '.format(scratch_path, slurm_dir),
                    'cd {} '.format(slurm_dir),
                    'tar -xf {}'.format('{}.tar.gz'.format(tag))
                ]
                cmdx = " && ".join(cmds)
                print("Running bash-cmds: \n{}".format(
                    cmdx.replace("&& ", "\n")))
                subprocess.run(cmdx, shell=True, check=True)

                assert os.path.isdir(join(slurm_dir, tag))
                path_heatmaps = join(slurm_dir, tag)

            else:
                raise ValueError(f"{scratch_path} not found.")
        else:
            baseurl = get_root_wsol_dataset()
            path_heatmaps = join(baseurl, tag)

        assert os.path.isdir(path_heatmaps), path_heatmaps
        return path_heatmaps

    else:
        return folder

if __name__ == '__main__':
    print(root_dir)
