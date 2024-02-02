# This module shouldn't import any of our modules to avoid recursive importing.
import os
from os.path import dirname, abspath
import sys
import argparse
import textwrap
from os.path import join
import fnmatch
from pathlib import Path
import subprocess

from sklearn.metrics import auc
import torch
import numpy as np
import munch
from pynvml.smi import nvidia_smi


root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)


CONST1 = 1000  # used to generate random numbers.


def str2bool(v):
    if isinstance(v, bool):
        return v

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def announce_msg(msg, upper=True, fileout=None):
    """
    Display sa message in the standard output. Something like this:
    =================================================================
                                message
    =================================================================

    :param msg: str, text message to display.
    :param upper: True/False, if True, the entire message is converted into
    uppercase. Else, the message is displayed
    as it is.
    :param fileout: file object, str, or None. if not None, we write the
    message in the file as well.
    :return: str, what was printed in the standard output.
    """
    if upper:
        msg = msg.upper()
    n = min(120, max(80, len(msg)))
    top = "\n" + "=" * n
    middle = " " * (int(n / 2) - int(len(msg) / 2)) + " {}".format(msg)
    bottom = "=" * n + "\n"

    output_msg = "\n".join([top, middle, bottom])

    # print to stdout
    print(output_msg)

    if fileout is not None:
        # print to file
        if isinstance(fileout, str):
            with open(fileout, "a") as fx:  # append
                print(output_msg + '\n', file=fx)
        elif hasattr(fileout, "write"):  # text file like.
            print(output_msg + '\n', file=fileout)
        else:
            raise NotImplementedError

    return output_msg


def fmsg(msg, upper=True):
    """
    Format message.
    :param msg:
    :param upper:
    :return:
    """
    if upper:
        msg = msg.upper()
    n = min(120, max(80, len(msg)))
    top = "\n" + "=" * n
    middle = " " * (int(n / 2) - int(len(msg) / 2)) + " {}".format(msg)
    bottom = "=" * n + "\n"

    output_msg = "\n".join([top, middle, bottom])
    return output_msg


def check_if_allow_multgpu_mode():
    """
    Check if we can do multigpu.
    If yes, allow multigpu.
    :return: ALLOW_MULTIGPUS: bool. If True, we enter multigpu mode:
    1. Computation will be dispatched over the AVAILABLE GPUs.
    2. Synch-BN is activated.
    """
    if "CC_CLUSTER" in os.environ.keys():
        ALLOW_MULTIGPUS = True  # CC.
    else:
        ALLOW_MULTIGPUS = False  # others.

    # ALLOW_MULTIGPUS = True
    os.environ["ALLOW_MULTIGPUS"] = str(ALLOW_MULTIGPUS)
    NBRGPUS = torch.cuda.device_count()
    ALLOW_MULTIGPUS = ALLOW_MULTIGPUS and (NBRGPUS > 1)

    return ALLOW_MULTIGPUS


def check_tensor_inf_nan(tn):
    """
    Check if a tensor has any inf or nan.
    """
    if any(torch.isinf(tn.view(-1))):
        raise ValueError("Found inf in projection.")
    if any(torch.isnan(tn.view(-1))):
        raise ValueError("Found nan in projection.")


def wrap_command_line(cmd):
    """
    Wrap command line
    :param cmd: str. command line with space as a separator.
    :return:
    """
    return " \\\n".join(textwrap.wrap(
        cmd, width=77, break_long_words=False, break_on_hyphens=False))


def find_files_pattern(fd_in_, pattern_):
    """
    Find paths to files with pattern within a folder recursively.
    :return:
    """
    msg = f"Folder {fd_in_} does not exist ... [NOT OK]"
    assert os.path.exists(fd_in_), msg

    print(f"Searching pattern '{pattern_}' @ {fd_in_} ...")

    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def check_nans(tens, msg=''):
    """
    Check if the tensor 'tens' contains any 'nan' values, and how many.

    :param tens: torch tensor.
    :param msg: str. message to display if there is nan.
    :return:
    """
    nbr_nans = torch.isnan(tens).float().sum().item()
    if nbr_nans > 0:
        print("NAN-CHECK: {}. Found: {} NANs.".format(msg, nbr_nans))


def compute_auc(vec, nbr_p):
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


def format_dict_2_str(obj: dict, initsp: str = '\t', seps: str = '\n\t'):
    """
    Convert dict into str.
    """
    assert isinstance(obj, dict)
    out = "{}".format(initsp)
    out += "{}".format(seps).join(
        ["{}: {}".format(k, obj[k]) for k in obj.keys()]
    )
    return out


def frmt_dict_mtr_str(obj: dict, dec_prec: int = 3, seps: str = " "):
    assert isinstance(obj, dict)
    return "{}".format(seps).join(
        ["{}: {}".format(k, "{0:.{1}f}".format(obj[k], dec_prec)) for k in
         obj.keys()])


def is_cc():
    return "CC_CLUSTER" in os.environ.keys()

def is_tay():
    if "HOST_XXX" in os.environ.keys():
        return os.environ['HOST_XXX'] == 'tay'
    return False

def is_gsys():
    if "HOST_XXX" in os.environ.keys():
        return os.environ['HOST_XXX'] == 'gsys'
    return False


def count_params(model: torch.nn.Module):
    return sum([p.numel() for p in model.parameters()])


def reformat_id(img_id):
    tmp = str(Path(img_id).with_suffix(''))
    return tmp.replace('/', '_')


def get_tag_device(args: object) -> str:
    tag = ''

    if torch.cuda.is_available():
        txt = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
        try:
            cudaids = args.cudaid.split(',')
            tag = 'CUDA devices: \n'
            for cid in cudaids:
                tag += 'ID: {} - {} \n'.format(cid, txt[int(cid)])
        except IndexError:
            tag = 'CUDA devices: lost.'

    return tag


def gpu_memory_stats(device: int) -> str:
    nvsmi = nvidia_smi.getInstance()
    nvsmi.DeviceQuery('memory.free, memory.total')

    assert isinstance(device, int)
    assert device >= 0

    z = nvsmi.DeviceQuery('memory.free, memory.total')['gpu'][device]
    used = z['fb_memory_usage']['total'] - z['fb_memory_usage']['free']

    unit = z['fb_memory_usage']['unit']
    norm = 1.
    if unit == 'MiB':
        unit = 'GB'
        norm = 1024

    msg = f'GPU {device} MEM: USED {used / norm}'
    msg += f" Free: {z['fb_memory_usage']['free'] / norm}"
    msg += f" Total: {z['fb_memory_usage']['total'] / norm}"
    msg += f' {unit}.'

    return msg


def move_state_dict_to_device(state_dict: dict, device):
    for k in state_dict:
        if torch.is_tensor(state_dict[k]):
            state_dict[k] = state_dict[k].to(device)

    return state_dict


# Data processing

def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = join(metadata_root, 'image_ids_proxy.txt')
    metadata.class_labels = join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = join(metadata_root, 'image_sizes.txt')
    metadata.localization = join(metadata_root, 'localization.txt')
    metadata.landmarks = join(metadata_root, 'landmarks.txt')
    return metadata


def get_image_ids(metadata, proxy=False):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata):
    """
    image_ids.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels

def get_landmarks(metadata):
    """
    The 68 facial landmarks text file has the following structure:
    <img-id>,x1,x2,...,x68,y1,y2,...,y68
    x: width.
    y: height.
    :param metadata:
    :return:
    """
    assert os.path.isfile(metadata.landmarks), metadata.landmarks

    n = 68 * 2
    landmarks = {}
    with open(metadata.landmarks, 'r') as f:
        for line in f.readlines():
            l = line.strip('\n').split(',')
            assert len(l) == n + 1, len(l)

            image_id = l[0]
            x = l[1:69]
            y = l[69:]
            x = [float(i) for i in x]
            y = [float(i) for i in y]
            coors: list = list(zip(x, y))

            assert image_id not in landmarks, image_id
            landmarks[image_id] = coors

    return landmarks


def get_bounding_boxes(metadata):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/to/image1.jpg,156,163,318,230
    path/to/image1.jpg,23,12,101,259
    path/to/image2.jpg,143,142,394,248
    path/to/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, x0s, y0s, x1s, y1s = line.strip('\n').split(',')
            x0, y0, x1, y1 = int(x0s), int(y0s), int(x1s), int(y1s)
            if image_id in boxes:
                boxes[image_id].append((x0, y0, x1, y1))
            else:
                boxes[image_id] = [(x0, y0, x1, y1)]
    return boxes


def get_mask_paths(metadata):
    """
    localization.txt (for masks) has the structure

    <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
    path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
    path/to/image1.jpg,path/to/mask1b.png,
    path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
    path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
    ...

    One image may contain multiple masks (multiple mask paths for same image).
    One image contains only one ignore mask.
    """
    mask_paths = {}
    ignore_paths = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths


def get_image_sizes(metadata):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/to/image1.jpg,500,300
    path/to/image2.jpg,1000,600
    path/to/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(metadata.image_sizes) as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes

# Per-class weight
def cl_w_tech1(w: list, n_cls: int) -> torch.Tensor:
    cpu = torch.device('cpu')
    w = torch.tensor(w, dtype=torch.float32, device=cpu,
                     requires_grad=False)

    w = (1. / w) * (w.sum() / (float(n_cls)))

    return w

def cl_w_tech2(w: list) -> torch.Tensor:
    cpu = torch.device('cpu')
    w = torch.tensor(w, dtype=torch.float32, device=cpu,
                     requires_grad=False)

    w = w.max() / w

    return w



# ==============================================================================
#                                            TEST
# ==============================================================================


def test_announce_msg():
    """
    Test announce_msg()
    :return:
    """
    announce_msg("Hello world!!!")
