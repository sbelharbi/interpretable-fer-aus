import os
from os.path import join
import sys
from os.path import dirname
from os.path import abspath
import fnmatch

import matplotlib.pyplot as plt
from PIL import Image
import tqdm
import csv

sys.path.append(dirname(dirname(abspath(__file__))))

from dlib.utils.tools import get_root_wsol_dataset

from dlib.utils.shared import announce_msg


__all__ = [
    "show_msg",
    "get_stats"
]


def find_files_pattern(fd_in_, pattern_):
    """
    Find paths to files with pattern within a folder recursively.
    :return:
    """
    assert os.path.exists(fd_in_), "Folder {} does not exist " \
                                   ".... [NOT OK]".format(fd_in_)
    files = []
    for r, d, f in os.walk(fd_in_):
        for file in f:
            if fnmatch.fnmatch(file, pattern_):
                files.append(os.path.join(r, file))

    return files


def show_msg(ms, lg):
    announce_msg(ms)
    lg.write(ms + "\n")


def drop_normal_samples(l_samples):
    """
    Remove normal samples from the list of samples.

    When to call this?
    # drop normal samples and keep metastatic if: 1. dataset=CAM16. 2.
    # al_type != AL_WSL.

    :param l_samples: list of samples resulting from csv_loader().
    :return: l_samples without any normal sample.
    """
    return [el for el in l_samples if el[3] == 'tumor']


def csv_loader(fname, rootpath, drop_normal=False):
    """
    Read a *.csv file. Each line contains:
     0. id_: str
     1. img: str
     2. mask: str or '' or None
     3. label: str
     4. tag: int in {0, 1}

     Example: 50162.0, test/img_50162_label_frog.jpeg, , frog, 0

    :param fname: Path to the *.csv file.
    :param rootpath: The root path to the folders of the images.
    :return: List of elements.
    :param drop_normal: bool. if true, normal samples are dropped.
    Each element is the path to an image: image path, mask path [optional],
    class name.
    """
    with open(fname, 'r') as f:
        out = [
            [row[0],
             join(rootpath, row[1]),
             join(rootpath, row[2]) if row[2] else None,
             row[3],
             int(row[4])
             ]
            for row in csv.reader(f)
        ]

    if drop_normal:
        out = drop_normal_samples(out)

    return out

def get_stats(args, split, fold, subset):
    """
    Get some stats on the image sizes of specific dataset, split, fold.
    """
    if not os.path.isdir(args.fold_folder):
        os.makedirs(args.fold_folder, exist_ok=True)

    tag = "ds-{}-s-{}-f-{}-subset-{}".format(args.dataset,
                                             split,
                                             fold,
                                             subset
                                             )
    log = open(join(
        args.fold_folder, "log-stats-ds-{}.txt".format(tag)), 'w')
    announce_msg("Going to check {}".format(args.dataset.upper()))

    relative_fold_path = join(args.fold_folder,
                              "split_{}".format(split),
                              "fold_{}".format(fold)
                              )

    subset_csv = join(relative_fold_path,
                      "{}_s_{}_f_{}.csv".format(subset, split, fold)
                      )
    rootpath = join(get_root_wsol_dataset(), args.dataset)
    samples = csv_loader(subset_csv, rootpath)

    lh, lw = [], []
    for el in tqdm.tqdm(samples, ncols=150, total=len(samples)):
        img = Image.open(el[1], 'r').convert('RGB')
        w, h = img.size
        lh.append(h)
        lw.append(w)

    msg = "min h {}, \t max h {}".format(min(lh), max(lh))
    show_msg(msg, log)
    msg = "min w {}, \t max w {}".format(min(lw), max(lw))
    show_msg(msg, log)

    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].hist(lh)
    axes[0].set_title('Heights')
    axes[1].hist(lw)
    axes[1].set_title('Widths')
    fig.tight_layout()
    plt.savefig(join(args.fold_folder, "size-stats-{}.png".format(tag)))

    log.close()
