from copy import deepcopy
import os
import sys
from os.path import join, dirname, abspath, basename
import datetime as dt
import argparse

import numpy as np
import yaml
import munch

import torch
import torch.nn.functional as F
from PIL import Image


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
from dlib.datasets.wsol_loader import get_eval_tranforms


from dlib.cams import build_std_cam_extractor
from dlib.utils.reproducibility import set_seed
from dlib.process.instantiators import get_model

from dlib.functional import _functional as dlibf
from dlib.utils.tools import t2n
from dlib.utils.tools import check_scoremap_validity
from dlib.visualization.vision_wsol import Viz_WSOL


def build_eval_transformer(args):
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

    crop_size = args.crop_size
    transform = get_eval_tranforms(args,
                                   crop_size,
                                   scale_img,
                                   img_mean,
                                   img_std,
                                   avg_train_pixel
                                   )

    return transform


def load_img(img_path: str, transforms):
    assert os.path.isfile(img_path), img_path

    image = Image.open(img_path)
    image = image.convert('RGB')
    unmasked_img = image.copy()
    raw_img = image.copy()

    z = transforms(image,
                   unmasked_img,  # not used.
                   raw_img,
                   None,
                   None,
                   None,
                   None,
                   None,
                   None
                   )
    image, _, raw_img, _, _, _, _, _, _ = z

    raw_img = np.array(raw_img, dtype=np.float32)  # h, w, 3
    raw_img = dlibf.to_tensor(raw_img).permute(2, 0, 1)  # 3, h, w.
    # image: 3, h, w
    return image, raw_img


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


def classify_single_img(model, img: torch.Tensor, args) -> int:
    assert img.ndim == 3, img.ndim  # 3, h, w
    _img = img.unsqueeze(0)
    _img = _img.cuda()
    target = torch.zeros(1, dtype=torch.long, requires_grad=False).cuda()
    cl_logits = cl_forward(model, _img, target, args.task).detach()

    pred = cl_logits.argmax(dim=1)

    return pred[0].item()  # int


def _build_std_cam_extractor(classifier, args):
    return build_std_cam_extractor(classifier=classifier, args=args)


def get_cam_one_sample(model,
                       args,
                       image: torch.Tensor,
                       target: int
                       ) -> torch.Tensor:

    special1 = args.method in [constants.METHOD_TSCAM]

    img_shape = image.shape[2:]
    std_cam_extractor = _build_std_cam_extractor(
        classifier=model, args=args)

    output = model(image.cuda(), target)

    if args.task == constants.STD_CL:
        cl_logits = output
        cam = std_cam_extractor(class_idx=target,
                                scores=cl_logits,
                                normalized=True,
                                reshape=img_shape if special1 else None)
        # (h`, w`)
    else:
        raise NotImplementedError

    cam = torch.nan_to_num(cam, nan=0.0, posinf=1., neginf=0.0)

    return cam


def build_attention_one_sample(model,
                               args,
                               raw_img: np.ndarray
                               ) -> np.ndarray:
    # Note: plot the attention of the layer used for alignment.
    #  if many, take the last one.
    idx_layer_att = layers_str_2_int(args)[-1]
    attention = build_att(model, args, layer_id=idx_layer_att)
    # h, w

    attention = F.interpolate(input=attention,
                              size=raw_img.shape[:2],  # h, w, 3
                              mode='bilinear',
                              align_corners=True,
                              antialias=True
                              )
    attention = torch.clamp(attention, min=0.0, max=1.0)
    attention = attention.detach().squeeze().cpu().numpy()

    return attention


def layers_str_2_int(args) -> list:

    assert args.align_atten_to_heatmap

    align_atten_to_heatmap_layers = args.align_atten_to_heatmap_layers
    layers = align_atten_to_heatmap_layers
    assert isinstance(layers, str), type(layers)

    assert isinstance(layers, str), type(str)
    z = layers.split('-')
    z = [int(s) for s in z]
    assert len(z) > 0, f"{len(z)} | {layers}"

    return z


def build_att(model, args, layer_id: int) -> torch.Tensor:

    assert args.align_atten_to_heatmap

    features = model.features
    assert features != []
    if layer_id == -1:
        layer_id = len(features) - 1
        for i, ft in enumerate(features):
            if ft.ndim == 4:
                layer_id = i

    assert layer_id < len(model.features), f"{layer_id} {len(features)}"

    with torch.no_grad():
        f = features[layer_id]
        attention = f.mean(dim=1, keepdim=True)  # b, 1, h, w

    return attention


def localize_single_img(model,
                        args,
                        img: torch.Tensor,
                        in_raw_img: np.ndarray,
                        label: int,
                        label_str: str,
                        outd: str
                        ):
    viz = Viz_WSOL()
    assert img.ndim == 3, img.ndim  # 3, h, w

    # raw_img: 3, h, w
    raw_img = in_raw_img.permute(1, 2, 0).numpy()  # h, w, 3
    raw_img = raw_img.astype(np.uint8)

    image_size = img.shape[1:]

    low_cam = get_cam_one_sample(model,
                                 args,
                                 image=img.unsqueeze(0),
                                 target=label)

    with torch.no_grad():
        cam = F.interpolate(low_cam.unsqueeze(0).unsqueeze(0),
                            image_size,
                            mode='bilinear',
                            align_corners=False
                            ).squeeze(0).squeeze(0)

        cam = torch.clamp(cam, min=0.0, max=1.)

    cam = torch.clamp(cam, min=0.0, max=1.)
    cam = t2n(cam)

    # cams shape (h, w).
    assert cam.shape == image_size

    cam_resized = cam
    cam_normalized = cam_resized
    check_scoremap_validity(cam_normalized)

    attention = build_attention_one_sample(model, args, raw_img)

    show_cam = (args.method != constants.METHOD_APVIT)

    datum = {'img': raw_img,
             'img_id': 'ID',
             'gt_bbox': None,
             'pred_bbox': None,
             'iou': None,
             'tau': None,
             'sigma': None,
             'cam': cam_normalized,
             'tag_cl': 'tag_cl',
             'heatmap': None,
             'tag_heatmap': None,
             'attention': attention,
             'img_class_str': label_str
             }

    outf = join(outd, 'visualization.png')
    viz.plot_fer_single_less(
        datum=datum, outf=outf, dpi=50,
        show_cam=show_cam)


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
    parser.add_argument("--checkpoint_type", type=str, default=None)
    parser.add_argument("--exp_path", type=str, default=None)

    parsedargs = parser.parse_args()
    exp_path = parsedargs.exp_path
    checkpoint_type = parsedargs.checkpoint_type

    img_path = join(root_dir, 'samples/test_0097.jpg')

    split = constants.TESTSET
    assert os.path.isdir(exp_path)
    assert split in constants.SPLITS

    assert checkpoint_type in [constants.BEST, constants.LAST]

    _CODE_FUNCTION = f'eval_single_{basename(img_path)}'

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

    outd = join(exp_path, _CODE_FUNCTION, f'checkpoint_{checkpoint_type}')

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
    DLLogger.log(fmsg(f"Evaluate epoch {epoch}, Single image {img_path}"))

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

    args.outd = outd

    basic_config = config.get_config(ds=args.dataset)
    args.data_root = basic_config['data_root']
    args.data_paths = basic_config['data_paths']
    args.metadata_root = basic_config['metadata_root']
    args.mask_root = basic_config['mask_root']

    folds_path = join(root_dir, args.metadata_root)
    path_class_id = join(folds_path, 'class_id.yaml')
    with open(path_class_id, 'r') as fcl:
        cl_int = yaml.safe_load(fcl)

    cl_to_int: dict = cl_int
    int_to_cl: dict = switch_key_val_dict(cl_int)

    transforms = build_eval_transformer(args)

    in_img, in_raw_img = load_img(img_path, transforms)
    # classification
    cl_pred = classify_single_img(model, in_img, args)
    msg = f"Predicted class (Single image): '{int_to_cl[cl_pred]}'."
    DLLogger.log(fmsg(msg))

    log_path = join(args.outd, f'eval_cl_log{tag}.txt')
    with open(log_path, 'w') as f:
        f.write(msg)

    # localization.
    localize_single_img(model,
                        args,
                        in_img,
                        in_raw_img,
                        cl_pred,
                        int_to_cl[cl_pred],
                        outd)

    DLLogger.log(fmsg("Bye."))


if __name__ == '__main__':
    evaluate()

