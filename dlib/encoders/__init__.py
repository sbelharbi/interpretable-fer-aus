import os
import sys
from os.path import dirname, abspath, join

import functools

import torch
import torch.utils.model_zoo as model_zoo

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.utils.shared import is_cc
import dlib.dllogger as DLLogger
# from dlib.vggface2 import keras_to_pytorch

from dlib.encoders.resnet import resnet_encoders
from dlib.encoders.dpn import dpn_encoders
from dlib.encoders.vgg import vgg_encoders
from dlib.encoders.senet import senet_encoders
from dlib.encoders.densenet import densenet_encoders
from dlib.encoders.inceptionresnetv2 import inceptionresnetv2_encoders
from dlib.encoders.inceptionv4 import inceptionv4_encoders
from dlib.encoders.inceptionv3 import inceptionv3_encoders
from dlib.encoders.efficientnet import efficient_net_encoders
from dlib.encoders.mobilenet import mobilenet_encoders
from dlib.encoders.xception import xception_encoders
from dlib.encoders.timm_efficientnet import timm_efficientnet_encoders
from dlib.encoders.timm_resnest import timm_resnest_encoders
from dlib.encoders.timm_res2net import timm_res2net_encoders
from dlib.encoders.timm_regnet import timm_regnet_encoders
from dlib.encoders.timm_sknet import timm_sknet_encoders
from dlib.encoders._preprocessing import preprocess_input

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inceptionresnetv2_encoders)
encoders.update(inceptionv4_encoders)
encoders.update(inceptionv3_encoders)
encoders.update(efficient_net_encoders)
encoders.update(mobilenet_encoders)
encoders.update(xception_encoders)
encoders.update(timm_efficientnet_encoders)
encoders.update(timm_resnest_encoders)
encoders.update(timm_res2net_encoders)
encoders.update(timm_regnet_encoders)
encoders.update(timm_sknet_encoders)


vggface2_settings = {
    constants.RESNET50: "pytorch-vggface2_notop_resnet50.pt"
}


def get_encoder(name,
                in_channels=3,
                depth=5,
                weights=None,
                large_maps=False,
                use_adl: bool = False,
                adl_drop_rate: float = .4,
                adl_drop_threshold: float = .1,
                apply_self_atten: bool = False,
                atten_layers_id: list = None
                ):

    try:
        Encoder = encoders[name]["encoder"]
    except KeyError:
        raise KeyError("Wrong encoder name `{}`, supported encoders: {}".format(
            name, list(encoders.keys())))

    params = encoders[name]["params"]
    params.update(depth=depth)
    # todo: update vgg, inception.
    encoder = Encoder(**params,
                      large_maps=large_maps,
                      use_adl=use_adl, adl_drop_rate=adl_drop_rate,
                      adl_drop_threshold=adl_drop_threshold,
                      apply_self_atten=apply_self_atten,
                      atten_layers_id=atten_layers_id
                      )

    state_dict = None
    strict = True

    if weights not in [None, constants.VGGFACE2]:
        try:
            settings = encoders[name]["pretrained_settings"][weights]
        except KeyError:
            raise KeyError(
                "Wrong pretrained weights `{}` for encoder `{}`. "
                "Available options are: {}".format(
                    weights, name,
                    list(encoders[name]["pretrained_settings"].keys()),)
            )

        if name in [constants.INCEPTIONV3, constants.VGG16]:
            strict = False
        model_dir = join(root_dir, f'pretrained-{weights}')
        os.makedirs(model_dir, exist_ok=True)
        state_dict = model_zoo.load_url(url=settings["url"],
                                        model_dir=model_dir)
        DLLogger.log(f"Loaded pretrained weights: {name} / {weights}.")


    if weights == constants.VGGFACE2:
        model_dir = join(root_dir, f'pretrained-{weights}')
        path_file = join(model_dir, vggface2_settings[name])

        if not os.path.isfile(path_file):
            os.makedirs(model_dir, exist_ok=True)
            # avoid circular import...
            # todo: deal with multi-gpu.
            cmd = "python dlib/vggface2/kerastopytorch.py"
            os.system(cmd)
            # keras_to_pytorch()

        device = next(encoder.parameters()).device
        state_dict = torch.load(path_file, map_location=device)
        DLLogger.log(f"Loaded pretrained weights: {name} / {weights}.")


    if state_dict is not None:

        if use_adl:
            strict = False

        encoder.load_state_dict(state_dict,  strict=strict)


    encoder.set_in_channels(in_channels)
    encoder.set_model_name(name)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):
    settings = encoders[encoder_name]["pretrained_settings"]

    if pretrained not in settings.keys():
        raise ValueError("Available pretrained options {}".format(settings.keys()))

    formatted_settings = {}
    formatted_settings["input_space"] = settings[pretrained].get("input_space")
    formatted_settings["input_range"] = settings[pretrained].get("input_range")
    formatted_settings["mean"] = settings[pretrained].get("mean")
    formatted_settings["std"] = settings[pretrained].get("std")
    return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
