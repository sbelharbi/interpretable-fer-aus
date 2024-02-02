import os
import sys
from os.path import dirname, abspath, join
from typing import Optional, Union, List

import numpy as np

import torch
from torch.cuda.amp import autocast

# from keras.preprocessing import image
import keras.utils as image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders import get_encoder
from dlib.configure import constants


__all__ = ['keras_to_pytorch']


def vggface_resnet():


    # tensorflow
    model = VGGFace(model='resnet50', weights='vggface')
    # default : VGG16 , you can use model='resnet50' or 'senet50'
    # issues:
    # 1.
    # https://github.com/rcmalli/keras-vggface/issues/73#issuecomment-907625857
    # from keras.engine.topology import get_source_inputs
    # to
    # from keras.utils.layer_utils import get_source_inputs in keras_vggface/models.py.

    # Change the image path with yours.
    d = join(root_dir, 'data/debug/input')
    img = image.load_img(join(d, 'ajb.jpg'), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print(x.min(), x.max())
    x = utils.preprocess_input(x, version=2)  # or version=2
    print(x.min(), x.max(), x.shape)
    preds = model.predict(x)
    print('Predicted:', utils.decode_predictions(preds))
    weights =  model.get_weights()
    names = [weight.name for layer in model.layers for weight in layer.weights]
    for name, weight in zip(names, weights):
        print(name, weight.shape)

    for i, w in enumerate(weights):
        print(i, w.shape, np.transpose(w).shape)

def transp_torch(x: np.ndarray) -> torch.Tensor:
    # transpose, to torch, float
    return torch.from_numpy(np.transpose(x)).float()

def to_torch(x: np.ndarray) -> torch.Tensor:
    # to torch, float.
    return torch.from_numpy(x).float()

def keras_to_pytorch():
    print(f"Building pretrained weights of {constants.RESNET50} over "
          f"{constants.VGGFACE2} ...")
    # keras resnet50, pretrained on vggface2, no top.
    k_resnet50 = VGGFace(include_top=False,
                         model='resnet50',
                         weights='vggface',
                         input_shape=(224, 244, 3)
                         )

    kweights = k_resnet50.get_weights()
    knames = [weight.name for layer in k_resnet50.layers for weight in
              layer.weights]
    print('Keras')
    i = 0
    for name, weight in zip(knames, kweights):
        # print(i, name, weight.shape)
        i += 1

    encoder_name = constants.RESNET50
    in_channels = 3
    encoder_depth = 5
    encoder_weights = None

    # encoder only.
    p_resnet50 = get_encoder(
        encoder_name,
        in_channels=in_channels,
        depth=encoder_depth,
        weights=encoder_weights
    )

    print('Pytorch')
    # for name, module in p_resnet50.named_modules():
    #     print(name)

    # Weights alignment.

    # preamble =================================================================
    # conv1: 0 conv1/7x7_s2/kernel:0 (7, 7, 3, 64)
    p_resnet50.conv1.weight.data = transp_torch(kweights[0])

    # bn1:
    #      1 conv1/7x7_s2/bn/gamma:0 (64,)
    #      2 conv1/7x7_s2/bn/beta:0 (64,)
    #      3 conv1/7x7_s2/bn/moving_mean:0 (64,)
    #      4 conv1/7x7_s2/bn/moving_variance:0 (64,)
    p_resnet50.bn1.weight.data = to_torch(kweights[1])
    p_resnet50.bn1.bias.data = to_torch(kweights[2])
    p_resnet50.bn1.running_mean.data = to_torch(kweights[3])
    p_resnet50.bn1.running_var.data = to_torch(kweights[4])

    # layer 1 ==================================================================
    # layer1
    # layer1.0

    # layer1.0.conv1: 5 conv2_1_1x1_reduce/kernel:0 (1, 1, 64, 64)
    p_resnet50.layer1[0].conv1.weight.data = transp_torch(kweights[5])

    # layer1.0.bn1:
    # 6 conv2_1_1x1_reduce/bn/gamma:0 (64,)
    # 7 conv2_1_1x1_reduce/bn/beta:0 (64,)
    # 8 conv2_1_1x1_reduce/bn/moving_mean:0 (64,)
    # 9 conv2_1_1x1_reduce/bn/moving_variance:0 (64,)
    p_resnet50.layer1[0].bn1.weight.data = to_torch(kweights[6])
    p_resnet50.layer1[0].bn1.bias.data = to_torch(kweights[7])
    p_resnet50.layer1[0].bn1.running_mean.data = to_torch(kweights[8])
    p_resnet50.layer1[0].bn1.running_var.data = to_torch(kweights[9])

    # layer1.0.conv2: 10 conv2_1_3x3/kernel:0 (3, 3, 64, 64)
    p_resnet50.layer1[0].conv2.weight.data = transp_torch(kweights[10])

    # layer1.0.bn2
    # 11 conv2_1_3x3/bn/gamma:0 (64,)
    # 12 conv2_1_3x3/bn/beta:0 (64,)
    # 13 conv2_1_3x3/bn/moving_mean:0 (64,)
    # 14 conv2_1_3x3/bn/moving_variance:0 (64,)
    p_resnet50.layer1[0].bn2.weight.data = to_torch(kweights[11])
    p_resnet50.layer1[0].bn2.bias.data = to_torch(kweights[12])
    p_resnet50.layer1[0].bn2.running_mean.data = to_torch(kweights[13])
    p_resnet50.layer1[0].bn2.running_var.data = to_torch(kweights[14])

    # layer1.0.conv3: 15 conv2_1_1x1_increase/kernel:0 (1, 1, 64, 256)
    p_resnet50.layer1[0].conv3.weight.data = transp_torch(kweights[15])
    # layer1.0.bn3
    # 17 conv2_1_1x1_increase/bn/gamma:0 (256,)
    # 18 conv2_1_1x1_increase/bn/beta:0 (256,)
    # 19 conv2_1_1x1_increase/bn/moving_mean:0 (256,)
    # 20 conv2_1_1x1_increase/bn/moving_variance:0 (256,)
    p_resnet50.layer1[0].bn3.weight.data = to_torch(kweights[17])
    p_resnet50.layer1[0].bn3.bias.data = to_torch(kweights[18])
    p_resnet50.layer1[0].bn3.running_mean.data = to_torch(kweights[19])
    p_resnet50.layer1[0].bn3.running_var.data = to_torch(kweights[20])

    # layer1.0.downsample
    # layer1.0.downsample.0 (conv): 16 conv2_1_1x1_proj/kernel:0 (1, 1, 64, 256)
    p_resnet50.layer1[0].downsample[0].weight.data = transp_torch(kweights[16])

    # layer1.0.downsample.1 (bn)
    # 21 conv2_1_1x1_proj/bn/gamma:0 (256,)
    # 22 conv2_1_1x1_proj/bn/beta:0 (256,)
    # 23 conv2_1_1x1_proj/bn/moving_mean:0 (256,)
    # 24 conv2_1_1x1_proj/bn/moving_variance:0 (256,)
    p_resnet50.layer1[0].downsample[1].weight.data = to_torch(kweights[21])
    p_resnet50.layer1[0].downsample[1].bias.data = to_torch(kweights[22])
    p_resnet50.layer1[0].downsample[1].running_mean.data = to_torch(
        kweights[23])
    p_resnet50.layer1[0].downsample[1].running_var.data = to_torch(kweights[24])


    # layer1.1
    # layer1.1.conv1: 25 conv2_2_1x1_reduce/kernel:0 (1, 1, 256, 64)
    p_resnet50.layer1[1].conv1.weight.data = transp_torch(kweights[25])

    # layer1.1.bn1:
    # 26 conv2_2_1x1_reduce/bn/gamma:0 (64,)
    # 27 conv2_2_1x1_reduce/bn/beta:0 (64,)
    # 28 conv2_2_1x1_reduce/bn/moving_mean:0 (64,)
    # 29 conv2_2_1x1_reduce/bn/moving_variance:0 (64,)
    p_resnet50.layer1[1].bn1.weight.data = to_torch(kweights[26])
    p_resnet50.layer1[1].bn1.bias.data = to_torch(kweights[27])
    p_resnet50.layer1[1].bn1.running_mean.data = to_torch(kweights[28])
    p_resnet50.layer1[1].bn1.running_var.data = to_torch(kweights[29])

    # layer1.1.conv2: 30 conv2_2_3x3/kernel:0 (3, 3, 64, 64)
    p_resnet50.layer1[1].conv2.weight.data = transp_torch(kweights[30])

    # layer1.1.bn2
    # 31 conv2_2_3x3/bn/gamma:0 (64,)
    # 32 conv2_2_3x3/bn/beta:0 (64,)
    # 33 conv2_2_3x3/bn/moving_mean:0 (64,)
    # 34 conv2_2_3x3/bn/moving_variance:0 (64,)
    p_resnet50.layer1[1].bn2.weight.data = to_torch(kweights[31])
    p_resnet50.layer1[1].bn2.bias.data = to_torch(kweights[32])
    p_resnet50.layer1[1].bn2.running_mean.data = to_torch(kweights[33])
    p_resnet50.layer1[1].bn2.running_var.data = to_torch(kweights[34])

    # layer1.1.conv3: 35 conv2_2_1x1_increase/kernel:0 (1, 1, 64, 256)
    p_resnet50.layer1[1].conv3.weight.data = transp_torch(kweights[35])

    # layer1.1.bn3
    # 36 conv2_2_1x1_increase/bn/gamma:0 (256,)
    # 37 conv2_2_1x1_increase/bn/beta:0 (256,)
    # 38 conv2_2_1x1_increase/bn/moving_mean:0 (256,)
    # 39 conv2_2_1x1_increase/bn/moving_variance:0 (256,)
    p_resnet50.layer1[1].bn3.weight.data = to_torch(kweights[36])
    p_resnet50.layer1[1].bn3.bias.data = to_torch(kweights[37])
    p_resnet50.layer1[1].bn3.running_mean.data = to_torch(kweights[38])
    p_resnet50.layer1[1].bn3.running_var.data = to_torch(kweights[39])


    # layer1.2
    # layer1.2.conv1: 40 conv2_3_1x1_reduce/kernel:0 (1, 1, 256, 64)
    p_resnet50.layer1[2].conv1.weight.data = transp_torch(kweights[40])

    # layer1.2.bn1:
    # 41 conv2_3_1x1_reduce/bn/gamma:0 (64,)
    # 42 conv2_3_1x1_reduce/bn/beta:0 (64,)
    # 43 conv2_3_1x1_reduce/bn/moving_mean:0 (64,)
    # 44 conv2_3_1x1_reduce/bn/moving_variance:0 (64,)
    p_resnet50.layer1[2].bn1.weight.data = to_torch(kweights[41])
    p_resnet50.layer1[2].bn1.bias.data = to_torch(kweights[42])
    p_resnet50.layer1[2].bn1.running_mean.data = to_torch(kweights[43])
    p_resnet50.layer1[2].bn1.running_var.data = to_torch(kweights[44])

    # layer1.2.conv2:  45 conv2_3_3x3/kernel:0 (3, 3, 64, 64)
    p_resnet50.layer1[2].conv2.weight.data = transp_torch(kweights[45])

    # layer1.2.bn2:
    # 46 conv2_3_3x3/bn/gamma:0 (64,)
    # 47 conv2_3_3x3/bn/beta:0 (64,)
    # 48 conv2_3_3x3/bn/moving_mean:0 (64,)
    # 49 conv2_3_3x3/bn/moving_variance:0 (64,)
    p_resnet50.layer1[2].bn2.weight.data = to_torch(kweights[46])
    p_resnet50.layer1[2].bn2.bias.data = to_torch(kweights[47])
    p_resnet50.layer1[2].bn2.running_mean.data = to_torch(kweights[48])
    p_resnet50.layer1[2].bn2.running_var.data = to_torch(kweights[49])

    # layer1.2.conv3: 50 conv2_3_1x1_increase/kernel:0 (1, 1, 64, 256)
    p_resnet50.layer1[2].conv3.weight.data = transp_torch(kweights[50])

    # layer1.2.bn3:
    # 51 conv2_3_1x1_increase/bn/gamma:0 (256,)
    # 52 conv2_3_1x1_increase/bn/beta:0 (256,)
    # 53 conv2_3_1x1_increase/bn/moving_mean:0 (256,)
    # 54 conv2_3_1x1_increase/bn/moving_variance:0 (256,)
    p_resnet50.layer1[2].bn3.weight.data = to_torch(kweights[51])
    p_resnet50.layer1[2].bn3.bias.data = to_torch(kweights[52])
    p_resnet50.layer1[2].bn3.running_mean.data = to_torch(kweights[53])
    p_resnet50.layer1[2].bn3.running_var.data = to_torch(kweights[54])

    # layer2 ===================================================================
    # layer2.0
    # layer2.0.conv1: 55 conv3_1_1x1_reduce/kernel:0 (1, 1, 256, 128)
    p_resnet50.layer2[0].conv1.weight.data = transp_torch(kweights[55])

    # layer2.0.bn1
    # 56 conv3_1_1x1_reduce/bn/gamma:0 (128,)
    # 57 conv3_1_1x1_reduce/bn/beta:0 (128,)
    # 58 conv3_1_1x1_reduce/bn/moving_mean:0 (128,)
    # 59 conv3_1_1x1_reduce/bn/moving_variance:0 (128,)
    p_resnet50.layer2[0].bn1.weight.data = to_torch(kweights[56])
    p_resnet50.layer2[0].bn1.bias.data = to_torch(kweights[57])
    p_resnet50.layer2[0].bn1.running_mean.data = to_torch(kweights[58])
    p_resnet50.layer2[0].bn1.running_var.data = to_torch(kweights[59])

    # layer2.0.conv2: 60 conv3_1_3x3/kernel:0 (3, 3, 128, 128)
    p_resnet50.layer2[0].conv2.weight.data = transp_torch(kweights[60])

    # layer2.0.bn2
    # 61 conv3_1_3x3/bn/gamma:0 (128,)
    # 62 conv3_1_3x3/bn/beta:0 (128,)
    # 63 conv3_1_3x3/bn/moving_mean:0 (128,)
    # 64 conv3_1_3x3/bn/moving_variance:0 (128,)
    p_resnet50.layer2[0].bn2.weight.data = to_torch(kweights[61])
    p_resnet50.layer2[0].bn2.bias.data = to_torch(kweights[62])
    p_resnet50.layer2[0].bn2.running_mean.data = to_torch(kweights[63])
    p_resnet50.layer2[0].bn2.running_var.data = to_torch(kweights[64])

    # layer2.0.conv3: 65 conv3_1_1x1_increase/kernel:0 (1, 1, 128, 512)
    p_resnet50.layer2[0].conv3.weight.data = transp_torch(kweights[65])

    # layer2.0.bn3
    # 67 conv3_1_1x1_increase/bn/gamma:0 (512,)
    # 68 conv3_1_1x1_increase/bn/beta:0 (512,)
    # 69 conv3_1_1x1_increase/bn/moving_mean:0 (512,)
    # 70 conv3_1_1x1_increase/bn/moving_variance:0 (512,)
    p_resnet50.layer2[0].bn3.weight.data = to_torch(kweights[67])
    p_resnet50.layer2[0].bn3.bias.data = to_torch(kweights[68])
    p_resnet50.layer2[0].bn3.running_mean.data = to_torch(kweights[69])
    p_resnet50.layer2[0].bn3.running_var.data = to_torch(kweights[70])

    # layer2.0.downsample
    # layer2.0.downsample.0 (conv):66 conv3_1_1x1_proj/kernel:0 (1, 1, 256, 512)
    p_resnet50.layer2[0].downsample[0].weight.data = transp_torch(kweights[66])

    # layer2.0.downsample.1 (bn)
    # 71 conv3_1_1x1_proj/bn/gamma:0 (512,)
    # 72 conv3_1_1x1_proj/bn/beta:0 (512,)
    # 73 conv3_1_1x1_proj/bn/moving_mean:0 (512,)
    # 74 conv3_1_1x1_proj/bn/moving_variance:0 (512,)
    p_resnet50.layer2[0].downsample[1].weight.data = to_torch(kweights[71])
    p_resnet50.layer2[0].downsample[1].bias.data = to_torch(kweights[72])
    p_resnet50.layer2[0].downsample[1].running_mean.data = to_torch(
        kweights[73])
    p_resnet50.layer2[0].downsample[1].running_var.data = to_torch(kweights[74])

    # layer2.1
    # layer2.1.conv1: 75 conv3_2_1x1_reduce/kernel:0 (1, 1, 512, 128)
    p_resnet50.layer2[1].conv1.weight.data = transp_torch(kweights[75])

    # layer2.1.bn1
    # 76 conv3_2_1x1_reduce/bn/gamma:0 (128,)
    # 77 conv3_2_1x1_reduce/bn/beta:0 (128,)
    # 78 conv3_2_1x1_reduce/bn/moving_mean:0 (128,)
    # 79 conv3_2_1x1_reduce/bn/moving_variance:0 (128,)
    p_resnet50.layer2[1].bn1.weight.data = to_torch(kweights[76])
    p_resnet50.layer2[1].bn1.bias.data = to_torch(kweights[77])
    p_resnet50.layer2[1].bn1.running_mean.data = to_torch(kweights[78])
    p_resnet50.layer2[1].bn1.running_var.data = to_torch(kweights[79])

    # layer2.1.conv2: 80 conv3_2_3x3/kernel:0 (3, 3, 128, 128)
    p_resnet50.layer2[1].conv2.weight.data = transp_torch(kweights[80])

    # layer2.1.bn2
    # 81 conv3_2_3x3/bn/gamma:0 (128,)
    # 82 conv3_2_3x3/bn/beta:0 (128,)
    # 83 conv3_2_3x3/bn/moving_mean:0 (128,)
    # 84 conv3_2_3x3/bn/moving_variance:0 (128,)
    p_resnet50.layer2[1].bn2.weight.data = to_torch(kweights[81])
    p_resnet50.layer2[1].bn2.bias.data = to_torch(kweights[82])
    p_resnet50.layer2[1].bn2.running_mean.data = to_torch(kweights[83])
    p_resnet50.layer2[1].bn2.running_var.data = to_torch(kweights[84])

    # layer2.1.conv3: 85 conv3_2_1x1_increase/kernel:0 (1, 1, 128, 512)
    p_resnet50.layer2[1].conv3.weight.data = transp_torch(kweights[85])

    # layer2.1.bn3
    # 86 conv3_2_1x1_increase/bn/gamma:0 (512,)
    # 87 conv3_2_1x1_increase/bn/beta:0 (512,)
    # 88 conv3_2_1x1_increase/bn/moving_mean:0 (512,)
    # 89 conv3_2_1x1_increase/bn/moving_variance:0 (512,)
    p_resnet50.layer2[1].bn3.weight.data = to_torch(kweights[86])
    p_resnet50.layer2[1].bn3.bias.data = to_torch(kweights[87])
    p_resnet50.layer2[1].bn3.running_mean.data = to_torch(kweights[88])
    p_resnet50.layer2[1].bn3.running_var.data = to_torch(kweights[89])


    # layer2.2
    # layer2.2.conv1: 90 conv3_3_1x1_reduce/kernel:0 (1, 1, 512, 128)
    p_resnet50.layer2[2].conv1.weight.data = transp_torch(kweights[90])

    # layer2.2.bn1
    # 91 conv3_3_1x1_reduce/bn/gamma:0 (128,)
    # 92 conv3_3_1x1_reduce/bn/beta:0 (128,)
    # 93 conv3_3_1x1_reduce/bn/moving_mean:0 (128,)
    # 94 conv3_3_1x1_reduce/bn/moving_variance:0 (128,)
    p_resnet50.layer2[2].bn1.weight.data = to_torch(kweights[91])
    p_resnet50.layer2[2].bn1.bias.data = to_torch(kweights[92])
    p_resnet50.layer2[2].bn1.running_mean.data = to_torch(kweights[93])
    p_resnet50.layer2[2].bn1.running_var.data = to_torch(kweights[94])

    # layer2.2.conv2: 95 conv3_3_3x3/kernel:0 (3, 3, 128, 128)
    p_resnet50.layer2[2].conv2.weight.data = transp_torch(kweights[95])

    # layer2.2.bn2
    # 96 conv3_3_3x3/bn/gamma:0 (128,)
    # 97 conv3_3_3x3/bn/beta:0 (128,)
    # 98 conv3_3_3x3/bn/moving_mean:0 (128,)
    # 99 conv3_3_3x3/bn/moving_variance:0 (128,)
    p_resnet50.layer2[2].bn2.weight.data = to_torch(kweights[96])
    p_resnet50.layer2[2].bn2.bias.data = to_torch(kweights[97])
    p_resnet50.layer2[2].bn2.running_mean.data = to_torch(kweights[98])
    p_resnet50.layer2[2].bn2.running_var.data = to_torch(kweights[99])

    # layer2.2.conv3: 100 conv3_3_1x1_increase/kernel:0 (1, 1, 128, 512)
    p_resnet50.layer2[2].conv3.weight.data = transp_torch(kweights[100])

    # layer2.2.bn3
    # 101 conv3_3_1x1_increase/bn/gamma:0 (512,)
    # 102 conv3_3_1x1_increase/bn/beta:0 (512,)
    # 103 conv3_3_1x1_increase/bn/moving_mean:0 (512,)
    # 104 conv3_3_1x1_increase/bn/moving_variance:0 (512,)
    p_resnet50.layer2[2].bn3.weight.data = to_torch(kweights[101])
    p_resnet50.layer2[2].bn3.bias.data = to_torch(kweights[102])
    p_resnet50.layer2[2].bn3.running_mean.data = to_torch(kweights[103])
    p_resnet50.layer2[2].bn3.running_var.data = to_torch(kweights[104])

    # layer2.3
    # layer2.3.conv1: 105 conv3_4_1x1_reduce/kernel:0 (1, 1, 512, 128)
    p_resnet50.layer2[3].conv1.weight.data = transp_torch(kweights[105])

    # layer2.3.bn1
    # 106 conv3_4_1x1_reduce/bn/gamma:0 (128,)
    # 107 conv3_4_1x1_reduce/bn/beta:0 (128,)
    # 108 conv3_4_1x1_reduce/bn/moving_mean:0 (128,)
    # 109 conv3_4_1x1_reduce/bn/moving_variance:0 (128,)
    p_resnet50.layer2[3].bn1.weight.data = to_torch(kweights[106])
    p_resnet50.layer2[3].bn1.bias.data = to_torch(kweights[107])
    p_resnet50.layer2[3].bn1.running_mean.data = to_torch(kweights[108])
    p_resnet50.layer2[3].bn1.running_var.data = to_torch(kweights[109])

    # layer2.3.conv2: 110 conv3_4_3x3/kernel:0 (3, 3, 128, 128)
    p_resnet50.layer2[3].conv2.weight.data = transp_torch(kweights[110])

    # layer2.3.bn2
    # 111 conv3_4_3x3/bn/gamma:0 (128,)
    # 112 conv3_4_3x3/bn/beta:0 (128,)
    # 113 conv3_4_3x3/bn/moving_mean:0 (128,)
    # 114 conv3_4_3x3/bn/moving_variance:0 (128,)
    p_resnet50.layer2[3].bn2.weight.data = to_torch(kweights[111])
    p_resnet50.layer2[3].bn2.bias.data = to_torch(kweights[112])
    p_resnet50.layer2[3].bn2.running_mean.data = to_torch(kweights[113])
    p_resnet50.layer2[3].bn2.running_var.data = to_torch(kweights[114])

    # layer2.3.conv3: 115 conv3_4_1x1_increase/kernel:0 (1, 1, 128, 512)
    p_resnet50.layer2[3].conv3.weight.data = transp_torch(kweights[115])

    # layer2.3.bn3
    # 116 conv3_4_1x1_increase/bn/gamma:0 (512,)
    # 117 conv3_4_1x1_increase/bn/beta:0 (512,)
    # 118 conv3_4_1x1_increase/bn/moving_mean:0 (512,)
    # 119 conv3_4_1x1_increase/bn/moving_variance:0 (512,)
    p_resnet50.layer2[3].bn3.weight.data = to_torch(kweights[116])
    p_resnet50.layer2[3].bn3.bias.data = to_torch(kweights[117])
    p_resnet50.layer2[3].bn3.running_mean.data = to_torch(kweights[118])
    p_resnet50.layer2[3].bn3.running_var.data = to_torch(kweights[119])

    # layer3 ===================================================================
    # layer3.0
    # layer3.0.conv1: 120 conv4_1_1x1_reduce/kernel:0 (1, 1, 512, 256)
    p_resnet50.layer3[0].conv1.weight.data = transp_torch(kweights[120])

    # layer3.0.bn1
    # 121 conv4_1_1x1_reduce/bn/gamma:0 (256,)
    # 122 conv4_1_1x1_reduce/bn/beta:0 (256,)
    # 123 conv4_1_1x1_reduce/bn/moving_mean:0 (256,)
    # 124 conv4_1_1x1_reduce/bn/moving_variance:0 (256,)
    p_resnet50.layer3[0].bn1.weight.data = to_torch(kweights[121])
    p_resnet50.layer3[0].bn1.bias.data = to_torch(kweights[122])
    p_resnet50.layer3[0].bn1.running_mean.data = to_torch(kweights[123])
    p_resnet50.layer3[0].bn1.running_var.data = to_torch(kweights[124])

    # layer3.0.conv2: 125 conv4_1_3x3/kernel:0 (3, 3, 256, 256)
    p_resnet50.layer3[0].conv2.weight.data = transp_torch(kweights[125])

    # layer3.0.bn2
    # 126 conv4_1_3x3/bn/gamma:0 (256,)
    # 127 conv4_1_3x3/bn/beta:0 (256,)
    # 128 conv4_1_3x3/bn/moving_mean:0 (256,)
    # 129 conv4_1_3x3/bn/moving_variance:0 (256,)
    p_resnet50.layer3[0].bn2.weight.data = to_torch(kweights[126])
    p_resnet50.layer3[0].bn2.bias.data = to_torch(kweights[127])
    p_resnet50.layer3[0].bn2.running_mean.data = to_torch(kweights[128])
    p_resnet50.layer3[0].bn2.running_var.data = to_torch(kweights[129])

    # layer3.0.conv3: 130 conv4_1_1x1_increase/kernel:0 (1, 1, 256, 1024)
    p_resnet50.layer3[0].conv3.weight.data = transp_torch(kweights[130])

    # layer3.0.bn3
    # 132 conv4_1_1x1_increase/bn/gamma:0 (1024,)
    # 133 conv4_1_1x1_increase/bn/beta:0 (1024,)
    # 134 conv4_1_1x1_increase/bn/moving_mean:0 (1024,)
    # 135 conv4_1_1x1_increase/bn/moving_variance:0 (1024,)
    p_resnet50.layer3[0].bn3.weight.data = to_torch(kweights[132])
    p_resnet50.layer3[0].bn3.bias.data = to_torch(kweights[133])
    p_resnet50.layer3[0].bn3.running_mean.data = to_torch(kweights[134])
    p_resnet50.layer3[0].bn3.running_var.data = to_torch(kweights[135])

    # layer3.0.downsample
    # layer3.0.downsample.0 (conv): 131 conv4_1_1x1_proj/kernel:0 (1, 1, 512, 1024)
    p_resnet50.layer3[0].downsample[0].weight.data = transp_torch(kweights[131])

    # layer3.0.downsample.1 (bn)
    # 136 conv4_1_1x1_proj/bn/gamma:0 (1024,)
    # 137 conv4_1_1x1_proj/bn/beta:0 (1024,)
    # 138 conv4_1_1x1_proj/bn/moving_mean:0 (1024,)
    # 139 conv4_1_1x1_proj/bn/moving_variance:0 (1024,)
    p_resnet50.layer3[0].downsample[1].weight.data = to_torch(kweights[136])
    p_resnet50.layer3[0].downsample[1].bias.data = to_torch(kweights[137])
    p_resnet50.layer3[0].downsample[1].running_mean.data = to_torch(
        kweights[138])
    p_resnet50.layer3[0].downsample[1].running_var.data = to_torch(
        kweights[139])


    # layer3.1
    # layer3.1.conv1: 140 conv4_2_1x1_reduce/kernel:0 (1, 1, 1024, 256)
    p_resnet50.layer3[1].conv1.weight.data = transp_torch(kweights[140])

    # layer3.1.bn1
    # 141 conv4_2_1x1_reduce/bn/gamma:0 (256,)
    # 142 conv4_2_1x1_reduce/bn/beta:0 (256,)
    # 143 conv4_2_1x1_reduce/bn/moving_mean:0 (256,)
    # 144 conv4_2_1x1_reduce/bn/moving_variance:0 (256,)
    p_resnet50.layer3[1].bn1.weight.data = to_torch(kweights[141])
    p_resnet50.layer3[1].bn1.bias.data = to_torch(kweights[142])
    p_resnet50.layer3[1].bn1.running_mean.data = to_torch(kweights[143])
    p_resnet50.layer3[1].bn1.running_var.data = to_torch(kweights[144])

    # layer3.1.conv2: 145 conv4_2_3x3/kernel:0 (3, 3, 256, 256)
    p_resnet50.layer3[1].conv2.weight.data = transp_torch(kweights[145])

    # layer3.1.bn2:
    # 146 conv4_2_3x3/bn/gamma:0 (256,)
    # 147 conv4_2_3x3/bn/beta:0 (256,)
    # 148 conv4_2_3x3/bn/moving_mean:0 (256,)
    # 149 conv4_2_3x3/bn/moving_variance:0 (256,)
    p_resnet50.layer3[1].bn2.weight.data = to_torch(kweights[146])
    p_resnet50.layer3[1].bn2.bias.data = to_torch(kweights[147])
    p_resnet50.layer3[1].bn2.running_mean.data = to_torch(kweights[148])
    p_resnet50.layer3[1].bn2.running_var.data = to_torch(kweights[149])

    # layer3.1.conv3: 150 conv4_2_1x1_increase/kernel:0 (1, 1, 256, 1024)
    p_resnet50.layer3[1].conv3.weight.data = transp_torch(kweights[150])

    # layer3.1.bn3
    # 151 conv4_2_1x1_increase/bn/gamma:0 (1024,)
    # 152 conv4_2_1x1_increase/bn/beta:0 (1024,)
    # 153 conv4_2_1x1_increase/bn/moving_mean:0 (1024,)
    # 154 conv4_2_1x1_increase/bn/moving_variance:0 (1024,)
    p_resnet50.layer3[1].bn3.weight.data = to_torch(kweights[151])
    p_resnet50.layer3[1].bn3.bias.data = to_torch(kweights[152])
    p_resnet50.layer3[1].bn3.running_mean.data = to_torch(kweights[153])
    p_resnet50.layer3[1].bn3.running_var.data = to_torch(kweights[154])


    # layer3.2
    # layer3.2.conv1: 155 conv4_3_1x1_reduce/kernel:0 (1, 1, 1024, 256)
    p_resnet50.layer3[2].conv1.weight.data = transp_torch(kweights[155])

    # layer3.2.bn1
    # 156 conv4_3_1x1_reduce/bn/gamma:0 (256,)
    # 157 conv4_3_1x1_reduce/bn/beta:0 (256,)
    # 158 conv4_3_1x1_reduce/bn/moving_mean:0 (256,)
    # 159 conv4_3_1x1_reduce/bn/moving_variance:0 (256,)
    p_resnet50.layer3[2].bn1.weight.data = to_torch(kweights[156])
    p_resnet50.layer3[2].bn1.bias.data = to_torch(kweights[157])
    p_resnet50.layer3[2].bn1.running_mean.data = to_torch(kweights[158])
    p_resnet50.layer3[2].bn1.running_var.data = to_torch(kweights[159])

    # layer3.2.conv2: 160 conv4_3_3x3/kernel:0 (3, 3, 256, 256)
    p_resnet50.layer3[2].conv2.weight.data = transp_torch(kweights[160])

    # layer3.2.bn2
    # 161 conv4_3_3x3/bn/gamma:0 (256,)
    # 162 conv4_3_3x3/bn/beta:0 (256,)
    # 163 conv4_3_3x3/bn/moving_mean:0 (256,)
    # 164 conv4_3_3x3/bn/moving_variance:0 (256,)
    p_resnet50.layer3[2].bn2.weight.data = to_torch(kweights[161])
    p_resnet50.layer3[2].bn2.bias.data = to_torch(kweights[162])
    p_resnet50.layer3[2].bn2.running_mean.data = to_torch(kweights[163])
    p_resnet50.layer3[2].bn2.running_var.data = to_torch(kweights[164])

    # layer3.2.conv3: 165 conv4_3_1x1_increase/kernel:0 (1, 1, 256, 1024)
    p_resnet50.layer3[2].conv3.weight.data = transp_torch(kweights[165])

    # layer3.2.bn3
    # 166 conv4_3_1x1_increase/bn/gamma:0 (1024,)
    # 167 conv4_3_1x1_increase/bn/beta:0 (1024,)
    # 168 conv4_3_1x1_increase/bn/moving_mean:0 (1024,)
    # 169 conv4_3_1x1_increase/bn/moving_variance:0 (1024,)
    p_resnet50.layer3[2].bn3.weight.data = to_torch(kweights[166])
    p_resnet50.layer3[2].bn3.bias.data = to_torch(kweights[167])
    p_resnet50.layer3[2].bn3.running_mean.data = to_torch(kweights[168])
    p_resnet50.layer3[2].bn3.running_var.data = to_torch(kweights[169])

    # layer3.3
    # layer3.3.conv1: 170 conv4_4_1x1_reduce/kernel:0 (1, 1, 1024, 256)
    p_resnet50.layer3[3].conv1.weight.data = transp_torch(kweights[170])

    # layer3.3.bn1
    # 171 conv4_4_1x1_reduce/bn/gamma:0 (256,)
    # 172 conv4_4_1x1_reduce/bn/beta:0 (256,)
    # 173 conv4_4_1x1_reduce/bn/moving_mean:0 (256,)
    # 174 conv4_4_1x1_reduce/bn/moving_variance:0 (256,)
    p_resnet50.layer3[3].bn1.weight.data = to_torch(kweights[171])
    p_resnet50.layer3[3].bn1.bias.data = to_torch(kweights[172])
    p_resnet50.layer3[3].bn1.running_mean.data = to_torch(kweights[173])
    p_resnet50.layer3[3].bn1.running_var.data = to_torch(kweights[174])

    # layer3.3.conv2: 175 conv4_4_3x3/kernel:0 (3, 3, 256, 256)
    p_resnet50.layer3[3].conv2.weight.data = transp_torch(kweights[175])

    # layer3.3.bn2
    # 176 conv4_4_3x3/bn/gamma:0 (256,)
    # 177 conv4_4_3x3/bn/beta:0 (256,)
    # 178 conv4_4_3x3/bn/moving_mean:0 (256,)
    # 179 conv4_4_3x3/bn/moving_variance:0 (256,)
    p_resnet50.layer3[3].bn2.weight.data = to_torch(kweights[176])
    p_resnet50.layer3[3].bn2.bias.data = to_torch(kweights[177])
    p_resnet50.layer3[3].bn2.running_mean.data = to_torch(kweights[178])
    p_resnet50.layer3[3].bn2.running_var.data = to_torch(kweights[179])

    # layer3.3.conv3: 180 conv4_4_1x1_increase/kernel:0 (1, 1, 256, 1024)
    p_resnet50.layer3[3].conv3.weight.data = transp_torch(kweights[180])

    # layer3.3.bn3
    # 181 conv4_4_1x1_increase/bn/gamma:0 (1024,)
    # 182 conv4_4_1x1_increase/bn/beta:0 (1024,)
    # 183 conv4_4_1x1_increase/bn/moving_mean:0 (1024,)
    # 184 conv4_4_1x1_increase/bn/moving_variance:0 (1024,)
    p_resnet50.layer3[3].bn3.weight.data = to_torch(kweights[181])
    p_resnet50.layer3[3].bn3.bias.data = to_torch(kweights[182])
    p_resnet50.layer3[3].bn3.running_mean.data = to_torch(kweights[183])
    p_resnet50.layer3[3].bn3.running_var.data = to_torch(kweights[184])


    # layer3.4
    # layer3.4.conv1: 185 conv4_5_1x1_reduce/kernel:0 (1, 1, 1024, 256)
    p_resnet50.layer3[4].conv1.weight.data = transp_torch(kweights[185])

    # layer3.4.bn1
    # 186 conv4_5_1x1_reduce/bn/gamma:0 (256,)
    # 187 conv4_5_1x1_reduce/bn/beta:0 (256,)
    # 188 conv4_5_1x1_reduce/bn/moving_mean:0 (256,)
    # 189 conv4_5_1x1_reduce/bn/moving_variance:0 (256,)
    p_resnet50.layer3[4].bn1.weight.data = to_torch(kweights[186])
    p_resnet50.layer3[4].bn1.bias.data = to_torch(kweights[187])
    p_resnet50.layer3[4].bn1.running_mean.data = to_torch(kweights[188])
    p_resnet50.layer3[4].bn1.running_var.data = to_torch(kweights[189])

    # layer3.4.conv2: 190 conv4_5_3x3/kernel:0 (3, 3, 256, 256)
    p_resnet50.layer3[4].conv2.weight.data = transp_torch(kweights[190])

    # layer3.4.bn2
    # 191 conv4_5_3x3/bn/gamma:0 (256,)
    # 192 conv4_5_3x3/bn/beta:0 (256,)
    # 193 conv4_5_3x3/bn/moving_mean:0 (256,)
    # 194 conv4_5_3x3/bn/moving_variance:0 (256,)
    p_resnet50.layer3[4].bn2.weight.data = to_torch(kweights[191])
    p_resnet50.layer3[4].bn2.bias.data = to_torch(kweights[192])
    p_resnet50.layer3[4].bn2.running_mean.data = to_torch(kweights[193])
    p_resnet50.layer3[4].bn2.running_var.data = to_torch(kweights[194])

    # layer3.4.conv3: 195 conv4_5_1x1_increase/kernel:0 (1, 1, 256, 1024)
    p_resnet50.layer3[4].conv3.weight.data = transp_torch(kweights[195])

    # layer3.4.bn3
    # 196 conv4_5_1x1_increase/bn/gamma:0 (1024,)
    # 197 conv4_5_1x1_increase/bn/beta:0 (1024,)
    # 198 conv4_5_1x1_increase/bn/moving_mean:0 (1024,)
    # 199 conv4_5_1x1_increase/bn/moving_variance:0 (1024,)
    p_resnet50.layer3[4].bn3.weight.data = to_torch(kweights[196])
    p_resnet50.layer3[4].bn3.bias.data = to_torch(kweights[197])
    p_resnet50.layer3[4].bn3.running_mean.data = to_torch(kweights[198])
    p_resnet50.layer3[4].bn3.running_var.data = to_torch(kweights[199])


    # layer3.5
    # layer3.5.conv1: 200 conv4_6_1x1_reduce/kernel:0 (1, 1, 1024, 256)
    p_resnet50.layer3[5].conv1.weight.data = transp_torch(kweights[200])

    # layer3.5.bn1
    # 201 conv4_6_1x1_reduce/bn/gamma:0 (256,)
    # 202 conv4_6_1x1_reduce/bn/beta:0 (256,)
    # 203 conv4_6_1x1_reduce/bn/moving_mean:0 (256,)
    # 204 conv4_6_1x1_reduce/bn/moving_variance:0 (256,)
    p_resnet50.layer3[5].bn1.weight.data = to_torch(kweights[201])
    p_resnet50.layer3[5].bn1.bias.data = to_torch(kweights[202])
    p_resnet50.layer3[5].bn1.running_mean.data = to_torch(kweights[203])
    p_resnet50.layer3[5].bn1.running_var.data = to_torch(kweights[204])

    # layer3.5.conv2: 205 conv4_6_3x3/kernel:0 (3, 3, 256, 256)
    p_resnet50.layer3[5].conv2.weight.data = transp_torch(kweights[205])

    # layer3.5.bn2
    # 206 conv4_6_3x3/bn/gamma:0 (256,)
    # 207 conv4_6_3x3/bn/beta:0 (256,)
    # 208 conv4_6_3x3/bn/moving_mean:0 (256,)
    # 209 conv4_6_3x3/bn/moving_variance:0 (256,)
    p_resnet50.layer3[5].bn2.weight.data = to_torch(kweights[206])
    p_resnet50.layer3[5].bn2.bias.data = to_torch(kweights[207])
    p_resnet50.layer3[5].bn2.running_mean.data = to_torch(kweights[208])
    p_resnet50.layer3[5].bn2.running_var.data = to_torch(kweights[209])

    # layer3.5.conv3: 210 conv4_6_1x1_increase/kernel:0 (1, 1, 256, 1024)
    p_resnet50.layer3[5].conv3.weight.data = transp_torch(kweights[210])

    # layer3.5.bn3
    # 211 conv4_6_1x1_increase/bn/gamma:0 (1024,)
    # 212 conv4_6_1x1_increase/bn/beta:0 (1024,)
    # 213 conv4_6_1x1_increase/bn/moving_mean:0 (1024,)
    # 214 conv4_6_1x1_increase/bn/moving_variance:0 (1024,)
    p_resnet50.layer3[5].bn3.weight.data = to_torch(kweights[211])
    p_resnet50.layer3[5].bn3.bias.data = to_torch(kweights[212])
    p_resnet50.layer3[5].bn3.running_mean.data = to_torch(kweights[213])
    p_resnet50.layer3[5].bn3.running_var.data = to_torch(kweights[214])


    # layer4 ===================================================================
    # layer4.0
    # layer4.0.conv1: 215 conv5_1_1x1_reduce/kernel:0 (1, 1, 1024, 512)
    p_resnet50.layer4[0].conv1.weight.data = transp_torch(kweights[215])

    # layer4.0.bn1
    # 216 conv5_1_1x1_reduce/bn/gamma:0 (512,)
    # 217 conv5_1_1x1_reduce/bn/beta:0 (512,)
    # 218 conv5_1_1x1_reduce/bn/moving_mean:0 (512,)
    # 219 conv5_1_1x1_reduce/bn/moving_variance:0 (512,)
    p_resnet50.layer4[0].bn1.weight.data = to_torch(kweights[216])
    p_resnet50.layer4[0].bn1.bias.data = to_torch(kweights[217])
    p_resnet50.layer4[0].bn1.running_mean.data = to_torch(kweights[218])
    p_resnet50.layer4[0].bn1.running_var.data = to_torch(kweights[219])

    # layer4.0.conv2: 220 conv5_1_3x3/kernel:0 (3, 3, 512, 512)
    p_resnet50.layer4[0].conv2.weight.data = transp_torch(kweights[220])

    # layer4.0.bn2
    # 221 conv5_1_3x3/bn/gamma:0 (512,)
    # 222 conv5_1_3x3/bn/beta:0 (512,)
    # 223 conv5_1_3x3/bn/moving_mean:0 (512,)
    # 224 conv5_1_3x3/bn/moving_variance:0 (512,)
    p_resnet50.layer4[0].bn2.weight.data = to_torch(kweights[221])
    p_resnet50.layer4[0].bn2.bias.data = to_torch(kweights[222])
    p_resnet50.layer4[0].bn2.running_mean.data = to_torch(kweights[223])
    p_resnet50.layer4[0].bn2.running_var.data = to_torch(kweights[224])

    # layer4.0.conv3: 225 conv5_1_1x1_increase/kernel:0 (1, 1, 512, 2048)
    p_resnet50.layer4[0].conv3.weight.data = transp_torch(kweights[225])

    # layer4.0.bn3
    # 227 conv5_1_1x1_increase/bn/gamma:0 (2048,)
    # 228 conv5_1_1x1_increase/bn/beta:0 (2048,)
    # 229 conv5_1_1x1_increase/bn/moving_mean:0 (2048,)
    # 230 conv5_1_1x1_increase/bn/moving_variance:0 (2048,)
    p_resnet50.layer4[0].bn3.weight.data = to_torch(kweights[227])
    p_resnet50.layer4[0].bn3.bias.data = to_torch(kweights[228])
    p_resnet50.layer4[0].bn3.running_mean.data = to_torch(kweights[229])
    p_resnet50.layer4[0].bn3.running_var.data = to_torch(kweights[230])


    # layer4.0.downsample
    # layer4.0.downsample.0 (conv): 226 conv5_1_1x1_proj/kernel:0 (1, 1, 1024, 2048)
    p_resnet50.layer4[0].downsample[0].weight.data = transp_torch(kweights[226])

    # layer4.0.downsample.1 (bn)
    # 231 conv5_1_1x1_proj/bn/gamma:0 (2048,)
    # 232 conv5_1_1x1_proj/bn/beta:0 (2048,)
    # 233 conv5_1_1x1_proj/bn/moving_mean:0 (2048,)
    # 234 conv5_1_1x1_proj/bn/moving_variance:0 (2048,)
    p_resnet50.layer4[0].downsample[1].weight.data = to_torch(kweights[231])
    p_resnet50.layer4[0].downsample[1].bias.data = to_torch(kweights[232])
    p_resnet50.layer4[0].downsample[1].running_mean.data = to_torch(
        kweights[233])
    p_resnet50.layer4[0].downsample[1].running_var.data = to_torch(
        kweights[234])

    # layer4.1
    # layer4.1.conv1: 235 conv5_2_1x1_reduce/kernel:0 (1, 1, 2048, 512)
    p_resnet50.layer4[1].conv1.weight.data = transp_torch(kweights[235])

    # layer4.1.bn1
    # 236 conv5_2_1x1_reduce/bn/gamma:0 (512,)
    # 237 conv5_2_1x1_reduce/bn/beta:0 (512,)
    # 238 conv5_2_1x1_reduce/bn/moving_mean:0 (512,)
    # 239 conv5_2_1x1_reduce/bn/moving_variance:0 (512,)
    p_resnet50.layer4[1].bn1.weight.data = to_torch(kweights[236])
    p_resnet50.layer4[1].bn1.bias.data = to_torch(kweights[237])
    p_resnet50.layer4[1].bn1.running_mean.data = to_torch(kweights[238])
    p_resnet50.layer4[1].bn1.running_var.data = to_torch(kweights[239])

    # layer4.1.conv2: 240 conv5_2_3x3/kernel:0 (3, 3, 512, 512)
    p_resnet50.layer4[1].conv2.weight.data = transp_torch(kweights[240])

    # layer4.1.bn2
    # 241 conv5_2_3x3/bn/gamma:0 (512,)
    # 242 conv5_2_3x3/bn/beta:0 (512,)
    # 243 conv5_2_3x3/bn/moving_mean:0 (512,)
    # 244 conv5_2_3x3/bn/moving_variance:0 (512,)
    p_resnet50.layer4[1].bn2.weight.data = to_torch(kweights[241])
    p_resnet50.layer4[1].bn2.bias.data = to_torch(kweights[242])
    p_resnet50.layer4[1].bn2.running_mean.data = to_torch(kweights[243])
    p_resnet50.layer4[1].bn2.running_var.data = to_torch(kweights[244])

    # layer4.1.conv3: 245 conv5_2_1x1_increase/kernel:0 (1, 1, 512, 2048)
    p_resnet50.layer4[1].conv3.weight.data = transp_torch(kweights[245])

    # layer4.1.bn3:
    # 246 conv5_2_1x1_increase/bn/gamma:0 (2048,)
    # 247 conv5_2_1x1_increase/bn/beta:0 (2048,)
    # 248 conv5_2_1x1_increase/bn/moving_mean:0 (2048,)
    # 249 conv5_2_1x1_increase/bn/moving_variance:0 (2048,)
    p_resnet50.layer4[1].bn3.weight.data = to_torch(kweights[246])
    p_resnet50.layer4[1].bn3.bias.data = to_torch(kweights[247])
    p_resnet50.layer4[1].bn3.running_mean.data = to_torch(kweights[248])
    p_resnet50.layer4[1].bn3.running_var.data = to_torch(kweights[249])

    # layer4.2
    # layer4.2.conv1: 250 conv5_3_1x1_reduce/kernel:0 (1, 1, 2048, 512)
    p_resnet50.layer4[2].conv1.weight.data = transp_torch(kweights[250])

    # layer4.2.bn1:
    # 251 conv5_3_1x1_reduce/bn/gamma:0 (512,)
    # 252 conv5_3_1x1_reduce/bn/beta:0 (512,)
    # 253 conv5_3_1x1_reduce/bn/moving_mean:0 (512,)
    # 254 conv5_3_1x1_reduce/bn/moving_variance:0 (512,)
    p_resnet50.layer4[2].bn1.weight.data = to_torch(kweights[251])
    p_resnet50.layer4[2].bn1.bias.data = to_torch(kweights[252])
    p_resnet50.layer4[2].bn1.running_mean.data = to_torch(kweights[253])
    p_resnet50.layer4[2].bn1.running_var.data = to_torch(kweights[254])


    # layer4.2.conv2: 255 conv5_3_3x3/kernel:0 (3, 3, 512, 512)
    p_resnet50.layer4[2].conv2.weight.data = transp_torch(kweights[255])

    # layer4.2.bn2
    # 256 conv5_3_3x3/bn/gamma:0 (512,)
    # 257 conv5_3_3x3/bn/beta:0 (512,)
    # 258 conv5_3_3x3/bn/moving_mean:0 (512,)
    # 259 conv5_3_3x3/bn/moving_variance:0 (512,)
    p_resnet50.layer4[2].bn2.weight.data = to_torch(kweights[256])
    p_resnet50.layer4[2].bn2.bias.data = to_torch(kweights[257])
    p_resnet50.layer4[2].bn2.running_mean.data = to_torch(kweights[258])
    p_resnet50.layer4[2].bn2.running_var.data = to_torch(kweights[259])

    # layer4.2.conv3: 260 conv5_3_1x1_increase/kernel:0 (1, 1, 512, 2048)
    p_resnet50.layer4[2].conv3.weight.data = transp_torch(kweights[260])

    # layer4.2.bn3
    # 261 conv5_3_1x1_increase/bn/gamma:0 (2048,)
    # 262 conv5_3_1x1_increase/bn/beta:0 (2048,)
    # 263 conv5_3_1x1_increase/bn/moving_mean:0 (2048,)
    # 264 conv5_3_1x1_increase/bn/moving_variance:0 (2048,)
    p_resnet50.layer4[2].bn3.weight.data = to_torch(kweights[261])
    p_resnet50.layer4[2].bn3.bias.data = to_torch(kweights[262])
    p_resnet50.layer4[2].bn3.running_mean.data = to_torch(kweights[263])
    p_resnet50.layer4[2].bn3.running_var.data = to_torch(kweights[264])


    # test:
    x = torch.rand(1, 3, 224, 224)
    p_resnet50(x)
    # save pytorch weights
    cpu = torch.device('cpu')
    dir_out = join(root_dir, f"pretrained-{constants.VGGFACE2}")
    os.makedirs(dir_out, exist_ok=True)
    path_file = join(dir_out, f'pytorch-vggface2_notop_resnet50.pt')
    torch.save(p_resnet50.to(cpu).state_dict(), path_file)
    print(f"Save pytorch weights of {constants.RESNET50} at {path_file}")



if __name__ == '__main__':
    # vggface_resnet()
    keras_to_pytorch()