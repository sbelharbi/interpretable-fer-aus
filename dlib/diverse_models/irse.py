import sys
from os.path import dirname, abspath, join
import os

import warnings
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, \
    Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple

import torch




root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib.dllogger as DLLogger
from dlib.diverse_models.apvit_backbones_basebackbone import BaseBackbone


# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class ABSPool2d(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        positive = MaxPool2d(*self.args, **self.kwargs)(x)
        negtive = MaxPool2d(*self.args, **self.kwargs)(-x)
        result = torch.where(positive >= negtive, positive, negtive)
        return result


pool_layer = MaxPool2d


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0,
            bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0,
            bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = pool_layer(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BasicBlockIR(Module):
    """
    BasicBlock for IRNet, stolen from TFace
    https://github.com/Tencent/TFace/blob/d57fd8d9ce9502240921f0998c57f84afa7
    eaeaa/torchkit/backbone/model_irse.py#L16
    Add a BatchNorm2d after the first Conv2d
    """

    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = pool_layer(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            BatchNorm2d(depth),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = pool_layer(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 8:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
        ]
    elif num_layers == 16:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
        ]
    elif num_layers == 34:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=6),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 44:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
        ]
    elif num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),  # (B, 64, 56, 56)
            get_block(in_channel=64, depth=128, num_units=4),
            # (B, 128, 28, 28)
            get_block(in_channel=128, depth=256, num_units=14),
            # (B, 256, 14, 14)
            get_block(in_channel=256, depth=512, num_units=3)  # (B, 512, 7, 7)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class IRSE(BaseBackbone):
    def __init__(self,
                 input_size,
                 num_layers,
                 mode='ir',
                 with_head=False,
                 pretrained=None,
                 return_index=(0, 1, 2),
                 return_type='Tuple'
                 ):
        super().__init__()
        msg = "input_size should be [112, 112] or [224, 224]"
        assert input_size[0] in [112, 224], msg
        msg = "num_layers should be 50, 100 or 152"
        assert num_layers in [0, 8, 16, 34, 44, 50, 100, 152], msg
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        assert return_type in (
        'Tensor', 'Tuple'), 'return_tensor must be Tensor or Tuple'
        if len(return_index) > 1:
            assert return_type == 'Tuple'
        self.num_layers = num_layers
        self.return_index = return_index
        self.return_type = return_type
        if num_layers == 0:
            return
        self.with_head = with_head
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            if num_layers == 34:
                warnings.warn('Using the IR_34 version from TFace')
                unit_module = BasicBlockIR
            else:
                unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        if with_head:
            if input_size[0] == 112:
                self.output_layer = Sequential(BatchNorm2d(512),
                                               Dropout(),
                                               Flatten(),
                                               Linear(512 * 7 * 7, 512),
                                               BatchNorm1d(512))
            else:
                self.output_layer = Sequential(BatchNorm2d(512),
                                               Dropout(),
                                               Flatten(),
                                               Linear(512 * 14 * 14, 512),
                                               BatchNorm1d(512))

        modules = []
        max_stage = max(return_index)
        for block in blocks[:max_stage + 1]:
            block_module = []
            for bottleneck in block:
                block_module.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
            modules.append(Sequential(*block_module))
        # self.body = Sequential(*modules)
        self.body = nn.ModuleList(modules)

        self._initialize_weights()
        if pretrained:
            self.init_weights(pretrained)

    def init_weights(self, pretrained):

        DLLogger.log(f'{self.__class__.__name__} load pretrain from '
                     f'{pretrained}')

        # model_dir = join(root_dir, f'pretrained-weights')
        # os.makedirs(model_dir, exist_ok=True)
        # state_dict = model_zoo.load_url(url=pretrained,
        #                                 model_dir=model_dir,
        #                                 map_location='cpu')

        assert os.path.isfile(pretrained), pretrained

        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        stage_unit_nums = {
            34: (3, 4, 6, 3),
            50: (3, 4, 14, 3),
        }
        stage_num = stage_unit_nums[self.num_layers]

        new_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith('body.'):
                index = int(k.split('.')[1])
                if 0 <= index < stage_num[0]:
                    new_key = k.replace('body.', 'body.0.')
                elif stage_num[0] <= index < sum(stage_num[:2]):
                    new_key = f"body.1.{index - sum(stage_num[:1])}.{'.'.join(k.split('.')[2:])}"
                elif sum(stage_num[:2]) <= index < sum(stage_num[:3]):
                    new_key = f"body.2.{index - sum(stage_num[:2])}.{'.'.join(k.split('.')[2:])}"
                else:
                    new_key = k
            else:
                new_key = k
            new_state_dict[new_key] = v

        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        if self.num_layers == 0:
            return x
        x = self.input_layer(x)
        output = []
        return_index = set(self.return_index)
        for index, m in enumerate(self.body):
            x = m(x)
            if index in return_index:
                output.append(x)
        if self.with_head:
            x = self.output_layer(x)
        if self.return_type == 'Tensor':
            return output[0]
        else:
            return tuple(output)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _freeze_stages(self):
        if self.frozen_blocks > 0:
            DLLogger.log(
                f'IRSE freeze the first {self.frozen_blocks} blocks, it has {len(self.body)} blocks ')
            self.input_layer.eval()
            DLLogger.log('in freeze', self.input_layer[1].training)
            for param in self.input_layer.parameters():
                param.requires_grad = False

        for i in range(self.frozen_blocks):
            m = self.body[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False


def IR_50(input_size, **kwargs):
    """Constructs a ir-50 model.
    """
    model = IRSE(input_size, 50, 'ir', **kwargs)

    return model


def IR_101(input_size, **kwargs):
    """Constructs a ir-101 model.
    """
    model = IRSE(input_size, 100, 'ir', **kwargs)

    return model


def IR_152(input_size, **kwargs):
    """Constructs a ir-152 model.
    """
    model = IRSE(input_size, 152, 'ir', **kwargs)

    return model


def IR_SE_50(input_size, **kwargs):
    """Constructs a ir_se-50 model.
    """
    model = IRSE(input_size, 50, 'ir_se', **kwargs)

    return model


def IR_SE_101(input_size, **kwargs):
    """Constructs a ir_se-101 model.
    """
    model = IRSE(input_size, 100, 'ir_se', **kwargs)

    return model


def IR_SE_152(input_size, **kwargs):
    """Constructs a ir_se-152 model.
    """
    model = IRSE(input_size, 152, 'ir_se', **kwargs)

    return model

def test_ir_50():

    sz = 112
    pretrained = join(root_dir, constants.PRETRAINED_WEIGHTS_DIR,
                      'backbone_ir50_ms1m_epoch120.pth')
    model = IR_50(input_size=(sz, sz), pretrained=pretrained,
                 return_index=[2],
                 return_type='Tuple'
                  )
    print("Testing {}".format(model.__class__.__name__))
    model.eval()
    print("Num. parameters: {}".format(sum([p.numel() for p in model.parameters()])))
    cuda_id = "0"
    DEVICE = torch.device(f'cuda:{cuda_id}')

    print("DEVICE AFTER: ", torch.cuda.current_device())
    # DEVICE = torch.device("cpu")
    model.to(DEVICE)

    b = 32

    x = torch.randn(b, 3, sz, sz)
    x = x.to(DEVICE)
    out = model(x)
    print(len(out), out[0].shape)  # 1, torch.Size([b, 256, 14, 14]

if __name__ == "__main__":
    from dlib.dllogger import Verbosity
    from dlib.dllogger import ArbStdOutBackend

    from dlib.configure import constants


    log_backends = [ArbStdOutBackend(Verbosity.VERBOSE)]
    DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())

    test_ir_50()