import sys
from os.path import dirname, abspath
from typing import Optional, Union, List

import torch
from torch.cuda.amp import autocast

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.encoders import get_encoder
from dlib.base import STDClModel

from dlib import poolings

from dlib.configure import constants
from dlib.base import SegmentationHead


class STDClassifier(STDClModel):
    """
    Standard classifier.
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        aux_params: Optional[dict] = None,
        scale_in: float = 1.,
        spatial_dropout: float = 0.0,
        large_maps: bool = False,
        use_adl: bool = False,
        adl_drop_rate: float = .4,
        adl_drop_threshold: float = .1,
        apply_self_atten: bool = False,
        atten_layers_id: list = None,
        do_segmentation: bool = False
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.task = constants.STD_CL
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        assert isinstance(spatial_dropout, float), spatial_dropout
        assert 0. <= spatial_dropout <= 1., spatial_dropout
        self.p_dropout2d = spatial_dropout

        self.x_in = None

        assert isinstance(use_adl, bool), type(use_adl)

        assert isinstance(adl_drop_rate, float), type(adl_drop_rate)
        assert 0 <= adl_drop_rate <= 1., adl_drop_rate

        assert isinstance(adl_drop_threshold, float), type(adl_drop_threshold)
        assert 0 <= adl_drop_threshold <= 1., adl_drop_threshold
        self.use_adl = use_adl
        self.adl_drop_rate = adl_drop_rate
        self.adl_threshold = adl_drop_threshold

        self.apply_self_atten = apply_self_atten
        self.atten_layers_id = atten_layers_id

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            large_maps=large_maps,  # todo: add to vgg and inception.
            # adl
            use_adl=use_adl,
            adl_drop_rate=adl_drop_rate,
            adl_drop_threshold=adl_drop_threshold,
            # self attention
            apply_self_atten=apply_self_atten,
            atten_layers_id=atten_layers_id
        )

        self.dropout_2d = torch.nn.Dropout2d(p=self.p_dropout2d,
                                             inplace=False)

        assert aux_params is not None
        pooling_head = aux_params['pooling_head']
        aux_params.pop('pooling_head')
        encoder_expansion = 1
        if hasattr(self.encoder, 'expansion'):
            encoder_expansion = self.encoder.expansion

        self.classification_head = poolings.__dict__[pooling_head](
            in_channels=self.encoder.out_channels[-1],
            encoder_expansion=encoder_expansion,
            **aux_params
        )

        assert isinstance(do_segmentation, bool), type(do_segmentation)
        self.do_segmentation = do_segmentation

        self.segmentation_head = None

        if do_segmentation:
            self.segmentation_head = SegmentationHead(
                in_channels=self.encoder.out_channels[-1],
                out_channels=1,
                activation=None,
                kernel_size=3
            )

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        out = super(STDClassifier, self).forward(x, labels)

        self.features = self.features + self.classification_head.features

        return out

    def freeze_classification_head(self):

        for module in (self.classification_head.modules()):

            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

    def flush(self):

        super(STDClassifier, self).flush()

        if hasattr(self.classification_head, 'flush'):
            self.classification_head.flush()

        if self.segmentation_head is not None:
            if hasattr(self.segmentation_head, 'flush'):
                self.segmentation_head.flush()



def findout_names(model, architecture):
    def string_contains_any(string, substring_list):
        for substring in substring_list:
            if substring in string:
                return True
        return False

    _FEATURE_PARAM_LAYER_PATTERNS = {
        'vgg': ['encoder.features.'],  # features
        'resnet': ['encoder.layer4.', 'classification_head.'],  # CLASSIFIER
        'inception': ['encoder.Mixed', 'encoder.Conv2d_1', 'encoder.Conv2d_2',
                      'encoder.Conv2d_3', 'encoder.Conv2d_4'],  # features
    }

    param_features = []
    param_classifiers = []

    def param_features_substring_list(architecture):
        for key in _FEATURE_PARAM_LAYER_PATTERNS:
            if architecture.startswith(key):
                return _FEATURE_PARAM_LAYER_PATTERNS[key]
        raise KeyError("Fail to recognize the architecture {}"
                       .format(architecture))

    for name, parameter in model.named_parameters():

        if string_contains_any(
                name,
                param_features_substring_list(architecture)):
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                # param_features.append(parameter)
                print(name, '==>', 'feature')
            elif architecture in [constants.RESNET18, constants.RESNET34,
                                  constants.RESNET50, constants.RESNET101,
                                  constants.RESNET152]:
                # param_classifiers.append(parameter)
                print(name, '==>', 'classifier')
        else:
            if architecture in (constants.VGG16, constants.INCEPTIONV3):
                # param_classifiers.append(parameter)
                print(name, '==>', 'classifier')
            elif architecture in [constants.RESNET18, constants.RESNET34,
                                  constants.RESNET50, constants.RESNET101,
                                  constants.RESNET152]:
                # param_features.append(parameter)
                print(name, '==>', 'feature')


if __name__ == "__main__":
    import datetime as dt
    import dlib
    from dlib.utils.shared import announce_msg
    from dlib.utils.reproducibility import set_seed
    import dlib.dllogger as DLLogger

    log_backends = []
    DLLogger.init_arb(backends=log_backends, master_pid=0)

    set_seed(0)
    cuda = "0"
    DEVICE = torch.device(
        "cuda:{}".format(cuda) if torch.cuda.is_available() else "cpu")

    encoders = dlib.encoders.get_encoder_names()
    vgg_encoders = dlib.encoders.vgg_encoders
    in_channels = 3
    SZ = 224
    sample = torch.rand((2, in_channels, SZ, SZ)).to(DEVICE)
    encoders = [constants.RESNET50, constants.INCEPTIONV3, constants.VGG16]
    encoders = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d',
                'resnext101_32x16d', 'resnext101_32x32d',
                'resnext101_32x48d']

    encoders = ['resnet50']

    amp = True

    large_maps = False

    use_adl = True
    adl_drop_rate = 0.4
    adl_drop_threshold = 0.1

    for encoder_name in encoders:

        announce_msg("Testing backbone {}".format(encoder_name))
        if encoder_name == constants.VGG16:
            encoder_depth = vgg_encoders[encoder_name]['params']['depth']
        else:
            encoder_depth = 5

        # task: STD_CL
        model = STDClassifier(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=constants.IMAGENET,
            # encoder_weights=None,
            in_channels=in_channels,
            aux_params=dict(pooling_head=constants.WGAP,
                            classes=3,
                            dense_dims='1024-512',
                            dense_dropout=0.1
                            ),
            large_maps=large_maps,
            use_adl=use_adl,
            adl_drop_rate=adl_drop_rate,
            adl_drop_threshold=adl_drop_threshold
        ).to(DEVICE)
        announce_msg("TESTING: {} -- amp={} \n {}".format(model, amp,
                     model.get_info_nbr_params()))
        t0 = dt.datetime.now()
        with autocast(enabled=amp):
            cl_logits = model(sample)
            for i, l in enumerate(model.features):
                print(f"Layer {i}. Feature shape {l.shape}")

        print('forward time {}'.format(dt.datetime.now() - t0))
        print("Num. parameters {}: {}".format(encoder_name,
            sum([p.numel() for p in model.parameters()])))

        #
        # print("x: {} \t cl_logits: {}".format(sample.shape, cl_logits.shape))
        # print('logits', cl_logits)
        # val, ind = torch.sort(cl_logits.cpu(), dim=1, descending=True,
        #                       stable=True)
        # print(val, ind)
        #
        # findout_names(model, encoder_name)
