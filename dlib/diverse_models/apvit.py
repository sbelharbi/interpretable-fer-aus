import sys
from os.path import dirname, abspath, join
import copy

import torch
import torch.nn as nn

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

from dlib.configure import constants
from dlib.utils.shared import count_params

from dlib.diverse_models.apvit_classifiers_base import BaseClassifier
from dlib.diverse_models.irse import IR_50
from dlib.diverse_models.apvit_vit_vit_siam_merge import PoolingViT
from dlib.diverse_models.apvit_heads_linear_head import LinearClsHead
from dlib.diverse_models.apvit_heads_linear_head import MultiLinearClsHead


# credit: https://github.com/youqingxiaozhua/APViT

__all__ = ['PoolingVitClassifier']


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

# hard-coded config: https://github.com/youqingxiaozhua/APViT

EXTRACTOR_CONFIG = dict(
    type='IRSE',
    input_size=(112, 112),
    num_layers=50,
    pretrained=join(root_dir, constants.PRETRAINED_WEIGHTS_DIR,
                    'backbone_ir50_ms1m_epoch120.pth'),
    mode='ir',
    return_index=[2],   # only use the first 3 stages
    return_type='Tuple',
    )

VIT_SMALL = dict(img_size=112,
                 patch_size=16,
                 embed_dim=768,
                 num_heads=8,
                 mlp_ratio=3,
                 qkv_bias=False,
                 norm_layer_eps=1e-6
                 )

VIT_CONFIG = dict(
    type='PoolingViT',
    pretrained=join(root_dir, constants.PRETRAINED_WEIGHTS_DIR,
                    'vit_small_p16_224-15ec54c9.pth'),
    input_type='feature',
    patch_num=196,
    in_channels=[256],
    attn_method=constants.ATT_SUM_ABS_1,  # more options.
    sum_batch_mean=False,
    depth=8,
    normalize_att=False,  # new
    apply_self_att=False,  # new
    hid_att_dim=128,  # new
    **VIT_SMALL,

)

def build_extractor(extractor_config: dict):

    assert extractor_config['type'] == 'IRSE', extractor_config['type']
    assert extractor_config['num_layers'] == 50, extractor_config['num_layers']
    assert extractor_config['mode'] == 'ir', extractor_config['mode']

    cnfg = copy.deepcopy(extractor_config)
    input_size = extractor_config['input_size']
    cnfg.pop('type', None)
    cnfg.pop('input_size', None)
    cnfg.pop('num_layers', None)
    cnfg.pop('mode', None)

    return IR_50(input_size=input_size, **cnfg)


def build_vit(vit_config: dict, k: int, r: float):
    assert isinstance(k, int), type(k)
    assert 0 <  k <= 196, k

    assert isinstance(r, float), type(r)
    assert 0. < r <= 1., r

    assert vit_config['type'] == 'PoolingViT', vit_config['type']

    vit_config.pop('type', None)

    cnn_pool_config = dict(keep_num=k, exclude_first=False)
    vit_pool_configs = dict(keep_rates=[1.] * 4 + [r] * 4, exclude_first=True,
                            attn_method='SUM')  # None by default

    config = copy.deepcopy(vit_config)
    config['cnn_pool_config'] = cnn_pool_config
    config['vit_pool_configs'] = vit_pool_configs

    model = PoolingViT(**config)
    return model

def build_head(num_classes: int, dense_dims: str = ''):
    assert isinstance(dense_dims, str), type(dense_dims)
    if dense_dims in ['', 'None']:
        return LinearClsHead(num_classes=num_classes, in_channels=768)
    else:
        z = dense_dims.split('-')  # accept manydense layer (>=1).
        # eg. 512-256-128.
        z = [int(i) for i in z]
        assert len(z) > 0, f"{len(z)}: {z}"

        return MultiLinearClsHead(num_classes=num_classes, in_channels=768,
                                  hidden_channels=z)


class PoolingVitClassifier(BaseClassifier):

    def __init__(self,
                 k: int = 160,
                 r: float = 0.9,
                 num_classes: int = 2,
                 dense_dims: str = '',
                 attn_method=constants.ATT_SUM_ABS_1,
                 normalize_att=False,
                 apply_self_att=False,
                 hid_att_dim=128,
                 pretrained=None,
                 freeze_backbone=False
                 ):
        super().__init__()

        self.encoder_name = constants.APVIT
        self.task = constants.STD_CL
        # self.name = constants.APVIT
        self.method = constants.METHOD_APVIT
        self.arch = constants.APVITCLASSIFIER

        self.name = "u-{}".format(self.encoder_name)

        self.extractor = build_extractor(EXTRACTOR_CONFIG)

        if freeze_backbone and (self.extractor is not None):
            print('freeze extractor backbone')
            self.extractor.eval()
            for param in self.extractor.parameters():
                param.requires_grad = False

        self.convert = None  # no convert.

        vit_config = copy.deepcopy(VIT_CONFIG)

        vit_config['attn_method'] = attn_method
        vit_config['normalize_att'] = normalize_att
        vit_config['apply_self_att'] = apply_self_att
        vit_config['hid_att_dim'] = hid_att_dim

        self.vit: nn.Module = build_vit(vit_config, k=k, r=r)

        self.neck = None  # no neck.

        # self.head = build_head(num_classes=num_classes, dense_dims=dense_dims)
        self.classification_head = build_head(num_classes=num_classes,
                                              dense_dims=dense_dims)

        self.features = []
        self.linear_features = None
        # self.linear_w = None
        self.att_maps = None
        self.cams = None

        self.init_weights(pretrained=pretrained)

    def flush(self):
        self.features = []
        self.linear_features = None
        # self.linear_w = None
        self.att_maps = None
        self.cams = None

    def get_info_nbr_params(self):
        info = self.__str__() + ' \n NBR-PARAMS: \n'
        if self.extractor:
            info += f'\tExtractor {count_params(self.extractor)}. \n'

        if self.vit:
            info += f'\tVIT: {count_params(self.vit)}. \n'

        if self.classification_head:
            info += f'\tHead: {count_params(self.classification_head)}. \n'

        info += '\tTotal: {}. \n'.format(count_params(self))

        return info

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        # self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.classification_head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        aux_loss = dict()
        if hasattr(self, 'extractor'):
            x = self.extractor(img)
        else:
            x = img
        if hasattr(self, 'convert'):
            x = self.convert(x)
        else:
            x = dict(x=x)
        x = self.vit(**x)
        if isinstance(x, dict):
            aux_loss.update(x['loss'])
            x = x['x']
        if self.with_neck:
            x = self.neck(x)
        return x, aux_loss

    def forward(self,
                im: torch.Tensor,
                labels: torch.Tensor = None
                ) -> torch.Tensor:

        self.features = []
        self.linear_features = None
        # self.linear_w = None
        self.att_maps = None

        x: tuple = self.extractor(im)
        x = dict(x=x)
        ft2d = x['x'][0]  # bsz, 256, 14, 14
        x: dict = self.vit(**x)
        assert isinstance(x, dict), type(x)
        f: torch.Tensor = x['x']
        attn_maps: torch.Tensor = x['attn_map']
        # attn_maps: bs, 1, 14, 14
        # print('att', attn_maps.min(), attn_maps.max())

        self.features = [im, ft2d, f]  # features at index 0 are not used by
        # convention in other arch such as resnet. f is dense features.
        self.linear_features = f
        self.att_maps = attn_maps
        self.cams = attn_maps.detach()

        x: torch.Tensor = self.classification_head(f)

        # self.linear_w = self.classification_head.get_linear_w()

        return x

    @property
    def linear_w(self):
        return self.classification_head.get_linear_w()

    def extract_attn_map(self, img):
        if hasattr(self, 'extractor'):
            x = self.extractor(img)
        else:
            x = img
        if hasattr(self, 'convert'):
            x = self.convert(x)
        else:
            x = dict(x=x)
        x = self.vit(**x)
        return x['attn_map']

    def forward_train(self, img, gt_label, au_label=None, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x, aux_loss = self.extract_feat(img)

        if au_label is None:
            losses = self.classification_head.forward_train(x, gt_label)
        else:
            losses = self.classification_head.forward_train(x, gt_label,
                                                            au_label)
        # losses['ce_loss'] = losses['loss']
        # losses['loss'] *= 0.
        # losses['aux_loss'] = aux_loss
        losses.update(aux_loss)

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x, _ = self.extract_feat(img)
        return self.classification_head.simple_test(x)

    def inference(self, img, **kwargs):
        x, _ = self.extract_feat(img)
        x = self.classification_head.extract_feat(x)
        return x

    def aug_test(self, imgs,
                 **kwargs):  # TODO: pull request: add aug test to mmcls
        logit = self.inference(imgs[0], **kwargs)
        for i in range(1, len(imgs)):
            cur_logit = self.inference(imgs[i])
            logit += cur_logit
        logit /= len(imgs)
        # pred = F.softmax(logit, dim=1)
        pred = logit
        pred = pred.cpu().numpy()
        # unravel batch dim
        pred = list(pred)
        return pred

    def __str__(self):
        return "{}. Task: {}.".format(self.name, self.task)

    def relprop(self, cam=None, method="transformer_attribution",
                is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.classification_head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.vit.pool.relprop(cam, **kwargs)
        cam = self.vit.norm.relprop(cam, **kwargs)
        for blk in reversed(self.vit.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.vit.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.vit.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.vit.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[
                    1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam

        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.vit.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            # cam = rollout[:, 1:, 0]
            return cam

        elif method == "last_layer":
            cam = self.vit.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.vit.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.vit.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.vit.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.vit.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam


def test_poolingvitclassifier():

    c = 7
    r = 0.9
    k = 160

    model = PoolingVitClassifier(k=k,
                                 r=r,
                                 num_classes=c
                                 )
    print("Testing {}".format(model.__class__.__name__))
    model.eval()
    print("Num. parameters: {}".format(
        sum([p.numel() for p in model.parameters()])))
    cuda_id = "0"
    DEVICE = torch.device(f'cuda:{cuda_id}')

    print("DEVICE AFTER: ", torch.cuda.current_device())
    # DEVICE = torch.device("cpu")
    model.to(DEVICE)

    b = 32

    x = torch.randn(b, 3, 112, 112)
    x = x.to(DEVICE)
    out = model(x)
    print(out.shape, c)


if __name__ == "__main__":
    import os

    import dlib.dllogger as DLLogger
    from dlib.dllogger import Verbosity
    from dlib.dllogger import ArbStdOutBackend


    log_backends = [ArbStdOutBackend(Verbosity.VERBOSE)]
    DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())

    test_poolingvitclassifier()