import sys
from os.path import dirname, abspath
from functools import partial
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

root_dir = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(root_dir)

import dlib.dllogger as DLLogger
from dlib.configure import constants

from dlib.diverse_models.apvit_backbones_basebackbone import BaseBackbone
from dlib.diverse_models.apvit_backbones_modules_vit import Block
from dlib.diverse_models.apvit_backbones_modules_vit_pooling import PoolingBlock
from dlib.diverse_models.apvit_vit_layers import to_2tuple
from dlib.diverse_models.apvit_vit_layers import trunc_normal_
from dlib.diverse_models.apvit_vit_layers import resize_pos_embed_v2
from dlib.diverse_models.apvit_backbones_modules_vit_pooling import top_pool


__all__ = ['PoolingViT']


class LANet(nn.Module):
    def __init__(self, channel_num, ratio=16):
        super().__init__()
        assert channel_num % ratio == 0, f"input_channel{channel_num} must be exact division by ratio{ratio}"
        self.channel_num = channel_num
        self.ratio = ratio
        self.relu = nn.ReLU(inplace=True)

        self.LA_conv1 = nn.Conv2d(channel_num, int(channel_num / ratio), kernel_size=1)
        self.bn1 = nn.BatchNorm2d(int(channel_num / ratio))
        self.LA_conv2 = nn.Conv2d(int(channel_num / ratio), 1, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        LA = self.LA_conv1(x)
        LA = self.bn1(LA)
        LA = self.relu(LA)
        LA = self.LA_conv2(LA)
        LA = self.bn2(LA)
        LA = self.sigmoid(LA)
        return LA
        # LA = LA.repeat(1, self.channel_num, 1, 1)
        # x = x*LA

        # return x


class MeanAttention(nn.Module):
    def __init__(self):
        super(MeanAttention, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # bsz, d, h, w

        att = torch.mean(x, dim=1, keepdim=True)  # bsz, 1, h, w
        att = F.sigmoid(att)  # bsz, 1, h, w

        return att


class CnnAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, hid_att_dim: int = 768):
        super(CnnAttention, self).__init__()

        assert isinstance(embed_dim, int), type(embed_dim)
        assert embed_dim > 0, embed_dim

        assert isinstance(hid_att_dim, int), type(hid_att_dim)
        assert hid_att_dim > 0, hid_att_dim

        self.embed_dim = embed_dim
        self.hid_att_dim = hid_att_dim
        self.n_att = 1

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=hid_att_dim,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh(),
            nn.Conv2d(in_channels=hid_att_dim, out_channels=self.n_att,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # bsz, d, h, w
        b, c, h, w = x.shape

        att = self.attention(x)  # b, 1, h, w

        att = F.sigmoid(att)  # b, 1, h, w

        # softmax
        # att = F.softmax(att.contiguous().view(b, h * w), dim=1)  # b, h * w
        # att = att.contiguous().view(b, 1, h, w)

        return att


class CnnGatedAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, hid_att_dim: int = 768):
        super(CnnGatedAttention, self).__init__()

        assert isinstance(embed_dim, int), type(embed_dim)
        assert embed_dim > 0, embed_dim

        assert isinstance(hid_att_dim, int), type(hid_att_dim)
        assert hid_att_dim > 0, hid_att_dim

        self.embed_dim = embed_dim
        self.hid_att_dim = hid_att_dim
        self.n_att = 1

        self.attention_v = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=hid_att_dim,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh()
        )

        self.attention_u = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=hid_att_dim,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Sequential(
            nn.Conv2d(in_channels=hid_att_dim, out_channels=self.n_att,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # bsz, d, h, w
        b, c, h, w = x.shape

        att_u = self.attention_u(x)  # b, hiden, h, w
        att_v = self.attention_v(x)  # b, hiden, h, w
        att = self.attention_weights(att_u * att_v)  # b, 1, h, w

        # att = F.sigmoid(att)  # b, 1, h, w
        # softmax.
        # att = F.softmax(att.contiguous().view(b, h * w), dim=1)  # b, h * w
        # att = att.contiguous().view(b, 1, h, w)


        return att


class Attention(nn.Module):
    def __init__(self, embed_dim: int = 768, hid_att_dim: int = 768):
        super(Attention, self).__init__()

        assert isinstance(embed_dim, int), type(embed_dim)
        assert embed_dim > 0, embed_dim

        assert isinstance(hid_att_dim, int), type(hid_att_dim)
        assert hid_att_dim > 0, hid_att_dim

        self.embed_dim = embed_dim
        self.hid_att_dim = hid_att_dim
        self.n_att = 1

        self.attention = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.hid_att_dim,
                      bias=True),
            nn.Tanh(),
            nn.Linear(in_features=self.hid_att_dim, out_features=self.n_att,
                      bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # bsz, d, h, w
        b, c, h, w = x.shape

        x = x.permute(0, 2, 3, 1)  # bsz, h, w, d
        x = x.contiguous().view(b * h * w, c)  # bsz * h * w, d

        att = self.attention(x)  # bsz * h * w, 1
        att = att.contiguous().view(b, h * w)
        att = F.softmax(att, dim=1)  # b, h * w
        att = att.view(b, h, w, 1).permute(0, 3, 1, 2)  # b, 1, h, w

        return att


class GatedAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, hid_att_dim: int = 768):
        super(GatedAttention, self).__init__()

        assert isinstance(embed_dim, int), type(embed_dim)
        assert embed_dim > 0, embed_dim

        assert isinstance(hid_att_dim, int), type(hid_att_dim)
        assert hid_att_dim > 0, hid_att_dim

        self.embed_dim = embed_dim
        self.hid_att_dim = hid_att_dim
        self.n_att = 1

        self.attention_v = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.hid_att_dim,
                      bias=True),
            nn.Tanh()
        )

        self.attention_u = nn.Sequential(
            nn.Linear(in_features=self.embed_dim, out_features=self.hid_att_dim,
                      bias=True),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(in_features=self.hid_att_dim,
                                           out_features=self.n_att,
                                           bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, x.ndim  # bsz, d, h, w
        b, c, h, w = x.shape

        x = x.permute(0, 2, 3, 1)  # bsz, h, w, d
        x = x.contiguous().view(b * h * w, c)  # bsz * h * w, d

        # u
        att_u = self.attention_u(x)  # bsz * h * w, d'

        # v
        att_v = self.attention_v(x)  # bsz * h * w, d'

        # fusion
        att = self.attention_weights(att_u * att_v)  # bsz * h * w, 1

        att = att.contiguous().view(b, h * w)
        att = F.softmax(att, dim=1)  # b, h * w
        att = att.view(b, h, w, 1).permute(0, 3, 1, 2)  # b, 1, h, w

        return att


class PoolingViT(BaseBackbone):
    """

    """

    def __init__(self,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer_eps=1e-5,
                 freeze=False,
                 input_type='image',
                 pretrained=None,
                 in_channels=[],
                 patch_num=0,
                 attn_method=constants.ATT_SUM_ABS_1,
                 cnn_pool_config=None,
                 vit_pool_configs=None,
                 multi_head_fusion=False,
                 sum_batch_mean=False,
                 normalize_att=False,
                 apply_self_att=False,
                 hid_att_dim=128,
                 **kwargs
                 ):
        super().__init__()

        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)

        assert isinstance(normalize_att, bool), type(normalize_att)
        self.normalize_att = normalize_att

        assert isinstance(apply_self_att, bool), type(apply_self_att)
        self.apply_self_att = apply_self_att

        assert isinstance(hid_att_dim, int), type(hid_att_dim)
        assert hid_att_dim > 0, hid_att_dim

        self.hid_att_dim = hid_att_dim

        assert input_type == 'feature', 'Only suit for hybrid model'
        self.sum_batch_mean = sum_batch_mean
        if sum_batch_mean:
            self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.multi_head_fusion = multi_head_fusion
        self.num_heads = num_heads
        if multi_head_fusion:
            assert vit_pool_configs is None, 'MultiHeadFusion only support original ViT Block, by now'

        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.projs = nn.ModuleList(
            [nn.Conv2d(in_channels[i], embed_dim, 1, ) for i in
             range(len(in_channels))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim),
                                      requires_grad=True)
        self.patch_pos_embed = nn.Parameter(
            torch.zeros(1, patch_num, embed_dim), requires_grad=True)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim),
                                          requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attn_method = attn_method
        self.cnn_pool_config = cnn_pool_config

        assert attn_method in constants.ATT_METHODS, attn_method

        if attn_method == constants.ATT_LA:
            # self.attn_f = LANet(in_channels[-1], 16)
            self.attn_f = LANet(embed_dim, 16)

        elif attn_method == constants.ATT_SUM:
            self.attn_f = lambda x: torch.sum(x, dim=1).unsqueeze(1)
        elif attn_method == constants.ATT_SUM_ABS_1:
            self.attn_f = lambda x: torch.sum(torch.abs(x), dim=1).unsqueeze(1)
        elif attn_method == constants.ATT_SUM_ABS_2:
            self.attn_f = lambda x: torch.sum(torch.pow(torch.abs(x), 2),
                                              dim=1).unsqueeze(1)
        elif attn_method == constants.ATT_MAX:
            self.attn_f = lambda x: torch.max(x, dim=1)[0].unsqueeze(1)
        elif attn_method == constants.ATT_MAX_ABS_1:
            self.attn_f = lambda x: torch.max(torch.abs(x), dim=1)[0].unsqueeze(
                1)
        elif attn_method == constants.ATT_RAND:
            self.attn_f = lambda x: x[:,
                                    torch.randint(high=x.shape[1], size=(1,))[
                                        0], ...].unsqueeze(1)

        elif attn_method == constants.ATT_MEAN:
            self.attn_f = MeanAttention()

        elif attn_method == constants.ATT_PARAM_ATT:
            self.attn_f = CnnAttention(embed_dim=embed_dim,
                                       hid_att_dim=hid_att_dim)

        elif attn_method == constants.ATT_PARAM_G_ATT:
            self.attn_f = CnnGatedAttention(embed_dim=embed_dim,
                                            hid_att_dim=hid_att_dim)
        else:
            raise ValueError("Unknown attn_method")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                depth)]  # stochastic depth decay rule
        if vit_pool_configs is None:

            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer, head_fusion=multi_head_fusion,
                )
                for i in range(depth)])
        else:
            vit_keep_rates = vit_pool_configs['keep_rates']
            self.blocks = nn.ModuleList([
                PoolingBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer,
                    pool_config=dict(keep_rate=vit_keep_rates[i],
                                     **vit_pool_configs),
                )
                for i in range(depth)]
            )
        self.norm = norm_layer(embed_dim)

        self.s2_pooling = nn.MaxPool2d(kernel_size=2)

        if pretrained:
            self.init_weights(pretrained, patch_num)
        else:
            trunc_normal_(self.patch_pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        if freeze:
            self.apply(self._freeze_weights)

    def init_weights(self, pretrained, patch_num=0):

        assert os.path.isfile(pretrained), pretrained

        DLLogger.log(
            f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')

        if 'model' in state_dict:
            state_dict = state_dict['model']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']  # [1, 197, 768] for base
        patch_pos_embed = pos_embed[:, 1:, :]

        if patch_num != pos_embed.shape[1] - 1:
            DLLogger.log(
                f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
            pos_embed_new = resize_pos_embed_v2(patch_pos_embed, patch_num, 0)
        else:  # remove cls_token
            print('does not need to resize!')
            pos_embed_new = patch_pos_embed
        del state_dict['pos_embed']
        state_dict['patch_pos_embed'] = pos_embed_new
        state_dict['cls_pos_embed'] = pos_embed[:, 0, :].unsqueeze(1)

        if self.multi_head_fusion:
            # convert blocks.0.attn.qkv.weight to blocks.0.attn.qkv.0.weight
            num_groups = self.blocks[0].attn.group_number
            d = self.embed_dim // num_groups
            print('d', d)
            for k in list(state_dict.keys()):
                if k.startswith('blocks.'):
                    keys = k.split('.')
                    if not (keys[2] == 'attn' and keys[3] == 'qkv'):
                        continue
                    for i in range(num_groups):
                        new_key = f'blocks.{keys[1]}.attn.qkv.{i}.weight'
                        new_value = state_dict[k][i * 3 * d:(i + 1) * 3 * d,
                                    i * d: i * d + d]
                        state_dict[new_key] = new_value

                    del state_dict[k]

        for k in (
        'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight',
        'head.bias'):
            del state_dict[k]
        self.load_state_dict(state_dict, strict=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _freeze_weights(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
        for param in m.parameters():
            param.requires_grad = False

    def forward_features(self, x):
        assert len(x) == 1, 'stage'
        assert isinstance(x, list) or isinstance(x, tuple)
        if len(x) == 2:  # S2, S3
            x[0] = self.s2_pooling(x[0])
        elif len(x) == 3:
            x[0] = nn.MaxPool2d(kernel_size=4)(x[0])
            x[1] = self.s2_pooling(x[1])


        x = [self.projs[i](x[i]) for i in range(len(x))]
        # x = x[0]
        B, C, H, W = x[-1].shape
        attn_map = self.attn_f(x[-1])  # B, 1, H, W
        # normalize: -----------------------------------------------------------
        # zz = attn_map.contiguous().view(B, -1).sum(dim=1).view(B, 1, 1, 1)
        # attn_map = attn_map / zz
        # ----------------------------------------------------------------------

        normed_attn = attn_map
        if self.normalize_att and self.attn_method not in[constants.ATT_LA]:
            zz = attn_map.contiguous().view(B, -1).sum(dim=1).view(B, 1, 1, 1)
            normed_attn = attn_map / zz


        if self.attn_method == constants.ATT_LA or self.apply_self_att:
            x[-1] = x[-1] * normed_attn  # to have gradient.


        x = [i.flatten(2).transpose(2, 1) for i in x]
        # x = self.projs[0](x).flatten(2).transpose(2, 1)
        # disable the first row and columns
        # attn_map[:, :, 0, :] = 0.
        # attn_map[:, :, :, 0] = 0.
        attn_weight = attn_map.flatten(2).transpose(2, 1)

        # attn_weight = torch.rand(attn_weight.shape, device=attn_weight.device)

        x = torch.stack(x).sum(dim=0)  # S1 + S2 + S3
        x = x + self.patch_pos_embed

        B, N, C = x.shape

        if self.cnn_pool_config is not None:
            keep_indexes = top_pool(attn_weight, dim=C, **self.cnn_pool_config)
            if keep_indexes is not None:
                x = x.gather(dim=1, index=keep_indexes)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens
        # impl from Phil Wang, thanks

        cls_tokens = cls_tokens + self.cls_pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # (B, N, dim)
        x = x[:, 0]
        if self.sum_batch_mean:
            x = x + x.mean(dim=0) * self.alpha
        loss = dict()
        return x, loss, attn_map

    def forward(self, x, **kwargs):
        x, loss, attn_map = self.forward_features(x)
        return dict(x=x, loss=dict(VitDiv_loss=loss), attn_map=attn_map)


def test_poolingvit():
    from os.path import join

    sz = 112
    pretrained = join(root_dir, constants.PRETRAINED_WEIGHTS_DIR,
                      'vit_small_p16_224-15ec54c9.pth')

    vit_small = dict(img_size=112, patch_size=16, embed_dim=768, num_heads=8,
                     mlp_ratio=3, qkv_bias=False, norm_layer_eps=1e-6)

    model = PoolingViT(input_type='feature',
                       pretrained=pretrained,
                       patch_num=196,
                       in_channels=[256],
                       attn_method=constants.ATT_PARAM_G_ATT,
                       sum_batch_mean=False,
                       cnn_pool_config=dict(keep_num=160, exclude_first=False),
                       vit_pool_configs=dict(keep_rates=[1.] * 4 + [0.9] * 4,
                                             exclude_first=True,
                                             attn_method='SUM'),
                       # None by default
                       depth=8,
                       apply_self_att=True,
                       hid_att_dim=128,
                       **vit_small
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

    x = torch.randn(b, 256, 14, 14)
    x = x.to(DEVICE)
    out = model(**dict(x=(x,)))
    _out_x = out['x']
    print(out['x'].shape,
          out['loss'],
          out['attn_map'].shape)


def run_mean_attention():
    device = torch.device('cuda:0')
    c = 768
    model = MeanAttention().to(device)
    x = torch.rand(32, c, 14, 14).to(device)
    att = model(x)

    print(x.shape, att.shape)


def run_cnn_attention():
    device = torch.device('cuda:0')
    c = 768
    model = CnnAttention(c, 128).to(device)
    x = torch.rand(32, c, 14, 14).to(device)
    att = model(x)

    print(x.shape, att.shape)


def run_cnn_gated_attention():
    device = torch.device('cuda:0')
    c = 768
    model = CnnGatedAttention(c, 128).to(device)
    x = torch.rand(32, c, 14, 14).to(device)
    att = model(x)

    print(x.shape, att.shape)


def run_attention():
    device = torch.device('cuda:0')
    c = 768
    model = Attention(c, 128).to(device)
    x = torch.rand(32, c, 14, 14).to(device)
    att = model(x)

    print(x.shape, att.shape)


def run_gated_attention():
    device = torch.device('cuda:0')
    c = 768
    model = GatedAttention(c, 128).to(device)
    x = torch.rand(32, c, 14, 14).to(device)
    att = model(x)

    print(x.shape, att.shape)

if __name__ == "__main__":
    from dlib.dllogger import Verbosity
    from dlib.dllogger import ArbStdOutBackend

    from dlib.configure import constants

    log_backends = [ArbStdOutBackend(Verbosity.VERBOSE)]
    DLLogger.init_arb(backends=log_backends, master_pid=os.getpid())

    test_poolingvit()

    # run_attention()
    # run_gated_attention()
    # run_mean_attention()
    # run_cnn_attention()
    # run_cnn_gated_attention()