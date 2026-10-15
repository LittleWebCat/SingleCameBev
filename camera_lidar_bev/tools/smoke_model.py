#!/usr/bin/env python3
"""Minimal inference-only port of the official SMOKE pretrained network.

Adapted from lzccccc/SMOKE commit bc5d2bba66e2d66fa56b7b599d55457cb1a05b33.
Copyright (c) 2020 lzccccc; MIT license in ../third_party/SMOKE/LICENSE.
This retains pretrained module names and operators; it does not train or download.
"""
from types import SimpleNamespace
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import deform_conv2d


def group_norm(channels):
    return nn.GroupNorm(32 if channels % 32 == 0 else 16, channels, eps=1e-5)


def _make_conv_level(in_channels, out_channels, num_convs, norm_func,
                     stride=1, dilation=1):
    modules = []
    for i in range(num_convs):
        modules.extend([nn.Conv2d(in_channels, out_channels, 3,
                                stride=stride if i == 0 else 1,
                                padding=dilation, bias=False, dilation=dilation),
                        norm_func(out_channels), nn.ReLU(inplace=True)])
        in_channels = out_channels
    return nn.Sequential(*modules)


def _fill_up_weights(up):
    # Initialized bilinear kernels are replaced by the strict pretrained load.
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - abs(i / f - c)) * (1 - abs(j / f - c))
    w[1:, 0] = w[0, 0]


class _Deform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, offset, mask, weight, bias):
        return deform_conv2d(x, offset, weight, bias, padding=(1, 1), mask=mask)

    @staticmethod
    def symbolic(g, x, offset, mask, weight, bias):
        # NVIDIA TensorRT 10.13 documented plugin: NCHW x,offset,mask,weight,bias.
        out = g.op('trt.plugins::ModulatedDeformConv2d', x, offset, mask, weight,
                   bias, stride_i=[1, 1], padding_i=[1, 1], dilation_i=[1, 1],
                   group_i=1, deformable_group_i=1, plugin_version_s='2',
                   plugin_namespace_s='')
        xs, ws, os = x.type().sizes(), weight.type().sizes(), offset.type().sizes()
        if xs is not None and ws is not None and os is not None:
            out.setType(x.type().with_sizes([xs[0], ws[0], os[2], os[3]]))
        return out


class DCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.conv_offset_mask = nn.Conv2d(in_channels, 27, 3, padding=1, bias=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        o1, o2, mask = torch.chunk(self.conv_offset_mask(x), 3, dim=1)
        # Preserve the original channel ordering; its CUDA kernel and torchvision
        # both index [h0,w0,h1,w1,...], despite the two chunk variable names.
        return _Deform.apply(x, torch.cat((o1, o2), dim=1), torch.sigmoid(mask),
                             self.weight, self.bias)


class DeformConv(nn.Module):
    def __init__(self, in_channel, out_channel, norm_func):
        super().__init__()
        self.norm = norm_func(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.deform_conv = DCN(in_channel, out_channel)

    def forward(self, x):
        return self.relu(self.norm(self.deform_conv(x)))

# -----------------------------------------------------------------------------
# DLA models
# -----------------------------------------------------------------------------
DLA34DCN = {
    "levels": [1, 1, 1, 2, 2, 1],
    "channels": [16, 32, 64, 128, 256, 512],
    "block": "BasicBlock"
}
# -----------------------------------------------------------------------------
_STAGE_SPECS = dict({
    "DLA-34-DCN": DLA34DCN,
})

_NORM_SPECS = dict({
    "BN": nn.BatchNorm2d,
    "GN": group_norm,
})


def get_base_model(model, norm_func):
    model = DLABase(levels=model["levels"],
                    channels=model["channels"],
                    block=eval(model["block"]),
                    norm_func=norm_func)
    return model


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func,
                 stride=1,
                 dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=dilation,
                               bias=False,
                               dilation=dilation)
        self.norm1 = norm_func(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=dilation,
                               bias=False,
                               dilation=dilation
                               )
        self.norm2 = norm_func(out_channels)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)

        return out


class Tree(nn.Module):
    def __init__(self,
                 level,
                 block,
                 in_channels,
                 out_channels,
                 norm_func,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False
                 ):
        super(Tree, self).__init__()

        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            root_dim += in_channels

        if level == 1:
            self.tree1 = block(in_channels,
                               out_channels,
                               norm_func,
                               stride,
                               dilation=dilation)

            self.tree2 = block(out_channels,
                               out_channels,
                               norm_func,
                               stride=1,
                               dilation=dilation)
        else:
            new_level = level - 1
            self.tree1 = Tree(new_level,
                              block,
                              in_channels,
                              out_channels,
                              norm_func,
                              stride,
                              root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)

            self.tree2 = Tree(new_level,
                              block,
                              out_channels,
                              out_channels,
                              norm_func,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation,
                              root_residual=root_residual)
        if level == 1:
            self.root = Root(root_dim,
                             out_channels,
                             norm_func,
                             root_kernel_size,
                             root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.level = level

        self.downsample = None
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=1,
                          bias=False),

                norm_func(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        if children is None:
            children = []

        if self.downsample:
            bottom = self.downsample(x)
        else:
            bottom = x

        if self.project:
            residual = self.project(bottom)
        else:
            residual = bottom

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)

        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class Root(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_func,
                 kernel_size,
                 residual):
        super(Root, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=1,
                              stride=1,
                              bias=False,
                              padding=(kernel_size - 1) // 2)

        self.norm = norm_func(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.norm(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class DLA(nn.Module):
    def __init__(self,
                 cfg,
                 last_level=5,
                 out_channel=0):
        super(DLA, self).__init__()

        down_ratio = cfg.MODEL.BACKBONE.DOWN_RATIO
        assert down_ratio in [2, 4, 8, 16]

        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level

        base_name = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]
        norm_func = _NORM_SPECS[cfg.MODEL.BACKBONE.USE_NORMALIZATION]
        channels = base_name['channels']

        # self.base = globals()[base_name]
        self.base = get_base_model(base_name, norm_func)

        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(startp=self.first_level,
                            channels=channels[self.first_level:],
                            scales=scales,
                            norm_func=norm_func)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        up_scales = [2 ** i for i in range(self.last_level - self.first_level)]
        self.ida_up = IDAUp(in_channels=channels[self.first_level:self.last_level],
                            out_channel=out_channel,
                            up_f=up_scales,
                            norm_func=norm_func)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        # todo: this can be further cleaned
        return y[-1]


class DLABase(nn.Module):
    def __init__(self,
                 levels,
                 channels,
                 block=BasicBlock,
                 residual_root=False,
                 norm_func=nn.BatchNorm2d,
                 ):
        super(DLABase, self).__init__()

        self.channels = channels
        self.level_length = len(levels)

        self.base_layer = nn.Sequential(nn.Conv2d(3,
                                                  channels[0],
                                                  kernel_size=7,
                                                  stride=1,
                                                  padding=3,
                                                  bias=False),

                                        norm_func(channels[0]),

                                        nn.ReLU(inplace=True))

        self.level0 = _make_conv_level(in_channels=channels[0],
                                       out_channels=channels[0],
                                       num_convs=levels[0],
                                       norm_func=norm_func)

        self.level1 = _make_conv_level(in_channels=channels[0],
                                       out_channels=channels[1],
                                       num_convs=levels[0],
                                       norm_func=norm_func,
                                       stride=2)

        self.level2 = Tree(level=levels[2],
                           block=block,
                           in_channels=channels[1],
                           out_channels=channels[2],
                           norm_func=norm_func,
                           stride=2,
                           level_root=False,
                           root_residual=residual_root)

        self.level3 = Tree(level=levels[3],
                           block=block,
                           in_channels=channels[2],
                           out_channels=channels[3],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self.level4 = Tree(level=levels[4],
                           block=block,
                           in_channels=channels[3],
                           out_channels=channels[4],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

        self.level5 = Tree(level=levels[5],
                           block=block,
                           in_channels=channels[4],
                           out_channels=channels[5],
                           norm_func=norm_func,
                           stride=2,
                           level_root=True,
                           root_residual=residual_root)

    def forward(self, x):
        y = []
        x = self.base_layer(x)

        for i in range(self.level_length):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)

        return y


class DLAUp(nn.Module):
    def __init__(self,
                 startp,
                 channels,
                 scales,
                 in_channels=None,
                 norm_func=nn.BatchNorm2d):
        super(DLAUp, self).__init__()

        self.startp = startp

        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)

        scales = np.array(scales, dtype=int)

        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self,
                    'ida_{}'.format(i),
                    IDAUp(in_channels[j:],
                          channels[j],
                          scales[j:] // scales[j],
                          norm_func))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class IDAUp(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channel,
                 up_f,  # todo: what is up_f here?
                 norm_func):
        super(IDAUp, self).__init__()

        for i in range(1, len(in_channels)):
            in_channel = in_channels[i]
            f = int(up_f[i])
            proj = DeformConv(in_channel, out_channel, norm_func)
            node = DeformConv(out_channel, out_channel, norm_func)

            up = nn.ConvTranspose2d(out_channel,
                                    out_channel,
                                    kernel_size=f * 2,
                                    stride=f,
                                    padding=f // 2,
                                    output_padding=0,
                                    groups=out_channel,
                                    bias=False)
            _fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class SMOKEPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_head = nn.Sequential(nn.Conv2d(64, 256, 3, padding=1),
                                        group_norm(256), nn.ReLU(inplace=True),
                                        nn.Conv2d(256, 3, 1))
        self.regression_head = nn.Sequential(nn.Conv2d(64, 256, 3, padding=1),
                                             group_norm(256), nn.ReLU(inplace=True),
                                             nn.Conv2d(256, 8, 1))

    def forward(self, features):
        heatmap = torch.sigmoid(self.class_head(features)).clamp(1e-4, 1 - 1e-4)
        raw = self.regression_head(features)
        regression = torch.cat((raw[:, :3], torch.sigmoid(raw[:, 3:6]) - .5,
                                F.normalize(raw[:, 6:8], p=2, dim=1)), dim=1)
        return heatmap, regression


class SMOKE(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = SimpleNamespace(MODEL=SimpleNamespace(BACKBONE=SimpleNamespace(
            DOWN_RATIO=4, CONV_BODY='DLA-34-DCN', USE_NORMALIZATION='GN')))
        self.backbone = nn.Module()
        self.backbone.body = DLA(cfg)
        self.heads = nn.Module()
        self.heads.predictor = SMOKEPredictor()

    def forward(self, image):
        return self.heads.predictor(self.backbone.body(image))


def load_model(checkpoint, device='cuda'):
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = state['model']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model = SMOKE()
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def prepare_image(bgr):
    """Original uniform centered affine, native image -> fixed input NCHW.

    Input is undistorted native BGR. The 1280/width uniform scale crops tall
    cameras vertically. Return the native-to-network affine for metric decoding.
    """
    import cv2
    height, width = bgr.shape[:2]
    scale = 1280. / width
    affine = np.array([[scale, 0., 0.],
                       [0., scale, (384. - height * scale) / 2.]], np.float32)
    warped = cv2.warpAffine(bgr, affine, (1280, 384), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    chw = np.transpose(warped.astype(np.float32) / 255., (2, 0, 1))
    chw = (chw - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array([.229, .224, .225], np.float32)[:, None, None]
    return np.ascontiguousarray(chw[None]), affine
