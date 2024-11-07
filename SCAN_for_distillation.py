
import torch.nn as nn
from nni.compression.pytorch.utils.counter import count_flops_params
import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CALayer_noSigmoid(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer_noSigmoid, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class SCA(nn.Module):
    def __init__(self, dim):
        super(SCA, self).__init__()

        self.conv_7 = nn.Sequential(nn.Conv2d(dim, dim, (15, 1), 1, (7, 0), groups=dim),
                                    nn.Conv2d(dim, dim, (1, 15), 1, (0, 7), groups=dim))
        self.mixer = nn.Conv2d(dim, dim, 1)
    def forward(self, x):
        c7 = self.conv_7(x)
        add = x + c7
        output = self.mixer(add)
        return output

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sca = SCA(dim=dim)
    def forward(self, x):
        u = x.clone()
        attn = self.sca(x)
        return u * attn

class SCAB(nn.Module):
    def __init__(self, d_model, d_atten):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_atten, 1)
        self.activation = nn.GELU()
        self.conv = nn.Sequential(nn.Conv2d(d_atten, d_atten, 1, 1),
                                  nn.Conv2d(d_atten, d_atten, 3, 1, 1, groups=d_atten))
        self.atten_branch = Attention(d_atten)
        self.proj_2 = nn.Conv2d(d_atten, d_model, 1)
        self.pixel_norm = nn.LayerNorm(d_model)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.conv(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut
        x = x.permute(0, 2, 3, 1) #(B, H, W, C)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() #(B, C, H, W)

        return x

def make_layer(block, n_layers, *kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(*kwargs))
    return nn.Sequential(*layers)

class BasicLayer(nn.Module):
    def __init__(self, d_model, d_atten, depth, reduction):
        super().__init__()
        self.body = make_layer(SCAB, depth, d_model, d_atten)
        self.reduction = reduction

        self.proj = CALayer_noSigmoid(channel=d_model, reduction=16)
    def forward(self, x):
        residual = x.clone()

        x = self.body(x)

        inter_features = self.proj(x + residual)
        return x + residual, inter_features

class MyCustomModule(nn.Module):
    def __init__(self, module):
        super(MyCustomModule, self).__init__()
        self.module = module

    def forward(self, x):
        feat = []
        for layer in self.module:
            x, inter_feat = layer(x)
            feat.append(inter_feat)
        return x, feat

def pixelshuffle(in_channels, out_channels, upscale_factor=4):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(2)
    upconv2 = nn.Conv2d(16, out_channels * 4, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, pixel_shuffle, lrelu, upconv2, pixel_shuffle])

#both scale X2 and X3 use this version
def pixelshuffle_single(in_channels, out_channels, upscale_factor=2):
    upconv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    upconv2 = nn.Conv2d(64, out_channels * upscale_factor * upscale_factor, 3, 1, 1)
    lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.Sequential(*[upconv1, lrelu, upconv2, pixel_shuffle])

class SCAN(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale, depths, num_feat=64, d_atten=64, conv_groups=1):
        super(SCAN, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=conv_groups)
        self.depths = depths
        self.scale = scale
        if num_in_ch == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)


        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.upconv1 = nn.Conv2d(num_feat, 64, 3, 1, 1)
        self.upconv2 = nn.Conv2d(64, num_out_ch * self.scale * self.scale, 3, 1, 1)

        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                d_model=num_feat,
                d_atten=d_atten,
                depth=depths[i_layer],
                reduction=16
            )
            self.layers.append(layer)

        self.layers = MyCustomModule(self.layers)

    def forward(self, feat):
        self.mean = self.mean.type_as(feat)
        feat = (feat - self.mean)
        feat = self.conv_first(feat)
        body_feat, intermedia_feat = self.layers(feat)
        body_out = self.conv_body(body_feat)
        feat = feat + body_out

        # upsample
        feat = self.upconv1(feat)
        feat = self.lrelu(feat)
        feat = self.upconv2(feat)
        feat = self.pixel_shuffle(feat)
        return feat + self.mean, intermedia_feat
        # return feat, contrast_feat

if __name__ == '__main__':
    x = torch.rand((1, 3, 64, 64))
    model = SCAN(num_in_ch=3, num_out_ch=3, scale=3, num_feat=48, depths=[2,2,2,2,2,2,2], d_atten=64, conv_groups=2)

    flops, params, results = count_flops_params(model, x)
    x, co_x = model(x)
    print(x.shape, co_x[0].shape, co_x[1].shape, co_x[2].shape, co_x[3], co_x[4], co_x[5], co_x[6].shape)