import torch
import torch.nn as nn
import torch.nn.functional as F
from .GDN import GDN
import math

def conv(in_channels, out_channels, kernel_size=3, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

def deconv(in_channels, out_channels, kernel_size=3, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )
# def Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding_mode='reflect'):
#     return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, padding_mode=padding_mode)


# def UpConv2d(in_channels, out_channels, kernel_size=5, stride=2):
#     return nn.ConvTranspose2d(
#         in_channels,
#         out_channels,
#         kernel_size=kernel_size,
#         stride=stride,
#         output_padding=stride - 1,
#         padding=kernel_size // 2,
#     )


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1), nn.PixelShuffle(r))


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class GDN1(GDN):

    def forward(self, x):
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        norm = F.conv2d(torch.abs(x), gamma, beta)

        if not self.inverse:
            norm = 1.0 / (norm + 1e-4)

        out = x * norm

        return out


class SFT(nn.Module):

    def __init__(self, x_nc, prior_nc=1, ks=3, nhidden=128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x, qmap):
        qmap = F.adaptive_avg_pool2d(qmap, x.size()[2:])
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = x * (1 + gamma) + beta

        return out


class SFTResblk(nn.Module):

    def __init__(self, x_nc, prior_nc, ks=3):
        super().__init__()
        self.conv_0 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(x_nc, x_nc, kernel_size=3, padding=1)

        self.norm_0 = SFT(x_nc, prior_nc, ks=ks)
        self.norm_1 = SFT(x_nc, prior_nc, ks=ks)

    def forward(self, x, qmap):
        dx = self.conv_0(self.actvn(self.norm_0(x, qmap)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, qmap)))
        out = x + dx

        return out

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class ResBlock_LeakyReLU_0_Point_1(nn.Module):

    def __init__(self, d_model):
        super(ResBlock_LeakyReLU_0_Point_1, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(d_model, d_model, 3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True),
                                  nn.Conv2d(d_model, d_model, 3, stride=1, padding=1), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = x + self.conv(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class RateEstNet(nn.Module):

    def __init__(self, channel_in):
        super(RateEstNet, self).__init__()
        self.h1 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.b1 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.h2 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.b2 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.h3 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.b3 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.h4 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.b4 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.a1 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.a2 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.a3 = nn.Parameter(torch.randn((1, channel_in, 1, 1), requires_grad=True))
        self.sp = torch.nn.Softplus()

    def forward(self, x):
        b, c, h, w = x.shape
        self.h1.repeat((b, 1, h, w))
        self.h2.repeat((b, 1, h, w))
        self.h3.repeat((b, 1, h, w))
        self.h4.repeat((b, 1, h, w))
        self.b1.repeat((b, 1, h, w))
        self.b2.repeat((b, 1, h, w))
        self.b3.repeat((b, 1, h, w))
        self.b4.repeat((b, 1, h, w))
        self.a1.repeat((b, 1, h, w))
        self.a2.repeat((b, 1, h, w))
        self.a3.repeat((b, 1, h, w))
        x = self.sp(self.h1).mul(x) + self.b1
        x = torch.tanh(self.a1).mul(torch.tanh(x)) + x
        x = self.sp(self.h2).mul(x) + self.b2
        x = torch.tanh(self.a2).mul(torch.tanh(x)) + x
        x = self.sp(self.h3).mul(x) + self.b3
        x = torch.tanh(self.a3).mul(torch.tanh(x)) + x
        x = self.sp(self.h4).mul(x) + self.b4
        x = torch.sigmoid(x)
        return x


class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        output = (input).round()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantization(nn.Module):

    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


class PixelShuffle(nn.Module):

    def __init__(self, scale):
        super(PixelShuffle, self).__init__()
        self.hscale = scale
        self.wscale = scale

    def forward(self, input):
        b, c, h, w = input.shape
        oc = c // (self.hscale * self.wscale)

        input = torch.reshape(input, (b, self.hscale, self.wscale, oc, h, w))
        input = input.permute((0, 3, 4, 1, 5, 2))
        input = torch.reshape(input, (b, oc, h * self.hscale, w * self.wscale))
        return input


class PixelInvShuffle(nn.Module):

    def __init__(self, scale):
        super(PixelInvShuffle, self).__init__()
        self.hscale = scale
        self.wscale = scale

    def forward(self, input):
        b, c, h, w = input.shape
        oh = h // self.hscale
        ow = w // self.wscale

        input = torch.reshape(input, (b, c, oh, self.hscale, ow, self.wscale))
        input = input.permute((0, 3, 5, 1, 2, 4))
        input = torch.reshape(input, (b, c * self.hscale * self.wscale, oh, ow))
        return input


def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)

    return grid


class WarpingLayer(nn.Module):

    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow, align):
        grid = torch.clip((get_grid(x).cuda() + flow).permute(0, 2, 3, 1), -1.0, 1.0)

        x_warp = F.grid_sample(x, grid, align_corners=align)

        return x_warp


class InvResBlock(nn.Module):

    def __init__(self, channel_in, channel_out, bias=True):
        super(InvResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, channel_out, 1, 1, 0, bias=bias)

        self.conv2 = nn.Conv2d(channel_out, channel_out, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_out, channel_out, 1, 1, 0, bias=bias)

        self.conv4 = nn.Conv2d(channel_out, channel_out, 3, 1, 1, bias=bias)

        self.conv5 = nn.Conv2d(channel_in, channel_out, 1, 1, 0, bias=bias)

        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv2(self.relu(self.conv1(x))))
        x2 = self.relu(self.conv4(self.relu(self.conv3(x1))))
        x3 = self.conv5(x) + x2
        return x3


class InvBlockPredTran(nn.Module):

    def __init__(self, channel_num):
        super(InvBlockPredTran, self).__init__()

        self.channel_num = channel_num

        self.P1 = InvResBlock(self.channel_num, self.channel_num)
        self.P2 = InvResBlock(self.channel_num * 2, self.channel_num)
        self.P3 = InvResBlock(self.channel_num * 3, self.channel_num)

        self.C = InvResBlock(self.channel_num * 3, self.channel_num)

    def forward(self, x, rev=False):
        if not rev:
            b, c, h, w = x.shape
            oh = h // 2
            ow = w // 2
            oc = c * 4
            x = torch.reshape(x, (b, c, oh, 2, ow, 2))
            x = x.permute((0, 3, 5, 1, 2, 4))
            x = torch.reshape(x, (b, oc, oh, ow))

            x1, x2, x3, x4 = (x.narrow(1, 0, self.channel_num), x.narrow(1, self.channel_num, self.channel_num),
                              x.narrow(1, 2 * self.channel_num, self.channel_num), x.narrow(1, 3 * self.channel_num, self.channel_num))
            y2 = x2 - self.P1(x1)
            y3 = x3 - self.P2(torch.cat((x1, x2), 1))
            y4 = x4 - self.P3(torch.cat((x1, x2, x3), 1))
            y1 = x1 + self.C(torch.cat((y2, y3, y4), 1))

            out = torch.cat((y1, y2, y3, y4), 1)

        else:
            x1, x2, x3, x4 = (x.narrow(1, 0, self.channel_num), x.narrow(1, self.channel_num, self.channel_num),
                              x.narrow(1, 2 * self.channel_num, self.channel_num), x.narrow(1, 3 * self.channel_num, self.channel_num))
            y1 = x1 - self.C(torch.cat((x2, x3, x4), 1))
            y2 = x2 + self.P1(y1)
            y3 = x3 + self.P2(torch.cat((y1, y2), 1))
            y4 = x4 + self.P3(torch.cat((y1, y2, y3), 1))
            out = torch.cat((y1, y2, y3, y4), 1)

            b, c, h, w = out.shape

            oh = h * 2
            ow = w * 2
            oc = c // 4
            out = torch.reshape(out, (b, 2, 2, oc, h, w))
            out = out.permute((0, 3, 4, 1, 5, 2))
            out = torch.reshape(out, (b, oc, oh, ow))

        return out


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=3, center_frame_idx=1):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)