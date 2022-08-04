import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from .GDN import GDNEncoder, GDNDecoder
# from .analysis_prior import *
# from compressai.models import CompressionModel
# from compressai.entropy_models import EntropyBottleneck, GaussianConditional
# from compressai.models.utils import update_registered_buffers
from .layers import Win_noShift_Attention, ResidualBlockNoBN
from .utils import *

import argparse
import glob
import torchvision as tv
from torch import optim
import pickle
from PIL import Image
import time
import os

#from .han import HAN_Head as HAN
#from .han import MeanShift

from .Quant import BypassRound, Quant
from .Neural_Syntax import PredictionModel_Syntax, Syntax_Model, conv_generator, GaussianModel, NeighborSample, BlockSample, h_analysisTransformModel, h_synthesisTransformModel, Depth2Space, Space2Depth


class EDSR_arch(nn.Module):

    def __init__(self):
        super(EDSR_arch, self).__init__()
        n_resblocks = 16
        n_feats = 64
        kernel_size = 3
        scale = 2
        n_colors = 6
        act = nn.ReLU(True)
        self.M = 16
        self.N = 192

        # self.intraframe_downscaler = nn.Sequential(PixelInvShuffle(2),
        #                                 nn.Conv2d(4*n_colors, n_feats, kernel_size,padding=(kernel_size//2)),
        #                                 ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
        #                                 ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
        #                                 default_conv(n_feats, n_colors, 1))
        # self.refine_net = nn.Sequential(nn.Conv2d(n_colors, n_feats, kernel_size,padding=(kernel_size//2)),
        #                                 ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
        #                                 ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
        #                                 ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
        #                                 ResBlock(default_conv, n_feats, kernel_size, act=act, res_scale=1),
        #                                 default_conv(n_feats, n_colors, 1))
        
        ## Downscale part
        # self.intraframe_downscaler = nn.Sequential(
        #     PixelInvShuffle(2),
        #     nn.Conv2d(4 * n_colors, n_feats, kernel_size, padding=(kernel_size // 2)),
        # )
        self.intraframe_downscaler = PixelInvShuffle(2)
        # self.local_atten = nn.Sequential(
        #     nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)),
        #     Win_noShift_Attention(dim=n_feats, num_heads=8, window_size=8, shift_size=4),
        #     nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)),
        #     Win_noShift_Attention(dim=n_feats, num_heads=8, window_size=8, shift_size=4),
        #     nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2)),
        # )
        self.res_net = nn.Sequential(
            ResBlock(default_conv, 64, kernel_size, act=act, res_scale=1), # 64 -> n_feats + 16
            ResBlock(default_conv, 64, kernel_size, act=act, res_scale=1), # 64 -> n_feats + 16
            default_conv(64, n_colors, 1), # 64 -> n_feats + 16
        )
        self.GDN_forward = GDNEncoder(self.N, self.N, n_colors) #n_colors -> n_feats
        self.IGDN = GDNDecoder(self.N - self.M, self.N, self.M)

        # backward upscale part
        self.res_net_backward = nn.Sequential(
            default_conv(n_colors, 64, 1),
            ResBlock(default_conv, 64, kernel_size, act=act, res_scale=1), # 64 -> n_feats + 16
            ResBlock(default_conv, 64, kernel_size, act=act, res_scale=1), # 64 -> n_feats + 16
            default_conv(64, 64, 1),
        )
        self.GDN_backward = GDNEncoder(self.N, self.N - self.M, self.M)
        self.upscaler =  nn.PixelShuffle(2)

        ## SR part
        # m_body = [ResidualBlockNoBN(num_feat=n_feats, res_scale=1, pytorch_init=True) for _ in range(16)]
        # m_body.append(nn.Conv2d(n_feats, n_feats, 3, 1, 1))

        # self.conv_first = nn.Conv2d(n_colors, n_feats, 3, 1, 1)
        # self.conv_body = nn.Sequential(*m_body)
        # self.upsample = nn.Sequential(nn.Conv2d(n_feats, 4 * n_feats, 3, 1, 1), nn.PixelShuffle(2), nn.Conv2d(n_feats, n_colors, 3, 1, 1))
    def forward(self, x, up=True, flow=None, I=True):
        if up:
            if (I):
                '''x_I = self.I_head(x[:, :3, :, :])
                res_I = self.I_body(x_I)
                res_I += x_I
                x_warp = self.I_tail(res_I)'''
                # x_warp = torch.cat((self.intra_head(x[:, :3, :, :]), self.inter_head(x[:, :3, :, :])), dim=1)
                # res = self.body(x_warp)
                # res += x_warp
                # x_warp = self.tail(res)

                ## SR
                # x = self.conv_first(x)
                # res = self.conv_body(x)
                # res += x
                # x = self.upsample(x)
                # return x

                x = self.res_net_backward(x)
                up = self.upscaler(x)
                x = self.GDN_backward(up)
                x_tilde = self.IGDN(x)
                return x_tilde
            
            else:
                flow = flow.repeat_interleave(16, dim=1)
                flow = self.upshuffle(flow) / 4
                flow = flow[:, :, :x.size(2), :x.size(3)]
                flow[:, 0, :, :] = flow[:, 0, :, :] / ((flow.size(3) - 1) / 2.0)
                flow[:, 1, :, :] = flow[:, 1, :, :] / ((flow.size(2) - 1) / 2.0)

                predict_flow = self.refine_flow(
                    torch.cat(
                        (self.predict_flow(torch.cat(
                            (self.warp(self.flow_head(x[:, :3, :, :]), flow, True), self.flow_head(x[:, 3:6, :, :])), dim=1)), flow), 1))

                x_warp = torch.cat((self.warp(self.intra_head(x[:, :3, :, :]), predict_flow, False), self.inter_head(x[:, 3:6, :, :])), dim=1)
                res = self.body(x_warp)
                res += x_warp
                x_warp = self.tail(res)
            return x_warp
        else:
            b, c, h, w = x.shape
            y = self.GDN_forward(x) # B * 192 * H/32 * W/32
            y_syntax = y[:,:self.M,:,:] # B * 16 * H/32 * W/32
            y_content = y[:,self.M:,:,:] # B * 176 * H/32 * W/32
            cdf = self.IGDN(y_content) # B * 16 * H/2 * W/2
            down = self.intraframe_downscaler(cdf) # B * 64 * H/4 * W/4
            content = self.res_net(down) # B * 6 * H/4 * W/4
            return y, y_syntax, content

            # down = self.intraframe_downscaler(x)
            # cdf = self.GDN(down)
            # cdf = self.IGDN(cdf)
            # down = self.local_atten(down)
            # out = torch.cat((down, cdf), axis=1)
            # out = self.res_net(out)
            # return out
            # b, c, h, w = x.shape
            # x = self.intraframe_downscaler(x)
            # out = self.refine_net(x)
            # return out

class analysis_arch(nn.Module):

    def __init__(self):
        super(analysis_arch, self).__init__()
        z_feats = 64
        kernel_size = 3
        scale = 2
        n_colors = 6
        act = nn.ReLU(True)

        operations = []
        current_channel = n_colors * 4
        for j in range(3):
            b = InvBlockPredTran(current_channel)
            operations.append(b)
            current_channel *= 4
        self.enc_operations = nn.ModuleList(operations)

        operations = []
        current_channel = n_colors * 4
        for j in range(3):
            b = InvBlockPredTran(current_channel)
            operations.append(b)
            current_channel *= 4
        self.dec_operations = nn.ModuleList(operations)

        # self.rate_est = RateEstNet(current_channel)
        # self.rate_est_p = RateEstNet(current_channel)
        # self.Quant = Quantization()

        self.PixelInvShuffle = PixelInvShuffle(2)
        self.upshuffle = PixelShuffle(2)
        # self.deblock_net = nn.Sequential(nn.Conv2d(2 * n_colors, z_feats, kernel_size, padding=(kernel_size // 2)),
        #                                  ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
        #                                  ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
        #                                  ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1), default_conv(z_feats, 2 * n_colors, 1))
        # self.warp = WarpingLayer()

        self.deblock_net_i = nn.Sequential(nn.Conv2d(n_colors, z_feats, kernel_size, padding=(kernel_size // 2)),
                                           ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
                                           Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
                                           ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
                                           Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
                                           ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1), default_conv(z_feats, n_colors, 1))

        self.scale_net_i = nn.Sequential(nn.Conv2d(n_colors, z_feats, kernel_size, padding=(kernel_size // 2)),
                                         ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
                                         Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
                                         ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1),
                                         Win_noShift_Attention(dim=z_feats, num_heads=8, window_size=8, shift_size=4),
                                         ResBlock(default_conv, z_feats, kernel_size, act=act, res_scale=1), default_conv(z_feats, n_colors, 1))

    def forward(self, x, mv, I=True):
        b, c, h, w = x.shape
        num_pixels = b * h * w
        if (I):
            x = self.scale_net_i(x)
            x = self.PixelInvShuffle(x)
            for op in self.enc_operations:
                x = op(x, False)
            # x_rec = self.Quant(x)
            # x_enc = x + (torch.rand(x.size()).cuda() - 0.5)
            # prob = self.rate_est(x_enc + 0.5) - self.rate_est(x_enc - 0.5)
            # rates = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            # bpp = rates / num_pixels
            # x = x_enc
            bpp = None
            for op in reversed(self.dec_operations):
                x = op(x, True)
            x = self.upshuffle(x)
            x = self.deblock_net_i(x)

        else:
            x = self.PixelInvShuffle(x)

            mv = mv[:, 2:, :x.size(2), :x.size(3)]
            mv[:, 0, :, :] = mv[:, 0, :, :] / 16 / (mv.size(3) / 2.0)
            mv[:, 1, :, :] = mv[:, 1, :, :] / 16 / (mv.size(2) / 2.0)

            xI = x[:, :48, :, :]
            xP = x[:, 48:, :, :]

            xP = torch.cat((xP, self.warp(xI, mv)), dim=1)
            for op in self.p_enc_operations:
                xP = op(xP, False)
            xP = torch.cat((self.Quant(torch.clamp(xP[:, :768, :, :], -16, 16) * 255) / 255, xP[:, 768:, :, :]), 1)
            prob = self.rate_est_p(xP[:, :768, :, :] * 255 + 0.5) - self.rate_est_p(xP[:, :768, :, :] * 255 - 0.5)
            rates = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
            bpp = rates / num_pixels
            for op in reversed(self.p_dec_operations):
                xP = op(xP, True)
            xP = xP[:, :48, :, :]

            x = self.upshuffle(xP)
            x = self.deblock_net_p(x)
        return x, bpp


bypass_round = BypassRound.apply
class Net(nn.Module):
  def __init__(self): # train_size, test_size, is_high, post_processing
    super(Net, self).__init__()
    #self.train_size = train_size # B * 256 * 256 * 3
    #self.test_size = test_size # 1 * 256 * 256 * 3
    
    #self.post_processing = post_processing
    #self.is_high = is_high

    # if self.is_high:
    #   N = 384
    #   M = 32
    # else:
    #   N = 192
    #   M = 16
    N = 192
    M = 16
    self.M = M 
    self.N = N

    self.syntax_model = Syntax_Model(M,M)

    self.conv_weights_gen = conv_generator(in_dim=M,out_dim=M)

    self.ha_model = h_analysisTransformModel(N, [N, N, N], [1, 2, 2])
    self.hs_model = h_synthesisTransformModel(N, [N, N, N], [2, 2, 1])

    self.entropy_bottleneck_z = GaussianModel()
    self.entropy_bottleneck_y_syntax = GaussianModel()

    # b, h, w, c = train_size
    # tb, th, tw, tc = test_size


    self.v_z_sigma = nn.Parameter(torch.ones((1,N,1,1), dtype=torch.float32, requires_grad=True))
    self.register_parameter('z_sigma', self.v_z_sigma)
    
    self.prediction_model_syntax = PredictionModel_Syntax(in_dim=N, dim=M, outdim=M * 2)

    # self.y_sampler = BlockSample((b,N - M,h//8,w//8))
    # self.h_sampler = BlockSample((b, N,h//8,w//8), False)
    # self.test_y_sampler = BlockSample((b,N - M,th//8,tw//8))
    # self.test_h_sampler = BlockSample((b,N,th//8,tw//8), False)

    
    # self.HAN = HAN(is_high=self.is_high)
    # self.conv_weights_gen_HAN = conv_generator(in_dim=M,out_dim=64)
    # self.add_mean = MeanShift(1.0, (0.4488, 0.4371, 0.4040), (1.0, 1.0, 1.0), 1)
  
  def post_processing_params(self):
    params = []
    params += self.conv_weights_gen_HAN.parameters()
    params += self.HAN.parameters()

    return params
  
  def base_params(self):
    params = []
    params += self.a_model.parameters()
    params += self.s_model.parameters()

    params += self.ha_model.parameters()
    params += self.hs_model.parameters()

    params += self.syntax_model.parameters()
    params += self.conv_weights_gen.parameters()

    params += self.prediction_model.parameters()
    params += self.prediction_model_syntax.parameters()

    params.append(self.v_z2_sigma)

    return params
  
  def batch_conv(self, weights, inputs): # conv_weights: B * 6 * M * 1 * 1, x_tilde: B * M * H/2 * H/2
    b, ch, _, _ = inputs.shape # b = B, ch = M
    _, ch_out, _, k, _ = weights.shape # ch_out = 6, k = 1
  
    weights = weights.reshape(b*ch_out, ch, k, k) # 6B * M * 1 * 1
    inputs = torch.cat(torch.split(inputs, 1, dim=0), dim=1) # 1 * BM * H/2 * H/2
    out = F.conv2d(inputs, weights,stride=1,padding=0,groups=b) # 1 * 6B * H/2 * H/2
    out = torch.cat(torch.split(out, ch_out, dim=1), dim=0) # B * 6 * H/2 * H/2
    
    return out
  
  def forward(self, y, x_tilde, mode='train'): # y: B * N * H/32 * W/32, x_tilde: B * M * H/2 * H/2
    # b, h, w, c = self.train_size
    # tb, th, tw, tc = self.test_size
    
    z = self.ha_model(y) # B * N * H/128 * W/128
    print('z size: ', z.shape)
    
    noise = torch.rand_like(z) - 0.5
    z_noisy = z + noise
    z_rounded = bypass_round(z)

    z_tilde = self.hs_model(z_rounded) # B * N * H/32 * W/32
    z_sigma = self.z_sigma.cuda()
    z_mu = torch.zeros_like(z_sigma)

    y_syntax = y[:,:self.M,:,:] # B * 16 * H/32 * W/32
    y_syntax = self.syntax_model(y_syntax) # syntax generator, B * 16 * 1 * 1
    print('y_syntax size: ', y_syntax.shape)

    # Syntax
    noise = torch.rand_like(y_syntax) - 0.5
    y_syntax_noisy = y_syntax + noise
    y_syntax_rounded = bypass_round(y_syntax)

    
    if mode == 'train':
      z_likelihoods = self.entropy_bottleneck_z(z_noisy, z_sigma, z_mu) # B * N * H/128 * W/128
      print('z_likelihoods size: ', z_likelihoods.shape)

      # Syntax
      y_syntax_sigma, y_syntax_mu = self.prediction_model_syntax(y_syntax_rounded, z_tilde) # B * M * 1 * 1, B * M * 1 * 1
      y_syntax_likelihoods = self.entropy_bottleneck_y_syntax(y_syntax_noisy, y_syntax_sigma, y_syntax_mu)
      print('y_syntax_likelihoods size: ', y_syntax_likelihoods.shape) # B * 16 * 1 * 1

    else:
      z_likelihoods = self.entropy_bottleneck_z(z_rounded, z_sigma, z_mu)
      
      # Syntax
      y_syntax_sigma, y_syntax_mu = self.prediction_model_syntax(y_syntax_rounded, z_tilde)
      y_syntax_likelihoods = self.entropy_bottleneck_y_syntax(y_syntax_rounded, y_syntax_sigma, y_syntax_mu)
    
    
    conv_weights = self.conv_weights_gen(y_syntax_rounded) # B * 6 * M * 1 * 1
    print('conv_weights size: ', conv_weights.shape)


    x_tilde = self.batch_conv(conv_weights, x_tilde)  # B * 6 * H/2 * H/2
    
    # if self.post_processing:
    #     x_tilde = self.HAN(x_tilde_bf)
    #     conv_weights = self.conv_weights_gen_HAN(y_syntax_rounded)
    #     x_tilde = self.batch_conv(conv_weights, x_tilde)
    #     x_tilde = self.add_mean(x_tilde)
    # else:
    #     x_tilde = x_tilde_bf


    b, h, w, c = x_tilde.shape
    tb, th, tw, tc = x_tilde.shape
    num_pixels = b * h * w # B * 256 * 256

    if mode == 'train':
      
      bpp_list = [torch.sum(torch.log(l), [0,1,2,3]) / (-np.log(2) * num_pixels) for l in [z_likelihoods, y_syntax_likelihoods]]

      train_bpp = bpp_list[0] + bpp_list[1]

        # train_mse = torch.mean((x[:,:,:h,:w] - x_tilde[:,:,:h,:w]) ** 2, [0,1,2,3])
        # train_mse *= 255**2

      return train_bpp, x_tilde
    

    elif mode == 'test':
      test_num_pixels = tb * th * tw

      bpp_list = [torch.sum(torch.log(l), [0,1,2,3]) / (-np.log(2) * test_num_pixels) for l in [z_likelihoods, y_syntax_likelihoods]]

      eval_bpp = bpp_list[0] + bpp_list[1]

      # Bring both images back to 0..255 range.
        # gt = torch.round((x + 1) * 127.5)
        # x_hat = torch.clamp((x_tilde + 1) * 127.5, 0, 255)
        # x_hat = torch.round(x_hat).float()

        # v_mse = torch.mean((x_hat - gt) ** 2, [1,2,3])
        # v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)
        
      return eval_bpp, x_tilde