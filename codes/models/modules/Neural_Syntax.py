import argparse
import glob

import numpy as np
import torch
import torchvision as tv
from torch import nn, optim
import torch.nn.functional as F

import pickle
from PIL import Image
from torch.autograd import Function
import time
import os

#from .han import HAN_Head as HAN
#from .han import MeanShift



class Space2Depth(nn.Module):
  def __init__(self, r):
    super(Space2Depth, self).__init__()
    self.r = r
  
  def forward(self, x):
    r = self.r
    b, c, h, w = x.size()
    out_c = c * (r**2)
    out_h = h//2
    out_w = w//2
    x_view = x.view(b, c, out_h, r, out_w, r)
    x_prime = x_view.permute(0,3,5,1,2,4).contiguous().view(b, out_c, out_h, out_w)
    return x_prime

class Depth2Space(nn.Module):
  def __init__(self, r):
    super(Depth2Space, self).__init__()
    self.r = r
  def forward(self, x):
    r = self.r
    b, c, h, w = x.size()
    out_c = c // (r**2)
    out_h = h * 2
    out_w = w * 2
    x_view = x.view(b, r, r, out_c, h, w)
    x_prime = x_view.permute(0,3,4,1,5,2).contiguous().view(b, out_c, out_h, out_w)
    return x_prime

class h_analysisTransformModel(nn.Module):
  def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
    super(h_analysisTransformModel, self).__init__()
    self.transform = nn.Sequential(
      nn.Conv2d(in_dim,         num_filters[0], 3, strides_list[0], 1),
      nn.ReLU(),
      nn.Conv2d(num_filters[0], num_filters[1], 5, strides_list[1], 2),
      nn.ReLU(),
      nn.Conv2d(num_filters[1], num_filters[2], 5, strides_list[2], 2)
    )
    
  def forward(self, inputs):
    x = torch.abs(inputs)
    x = self.transform(x)
    return x

class h_synthesisTransformModel(nn.Module):
  def __init__(self, in_dim, num_filters, strides_list, conv_trainable=True):
    super(h_synthesisTransformModel, self).__init__()
    self.transform = nn.Sequential(
      nn.ConvTranspose2d(in_dim,         num_filters[0], 5, strides_list[0], 2, output_padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(num_filters[0], num_filters[1], 5, strides_list[1], 2, output_padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(num_filters[1], num_filters[2], 3, strides_list[2], 1),
    )
  
  def forward(self, inputs):
    x = self.transform(inputs)
    return x

class BlockSample(nn.Module):
  def __init__(self, in_shape, masked=True):
    super(BlockSample, self).__init__()
    self.masked = masked
    dim = in_shape[1]
    flt = np.zeros((dim*16, dim, 7, 7), dtype=np.float32)
    for i in range(0, 4):
      for j in range(0, 4):
        if self.masked:
          if i == 3:
            if j == 2 or j == 3:
              break
        for k in range(0,dim):
          s = k*16 + i * 4 + j
          flt[s, k, i, j+1] = 1
    flt_tensor = torch.Tensor(flt).float().cuda()
    self.register_buffer('sample_filter', flt_tensor)
  
  def forward(self, inputs):
    t = F.conv2d(inputs, self.sample_filter, padding=3)
    b, c, h, w = inputs.size()
    t = t.contiguous().view(b, c, 4, 4, h, w).permute(0, 4, 5, 1, 2, 3)
    t = t.contiguous().view(b*h*w, c, 4, 4)
    return t

class NeighborSample(nn.Module):
  def __init__(self, in_shape):
    super(NeighborSample, self).__init__()
    dim = in_shape[1]
    flt = np.zeros((dim*25, dim, 5, 5), dtype=np.float32)
    for i in range(0, 5):
      for j in range(0, 5):
        for k in range(0, dim):
          s = k*25 + i * 5 + j
          flt[s, k, i, j] = 1
    flt_tensor = torch.Tensor(flt).float().cuda()
    self.register_buffer('sample_filter', flt_tensor)
  
  def forward(self, inputs):
    t = F.conv2d(inputs, self.sample_filter, padding=2)
    b, c, h, w = inputs.size()
    t = t.contiguous().view(b, c, 5, 5, h, w).permute(0, 4, 5, 1, 2, 3)
    t = t.contiguous().view(b*h*w, c, 5, 5)
    return t
  
class GaussianModel(nn.Module):
  def __init__(self):
    super(GaussianModel, self).__init__()
    
    self.m_normal_dist = torch.distributions.normal.Normal(0., 1.)

  def _cumulative(self, inputs, stds, mu):
    half = 0.5
    eps = 1e-6
    upper = (inputs - mu + half) / (stds)
    lower = (inputs - mu - half) / (stds)
    cdf_upper = self.m_normal_dist.cdf(upper)
    cdf_lower = self.m_normal_dist.cdf(lower)
    res = cdf_upper - cdf_lower
    return res
  
  def forward(self, inputs, hyper_sigma, hyper_mu):
    likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
    likelihood_bound = 1e-8
    likelihood = torch.clamp(likelihood, min=likelihood_bound)
    return likelihood
    
class PredictionModel_Context(nn.Module):
  def __init__(self, in_dim, dim=192, trainable=True, outdim=None):
    super(PredictionModel_Context, self).__init__()
    if outdim is None:
      outdim = dim
    self.transform = nn.Sequential(
      nn.Conv2d(in_dim, dim, 3, 1, 1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(dim, dim, 3, 2, 1),
      nn.LeakyReLU(0.2),
      nn.Conv2d(dim, dim, 3, 1, 1),
      nn.LeakyReLU(0.2)
    )
    self.fc = nn.Linear(dim*2*2, outdim)
    self.flatten = nn.Flatten()
    
  def forward(self, y_rounded, h_tilde, y_sampler, h_sampler):
    b, c, h, w = y_rounded.size()
    y_sampled = y_sampler(y_rounded)
    h_sampled = h_sampler(h_tilde)
    merged = torch.cat([y_sampled, h_sampled], 1)
    y_context = self.transform(merged)
    y_context = self.flatten(y_context)
    y_context = self.fc(y_context)
    hyper_mu = y_context[:, :c]
    hyper_mu = hyper_mu.view(b, h, w, c).permute(0, 3, 1, 2)
    hyper_sigma = y_context[:, c:]
    hyper_sigma = torch.exp(hyper_sigma)
    hyper_sigma = hyper_sigma.contiguous().view(b, h, w, c).permute(0, 3, 1, 2)

    return hyper_mu, hyper_sigma


class conv_generator(nn.Module):
    def __init__(self,in_dim,out_dim): # in_dim=M, out_dim=M
      super(conv_generator,self).__init__()
      self.in_dim = in_dim
      self.out_dim = out_dim
      self.transform = nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.LeakyReLU(0.2),
        nn.Linear(128,256),
        nn.LeakyReLU(0.2),
        nn.Linear(256,out_dim*6)
      )

    def forward(self,x):  # x: B * M * 1 * 1
      b,_,_,_ = x.shape
      x = x.view(b,-1) # B * M
      
      weights = self.transform(x) # B * 6M
      weights = weights.view(b, 6, self.out_dim,1,1)

      return weights # B * 6 * M * 1 * 1

class Syntax_Model(nn.Module):
  def __init__(self,in_dim,out_dim):
    super(Syntax_Model, self).__init__()
    self.down0 = nn.Conv2d(in_dim, 32, 3,2,1)
    self.down1 = nn.Conv2d(32,64,3,2,1)

    self.conv = nn.Conv2d(in_dim+32+64, out_dim,1,1,0)
    self.pooling = nn.AdaptiveAvgPool2d(1)
  def forward(self,syntax):

    out1 = self.pooling(syntax)
    
    ds1 = self.down0(syntax)
    ds1 = F.relu(ds1)
    out2 = self.pooling(ds1)

    ds2 = self.down1(ds1)
    ds2 = F.relu(ds2)
    out3 = self.pooling(ds2)

    out = torch.cat((out1,out2,out3),1)
    out = self.conv(out)
    return out

class PredictionModel_Syntax(nn.Module):
  def __init__(self, in_dim, dim=192, trainable=True, outdim=None): # in_dim=N, dim=M, outdim=M * 2
    super(PredictionModel_Syntax, self).__init__()
    if outdim is None:
      outdim = dim
    
    self.down0 = nn.Conv2d(in_dim, dim, 3,2,1)
    self.down1 = nn.Conv2d(dim,dim,3,2,1)
    self.pooling = nn.AdaptiveAvgPool2d(1)

    self.fc = nn.Linear(dim*2+in_dim, outdim)
    self.flatten = nn.Flatten()
    
  def forward(self, y_rounded, h_tilde, h_sampler=None): # y_rounded: B * 16 * 1 * 1, h_tilde: B * N * H/32 * W/32
    b, c, h, w = y_rounded.size()
    
    ds0 = self.down0(h_tilde) # B * M * H/64 * W/64
    ds0 = F.relu(ds0)

    ds1 = self.down1(ds0) # B * M * H/128 * W/128
    ds1 = F.relu(ds1)

    ds0 =self.pooling(ds0) # B * M * 1 * 1
    ds1 = self.pooling(ds1) # B * M * 1 * 1
    ori = self.pooling(h_tilde) # B * N * 1 * 1
    y_context = torch.cat((ori,ds0,ds1),1) # B * (N+2M) * 1 * 1

    y_context = self.flatten(y_context) # B * (N+2M)
    y_context = self.fc(y_context) # B * 2M
    hyper_mu = y_context[:, :c] # B * M
    hyper_mu = hyper_mu.view(b, h, w, c).permute(0, 3, 1, 2) # B * M * 1 * 1
    hyper_sigma = y_context[:, c:] # B * M
    hyper_sigma = torch.exp(hyper_sigma)
    hyper_sigma = hyper_sigma.contiguous().view(b, h, w, c).permute(0, 3, 1, 2) # B * M * 1 * 1

    return hyper_mu, hyper_sigma # B * M * 1 * 1, B * M * 1 * 1

class BypassRound(Function):
  @staticmethod
  def forward(ctx, inputs):
    return torch.round(inputs)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output