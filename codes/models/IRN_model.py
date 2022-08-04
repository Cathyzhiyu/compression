import logging
from collections import OrderedDict
import numpy as np
import torch
import cv2
import math
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.ffmpeg_DE import DecodeEncode, DownSample, UpSample #codecImage, rgbdown
import sys
import torch.nn.functional as F
from models.modules.Quant import BypassRound, Quant
from models.modules.format_trans import rgb2yuv420_down, yuv4202rgb

logger = logging.getLogger('base')

class IRNModel(BaseModel):
    def __init__(self, opt):
        super(IRNModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.scalea = opt['scalea']
        self.scaleb = opt['scaleb']

        self.netG = networks.define_G(opt).to(self.device)
        self.netA = networks.define_Analysis(opt).to(self.device)
        self.netS = networks.define_S(opt).to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
            self.netA = DistributedDataParallel(self.netA, device_ids=[torch.cuda.current_device()])
            self.netS = DistributedDataParallel(self.netS, device_ids=[torch.cuda.current_device()])

        else:
            self.netA = DataParallel(self.netA)
            self.netG = DataParallel(self.netG)
            self.netS = DataParallel(self.netS)

        # print network
        self.print_network()
        self.load()

        self.DecodeEncode = DecodeEncode()
        self.bypass_round = BypassRound

        if self.is_train:
            self.netG.train()
            self.netA.train()
            self.netS.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            optim_params = []
            optim_rates = []
            for k, v in self.netA.named_parameters():
                if v.requires_grad:
                    if ('rate' not in k):
                        optim_params.append(v)
                    else:
                        optim_rates.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_A = torch.optim.Adam(optim_params, lr=train_opt['lr_A'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_A)

            #self.optimizer_C = torch.optim.Adam(optim_rates, lr=train_opt['lr_A'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
            #self.optimizers.append(self.optimizer_C)

            optim_params = []
            for k, v in self.netS.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_S = torch.optim.Adam(optim_params, lr=train_opt['lr_S'], weight_decay=wd_G, betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_S)

            

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()


    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y):
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)

        return l_forw_fit
    
    def loss_backward(self, out, y):
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, y)

        return l_back_rec
    
    def feed_data(self, data):
        #self.ref_L_rgb = data['LQ'].to(self.device)  # LQ
        self.real_H_rgb = data['GT'].to(self.device)  # GT

    def optimize_parameters(self, step):

        ### compress
        qp = "6"
        print('real_H_rgb size: ', self.real_H_rgb.shape)
        self.real_H, self.ref_L = rgb2yuv420_down(self.real_H_rgb, self.scalea, self.scaleb) # (B * 6 * H/2 * W/2 , B * 6 * H/4 * W/4)
        print('real_H size: ', self.real_H.shape)
        print('ref_L size: ', self.ref_L.shape)
        _, _, h, w = self.real_H.size()
        out_mv = None

        # update netA virtual codec
        lr_lanczos = self.ref_L # B * 6 * H/4 * W/4
        lr_lanczos_codec, out_mv = self.DecodeEncode(lr_lanczos, qp, False)
        self.optimizer_A.zero_grad()
        lr_lanczos_codec_est, _ = self.netA(x=lr_lanczos, mv=out_mv, I=True)
        l_codec_est = self.loss_forward(lr_lanczos_codec_est, lr_lanczos_codec) # loss between netA & x265 decoded lr
        loss_A = l_codec_est
        loss_A.backward()
        torch.nn.utils.clip_grad_norm_(self.netA.parameters(), 1.0)
        self.optimizer_A.step()

        # update encoder & decoder netG (& rate estimation network in netA), netS
        self.optimizer_G.zero_grad()
        # self.optimizer_C.zero_grad()
        y, y_syntax, content = self.netG(x=self.real_H, up=False)  # forward downscaling, B * 6 * H/4 * W/4
        print('content size: ', content.shape) # B * 6 * H/4 * W/4
        print('y size: ', y.shape) # B * 192 * H/32 * W/32
        print('y_syntax size: ', y_syntax.shape) # B * 16 * H/32 * W/32
        content_codec_est, bpp_netA = self.netA(x=content, mv=out_mv, I=True)
        # l_size_fit = torch.mean(bpp_netA)
        sr_netG = self.netG(x=content_codec_est, up=True) # B * M * H/2 * W/2

        # Net for syntax
        bpp_netS, sr_tilde = self.netS(y = y, x_tilde = sr_netG)  # B * 6 * H/2 * H/2    

        l_tilde_real = self.loss_forward(sr_tilde, self.real_H)
        l_content_ref = self.loss_forward(content, self.ref_L)

        loss = l_tilde_real + 0.5 * l_content_ref  + bpp_netS #+ 8e-4 * l_size_fit
        loss = loss.mean()
        l_tilde_real = l_tilde_real.mean()
        l_content_ref = l_content_ref.mean()
        bpp_netS = bpp_netS.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.netS.parameters(), 1.0)
        self.optimizer_G.step()
        self.optimizer_S.step()
        #self.optimizer_C.step()

        # set log
        self.log_dict['l_codec_est'] = l_codec_est.item()
        self.log_dict['l_content_ref'] = l_content_ref.item()
        self.log_dict['l_tilde_real'] = l_tilde_real.item()
        self.log_dict['bpp_netS'] = bpp_netS.item()
        self.log_dict['distortion'] = (l_tilde_real + 0.5 * l_content_ref).item()
        #self.log_dict['l_size_fit'] = l_size_fit.item()



    def test(self):
        qp = "22"
        self.netG.eval()
        self.netS.eval()
        with torch.no_grad():
            # YUV train
            real_H, ref_L = rgb2yuv420_down(self.real_H_rgb, self.scalea, self.scaleb) # (B * 6 * H/2 * W/2 , B * 6 * H/4 * W/4)
            self.ref_L = yuv4202rgb(ref_L) # B * 3 * H/2 * W/2
            # _, _, h, w = real_H.size()
            # ref_L, _ = self.DecodeEncode(ref_L, qp, True) # B * 6 * H/4 * W/4
            # fake_H_bic = self.yuvUpsample(ref_L, h, w) # b * 6 * height//2 * width//2
            # self.fake_H_bic = yuv4202rgb(fake_H_bic) # b * 3 * height * width
            # self.psnr_fix = 10 * torch.log10(1**2 / torch.mean((real_H - fake_H_bic)**2))

            y, y_syntax, content = self.netG(x=real_H, up=False) # forward downscaling, B * 6 * H/4 * W/4
            self.forw_L = yuv4202rgb(content) # B * 3 * H/2 * W/2
            content_codec, _ = self.DecodeEncode(content, qp, True) # B * 6 * H/4 * W/4
            sr_netG = self.netG(x=content_codec, up=True) # B * M * H/2 * W/2
            bpp_netS, sr_tilde = self.netS(y = y, x_tilde = sr_netG, mode='test')  # B * 6 * H/2 * H/2 
            self.fake_H = yuv4202rgb(sr_tilde) # B * 3 * H * W
            self.psnr = 10 * torch.log10(1**2 / torch.mean((real_H - sr_tilde)**2))
            #self.psnr = 10 * torch.log10(255**2 / torch.mean((real_H - sr_tilde)**2))
            self.bpp_netS = bpp_netS
            self.psnr = self.psnr.mean()
            self.bpp_netS = self.bpp_netS.mean()
        
        self.netG.train()
        self.netS.train()


    def downscale(self, HR_img):
        self.netG.eval()
        with torch.no_grad():
            y, y_syntax, LR_img = self.netG(x=HR_img, up=False)
        return LR_img

    def upscale(self, LR_img, scale, gaussian_scale=1):
        with torch.no_grad():
            HR_img = self.netG(x=LR_img, up=True, flow=None, I=True)
        self.netG.eval()
        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        #out_dict['SR_bic'] = self.fake_H_bic.detach()[0].float().cpu()
        #out_dict['PSNR_fix'] = self.psnr_fix.detach().item()
        out_dict['PSNR'] = self.psnr.detach().item()
        out_dict['bpp_netS'] = self.bpp_netS.detach().item()
        # out_dict['SSIM_fix'] = self.ssim_fix.detach().item()
        # out_dict['SSIM'] = self.ssim.detach().item()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
        
        load_path_A = self.opt['path']['pretrain_model_A']
        if load_path_A is not None:
            logger.info('Loading model for A [{:s}] ...'.format(load_path_A))
            self.load_network(load_path_A, self.netA, self.opt['path']['strict_load'])
        
        load_path_S = self.opt['path']['pretrain_model_S']
        if load_path_S is not None:
            logger.info('Loading model for S [{:s}] ...'.format(load_path_S))
            self.load_network(load_path_S, self.netS, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
        self.save_network(self.netA, 'A', iter_label)
        self.save_network(self.netS, 'S', iter_label)
        
