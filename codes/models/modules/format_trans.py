import torch
import torch.nn as nn
import cv2
import numpy as np
import subprocess as sp
import torch.nn.functional as F
import av
from io import BytesIO
import math
from fractions import Fraction
import logging


def rgb2yuv420_down(rgb, scalea, scaleb): # B*3*H*W, 1, 2
    b, c, height, width = rgb.shape
    assert c == 3, 'channel dose not match'
    rgb = (torch.clamp(rgb, 0, 1) * 255).cpu().numpy()
    rgb = np.round(np.transpose(rgb, (0, 2, 3, 1))).astype(np.uint8)

    out = torch.zeros((b, 2 * c, height // 2, width // 2)).cuda()
    down = torch.zeros((b, 2 * c, height // 4, width // 4)).cuda()
    for i in range(b):
        frame = av.VideoFrame.from_ndarray(rgb[i, :, :, :], format='rgb24').reformat(format='yuv420p')
        yuv420p = frame.to_ndarray()
        img_cuda = np.array(yuv420p).astype(np.float32) / 255.
        img_cuda = torch.tensor(img_cuda).cuda()
        y_tensor = torch.nn.functional.pixel_unshuffle(img_cuda[:height, :].view(1, 1, height, width), 2)
        u_tensor = img_cuda[height:height + height // 4, :].view(1, 1, height // 2, width // 2)
        v_tensor = img_cuda[height + height // 4:, :].view(1, 1, height // 2, width // 2)
        out[i, :] = torch.cat((y_tensor, u_tensor, v_tensor), 1).squeeze(0) # 6*H/2*W/2

        h = int(height * scalea / scaleb)
        w = int(width * scalea / scaleb)
        yuv_down = frame.reformat(width=w, height=h, interpolation='LANCZOS').to_ndarray()
        img_down = np.array(yuv_down).astype(np.float32) / 255.
        img_down = torch.tensor(img_down).cuda()
        y = torch.nn.functional.pixel_unshuffle(img_down[:h, :].view(1, 1, h, w), 2)
        u = img_down[h:h + h // 4, :].view(1, 1, h // 2, w // 2)
        v = img_down[h + h // 4:, :].view(1, 1, h // 2, w // 2)
        down[i, :] = torch.cat((y, u, v), 1).squeeze(0)
    return out, down



def yuv4202rgb(yuv):
    b, c, height, width = yuv.shape  # b * 6 * H//4 * W//4
    out = torch.zeros((b, c // 2, height * 2, width * 2))
    for i in range(b):
        img = yuv[i, :, :, :].view(1, 6, height, width).clone().detach()
        y_tensor = torch.nn.functional.pixel_shuffle(img[:, :4, :, :], 2)[0, 0, :, :].cpu().numpy()
        u_tensor = np.reshape(img[0, 4, :, :].cpu().numpy(), (height // 2, width * 2))
        v_tensor = np.reshape(img[0, 5, :, :].cpu().numpy(), (height // 2, width * 2))
        img = np.round(np.clip(np.concatenate((y_tensor, u_tensor, v_tensor), axis=0), 0, 1) * 255.).astype(np.uint8)
        rgb = (av.VideoFrame.from_ndarray(img, format='yuv420p')).reformat(format='rgb24').to_ndarray()
        img_cuda = np.array(rgb).astype(np.float32) / 255.
        img_cuda = torch.tensor(img_cuda).cuda().view(1, height * 2, width * 2, 3).permute(0, 3, 1, 2) # 1 * 3 * H//2 * W//2
        out[i, :] = img_cuda.view(1, 3, height * 2, width * 2)
    return out