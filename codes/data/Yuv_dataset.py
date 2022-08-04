from pathlib import Path
import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import os


def yuv420_to_yuv444(yuv):
    total_h, w = yuv.shape
    h = int(total_h / 1.5)
    uv_h = h // 4
    _w = int(w / 2)
    _h = int(h / 2)
    y = yuv[0:h, :]
    y = np.reshape(y, (_h, 2, _w, 2))
    y = np.transpose(y, (0, 2, 1, 3))
    y = np.reshape(y, (_h, _w, 4))
    u = np.reshape(yuv[h:h + uv_h, :], (_h, _w, 1))
    v = np.reshape(yuv[h + uv_h:, :], (_h, _w, 1))

    return np.concatenate((y, u, v), axis=2)


class YuvDataset(data.Dataset):
    """Vimeo90K dataset for training.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, seperated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """
    def __init__(self, opt):
        super(YuvDataset, self).__init__()
        self.opt = opt
        self.frame_height = opt['height']
        self.frame_width = opt['width']
        self.yuv_path = opt['yuvpath']
        self.frame_size = self.frame_height * self.frame_width
        self.frame_length = int((self.frame_size * 3) / 2)
        self.video_size = os.stat(self.yuv_path)[6]
        self.num_frames = int(self.video_size / self.frame_length)
        self.f = open(self.yuv_path, 'rb')
        self.vid_arr = []
        '''for i in range(self.num_frames):
            frame = self.read()
            self.vid_arr.append(frame)
        self.f.close()'''

    def __getitem__(self, index):
        frame = self.read()
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(frame, (2, 0, 1)))).float() / 255.0
        return {'GT': img_GT}

    def __len__(self):
        return self.num_frames

    def read(self):
        raw = self.f.read(self.frame_length)
        yuv = np.frombuffer(raw, dtype=np.uint8)
        yuv = yuv.reshape((int(self.frame_height * 1.5), self.frame_width))
        frame = yuv420_to_yuv444(yuv)
        return frame
