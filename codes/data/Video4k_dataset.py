import os.path as osp
import random
import pickle
import logging
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util

from pathlib import Path


class Video4kDataset(data.Dataset):
    '''
    Reading the training REDS dataset
    key example: 000/00001.png
    GT: Ground-Truth;
    LQ: Low-Quality, e.g., low-resolution frames
    support reading N HR frames, N = 3, 5, 7
    '''

    def __init__(self, opt):
        super(Video4kDataset, self).__init__()
        self.opt = opt

        self.gt_root = Path(opt['dataroot_GT'])
        # self.lq_root = Path(opt['dataroot_LQ'])
        # self.codec_root = Path(opt['dataroot_codec'])
        self.random_reverse = opt['random_reverse']
        self.interval = opt['interval']
        self.keys = []
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _ = line.split(' ')
                self.keys.extend([f'{folder}/{i:05d}' for i in range(1, int(frame_num) + 1, self.interval)])

        # remove the video clips used in validation
        # if opt['val_partition'] == 'REDS4':
        #     val_partition = ['000', '011', '015', '020']
        # elif opt['val_partition'] == 'official':
        #     val_partition = [f'{v:03d}' for v in range(240, 270)]
        # else:
        #     raise ValueError(f'Wrong validation partition {opt["val_partition"]}.' f"Supported ones are ['official', 'REDS4'].")
        # self.keys = [v for v in self.keys if v.split('/')[0] not in val_partition]

    def __getitem__(self, index):
        scale = self.opt['scale']
        # GT_size = self.opt['GT_size']

        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 000/00001
        center_frame_idx = int(seq)

        #### determine the neighbor frames
        # interval = random.choice(self.opt['interval_list'])
        # # ensure not exceeding the borders
        # start_frame_idx = center_frame_idx - self.half_N_frames * interval
        # end_frame_idx = center_frame_idx + self.half_N_frames * interval
        # # each clip has 100 frames starting from 0 to 99
        # while (start_frame_idx < 0) or (end_frame_idx > 99):
        #     center_frame_idx = random.randint(0, 99)
        #     start_frame_idx = (center_frame_idx -
        #                        self.half_N_frames * interval)
        #     end_frame_idx = center_frame_idx + self.half_N_frames * interval
        # frame_name = f'{center_frame_idx:08d}'
        # neighbor_list = list(
        #     range(center_frame_idx - self.half_N_frames * interval,
        #           center_frame_idx + self.half_N_frames * interval + 1,
        #           interval))
        # # random reverse
        # if self.random_reverse and random.random() < 0.5:
        #     self.neighbor_list.reverse()
        # 1 frame for video compression
        self.neighbor_list = [center_frame_idx]
        # get the neighboring LQ frames
        img_list_GT = []
        # img_list_LQ = []
        # img_list_codec = []
        crop_border = self.opt['crop_border']

        for neighbor in self.neighbor_list:
            GT_path = str(self.gt_root / clip / f'{neighbor:05d}.png')
            img_GT = util.read_img(None, GT_path, None)
            # BGR => RGB
            img_GT = img_GT[:, :, [2, 1, 0]]
            # img_LQ = img_LQ[:, :, [2, 1, 0]]
            # img_codec = img_codec[:, :, [2, 1, 0]]
            if crop_border:
                img_GT = img_GT[crop_border:-crop_border, :, :]
                img_list_GT.append(img_GT)
            else:
                img_GT = util.random_crop(img_GT, 512, 512)
                img_list_GT.append(img_GT)
        if crop_border is None:
            img_list_GT = util.augment(img_list_GT)

        img_list_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(np.concatenate(img_list_GT, axis=2), (2, 0, 1)))).float()
        GT_path = str(self.gt_root / (clip + '_' + seq))

        return {'GT': img_list_GT, 'GT_path': GT_path}

    def __len__(self):
        return len(self.keys)