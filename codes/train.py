import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from data.util import bgr2ycbcr

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',
                        type=str,
                        default='/home/admin123/Documents/compression/compress/codes/options/train/train_IRN_x2.yml',
                        help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 100  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            print('train_data size: ', train_data['GT'].shape)
            print('train_data path: ', train_data['GT_path'])
            current_step += 1
            if current_step > total_iters:
                break
            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                # avg_bic_psnr = 0.0
                # avg_ssim = 0.0
                # avg_bic_ssim = 0.0
                avg_bppS = 0.0
                idx = 0
                logger_val = logging.getLogger('val')  # validation logger
                '''model_dict = model.netG.state_dict()
                for k, v in model_dict.items():
                    if ('weight' not in k):
                        print(k, v)'''

                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['GT_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()

                    if current_step % (opt['train']['val_freq'] * 10) == 0:
                        sr_img = util.tensor2img(visuals['SR'])  # uint8
                        lr_img = util.tensor2img(visuals['LR'])

                        # Save reconstructed SR & content LR images for reference
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                        util.save_img(sr_img, save_img_path) # sr reconstructed img, B * 3 * H * W

                        save_img_path_L = os.path.join(img_dir, '{:s}_forwLR_{:d}.png'.format(img_name, current_step))
                        util.save_img(lr_img, save_img_path_L) # lr content img before encoded, B * 3 * H/2 * W/2

                    # Save ground truth
                    if current_step == opt['train']['val_freq']:
                        #sr_bic_img = util.tensor2img(visuals['SR_bic'])
                        gt_img = util.tensor2img(visuals['GT'])  # uint8
                        gtl_img = util.tensor2img(visuals['LR_ref'])

                        save_img_path_gt = os.path.join(img_dir, '{:s}_GT_{:d}.png'.format(img_name, current_step))
                        util.save_img(gt_img, save_img_path_gt) # hr ground truth img, B * 3 * H * W

                        # save_img_path_bic = os.path.join(img_dir, '{:s}_BIC_{:d}.png'.format(img_name, current_step))
                        # util.save_img(sr_bic_img, save_img_path_bic)

                        save_img_path_refl = os.path.join(img_dir, '{:s}_REFL_{:d}.png'.format(img_name, current_step))
                        util.save_img(gtl_img, save_img_path_refl) # LANCZOS downscaled ground truth lr img, B * 3 * H/2 * W/2

                    # calculate PSNR
                    '''crop_size = opt['scale']
                    sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                    gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                    sr_bic_img_y = bgr2ycbcr(sr_bic_img, only_y=True)
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]
                    cropped_sr_bic_img = sr_bic_img[crop_size:-crop_size, crop_size:-crop_size]'''

                    avg_psnr += visuals['PSNR']
                    # avg_bic_psnr += visuals['PSNR_fix']
                    logger_val.info('# Validation # PSNR: {:.4e}.'.format(visuals['PSNR']))
                    # logger_val.info('# Validation # fix PSNR: {:.4e}.'.format(visuals['PSNR_fix']))
                    # avg_ssim += visuals['SSIM']
                    # avg_bic_ssim += visuals['SSIM_fix']
                    # logger_val.info('# Validation # SSIM: {:.6e}.'.format(visuals['SSIM']))
                    # logger_val.info('# Validation # fix SSIM: {:.6e}.'.format(visuals['SSIM_fix']))

                    avg_bppS += visuals['bpp_netS']

                avg_psnr = avg_psnr / idx
                # avg_bic_psnr = avg_bic_psnr / idx
                avg_bppS = avg_bppS / idx

                # log
                logger.info('# Validation # average PSNR: {:.4e}.'.format(avg_psnr))
                # logger.info('# Validation # average fix PSNR: {:.4e}.'.format(avg_bic_psnr))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> average psnr: {:.4e}.'.format(epoch, current_step, avg_psnr))
                # logger_val.info('<epoch:{:3d}, iter:{:8,d}> average fix psnr: {:.4e}.'.format(epoch, current_step, avg_bic_psnr))

                # logger.info('# Validation # average SSIM: {:.4e}.'.format(avg_psnr))
                # logger.info('# Validation # average fix SSIM: {:.4e}.'.format(avg_bic_psnr))
                # logger_val.info('<epoch:{:3d}, iter:{:8,d}> average ssim: {:.4e}.'.format(epoch, current_step, avg_psnr))
                # logger_val.info('<epoch:{:3d}, iter:{:8,d}> average fix ssim: {:.4e}.'.format(epoch, current_step, avg_bic_psnr))

                logger.info('# Validation # average bpp_netS: {:.4e}.'.format(avg_bppS))
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> average bpp_netS: {:.4e}.'.format(epoch, current_step, avg_bppS))
                

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models.')
                    model.save(current_step)

            if current_step % (opt['logger']['save_checkpoint_freq'] * 10) == 0:
                if rank <= 0:
                    logger.info('Saving training states.')
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
