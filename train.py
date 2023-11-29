import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import DataHolder, load_tensor
import numpy as np
import torch.nn.functional
import sys
import math

import setting
from model.model_l_base import UNet_Large_Basic
from datasets import create_dataloader, create_dataset
from utils import objectify
import logging
logger = logging.getLogger('base')


if __name__ == "__main__":
    opt = {
        "train": True,
        "epoch": 10,
        "show_progress": True,
        "save_model": False,
        "print_freq": 10,
    }
    opt = objectify(opt)
    dataset_opt = {
        "dataroot_GT" : "data/DIV2K_train_HR",
        "dataroot_LQ" : None,
        "batch_size" : 16,
        "mode" : "LQGT",
        "data_type": "img",
        "phase" : "train",
        "name" : "DIV2K",
        "n_workers" : 4,
        "scale" : 2, # for modcrop in the validation / test phase
        "GT_size" : 1024,
        "color" : "RGB",
        "use_flip": True,
        "use_rot": True,
    }
    net = UNet_Large_Basic(3, 3)
    net.to(setting.device)
    
    train_set = create_dataset(dataset_opt)
    train_loader = create_dataloader(train_set, dataset_opt, opt)
    print("Start training")
    net.train()
    loss_func = nn.MSELoss()
    for epoch in tqdm(range(opt.epoch), total=opt.epoch, desc="Training ...", position=1, disable=opt.show_progress):
        for current_step, train_data in enumerate(train_loader):

            lr = train_data['LQ'].to(setting.device)
            print(lr.shape)
            hr = train_data['GT'].to(setting.device)
            print(hr.shape)
            sr = net(lr)
            print(sr.shape)
            loss = loss_func(sr, hr)
            loss.backward()
            optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
            optimizer.step()
            optimizer.zero_grad()
            
        
            #### log
            if current_step % opt.print_freq == 0:
                logger.info(
                    "Epoch: {} | Iteration: {} | Loss: {}".format(epoch, current_step, loss.item()))
            #### validation
            if opt['datasets'].get('val', None) and current_step % opt['train']['val_freq'] == 0:
                if opt['model'] in ['sr', 'srgan'] and rank <= 0:  # image restoration validation
                    # does not support multi-GPU validation
                    pbar = util.ProgressBar(len(val_loader))
                    avg_psnr = 0.
                    idx = 0
                    for val_data in val_loader:
                        idx += 1
                        img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)

                        model.feed_data(val_data)
                        model.test()

                        visuals = model.get_current_visuals()
                        sr_img = util.tensor2img(visuals['rlt'])  # uint8
                        gt_img = util.tensor2img(visuals['GT'])  # uint8

                        # Save SR images for reference
                        save_img_path = os.path.join(img_dir,
                                                     '{:s}_{:d}.png'.format(img_name, current_step))
                        if opt['save_img']:
                            util.save_img(sr_img, save_img_path)

                        # calculate PSNR
                        sr_img, gt_img = util.crop_border([sr_img, gt_img], opt['scale'])
                        avg_psnr += util.calculate_psnr(sr_img, gt_img)
                        pbar.update('Test {}'.format(img_name))

                    avg_psnr = avg_psnr / idx

                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))


        

        if (not opt.show_progress):
            print("\n")
            print(
                f'Loss_G for Training on epoch {str(epoch)} is {str(loss_G)} \n')
            print(
                f'Loss_D for Training on epoch {str(epoch)} is {str(loss_D)} \n')

        # if epoch % 100 == 0 and opt.save_model:
        #     save_model(net, opt)
    print("\nTraining Complete\n")