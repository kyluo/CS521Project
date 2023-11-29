import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional
import sys
import math

import setting
import utils
from model.model_l_base import UNet_Large_Basic
from datasets import create_dataloader, create_dataset
from utils import objectify
import logging

logname = "train.log"
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger('base')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def compute_avg_psnr(val_loader, net):
    net.eval()
    avg_psnr = 0.
    idx = 0
    for val_data in val_loader:
        if(idx > 10):
            break
        idx += 1
        lr = val_data['LQ'].to(setting.device)
        gt = val_data['GT'].to(setting.device)
        sr = net(lr)
        sr = sr.detach().cpu()
        gt = gt.detach().cpu()
        sr_img = utils.tensor2img(sr)  # uint8
        gt_img = utils.tensor2img(gt)  # uint8
        avg_psnr += utils.calculate_psnr(sr_img, gt_img)
    avg_psnr = avg_psnr / idx
    net.train()

    # log
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))

if __name__ == "__main__":
    opt = {
        "train": True,
        "epoch": 100,
        "save_model": True,
        "show_progress": True,
        "print_freq": 10,
        "val_freq": 100,
        "checkpoint": "model_checkpoints/model_99.pth",
    }
    opt = objectify(opt)
    dataset_opt = {
        "dataroot_GT" : "data/DIV2K_train_HR",
        "dataroot_LQ" : None,
        "batch_size" : 4,
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
    val_dataset_opt = {
        "dataroot_GT" : "data/DIV2K_valid_HR",
        "dataroot_LQ" : None,
        "batch_size" : 16,
        "mode" : "LQGT",
        "data_type": "img",
        "phase" : "test",
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
    if opt.checkpoint is not None:
        net.load_state_dict(torch.load(opt.checkpoint))
    train_set = create_dataset(dataset_opt)
    train_loader = create_dataloader(train_set, dataset_opt, opt)
    val_set = create_dataset(dataset_opt)
    val_loader = create_dataloader(val_set, val_dataset_opt, opt)
    print("Start training")
    net.train()
    loss_func = nn.MSELoss()
    if not os.path.exists(setting.model_path):
        os.mkdir(setting.model_path)
    for epoch in tqdm(range(opt.epoch), total=opt.epoch, desc="Training ...", position=1, disable=opt.show_progress):
        logger.info("Epoch: {}".format(epoch))
        for current_step, train_data in enumerate(tqdm(train_loader)):

            lr = train_data['LQ'].to(setting.device)
            hr = train_data['GT'].to(setting.device)
            sr = net(lr)
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
            if current_step % opt.val_freq == 0:
                compute_avg_psnr(val_loader, net)

        if epoch % 10 == 0 and opt.save_model:
            torch.save(net.state_dict(), os.path.join(setting.model_path, "model_{}.pth".format(epoch)))
    compute_avg_psnr(val_loader, net)
    torch.save(net.state_dict(), os.path.join(setting.model_path, "model_{}.pth".format(epoch)))
    print("\nTraining Complete\n")