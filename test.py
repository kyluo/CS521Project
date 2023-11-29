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
import utils
from model.model_l_base import UNet_Large_Basic
from datasets import create_dataloader, create_dataset
from utils import objectify
import logging
logname = "test.log"
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
    for val_data in tqdm(val_loader):
        idx += 1
        if idx > 100:
            break
        lr = val_data['LQ'].to(setting.device)
        gt = val_data['GT'].to(setting.device)
        sr = net(lr)
        sr = sr.detach().cpu()
        gt = gt.detach().cpu()
        sr_img = utils.tensor2img(sr)  # uint8
        gt_img = utils.tensor2img(gt)  # uint8
        avg_psnr += utils.calculate_psnr(sr_img, gt_img)
    avg_psnr = avg_psnr / idx

    # log
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))

if __name__ == "__main__":
    opt = {
        "train": False,
        "epoch": 1,
        "show_progress": True,
        "save_model": False,
        "print_freq": 10,
        "val_freq": 50,
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
    val_set = create_dataset(dataset_opt)
    val_loader = create_dataloader(val_set, val_dataset_opt, opt)
    print("Start validation")
    net.eval()
    compute_avg_psnr(val_loader, net)

        
