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
from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


import setting
import utils
from model import create_model
from datasets import create_dataloader, create_dataset
from utils import objectify
import logging
from model.loss import CharbonnierLoss


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
        print("avg_psnr:", avg_psnr)
    avg_psnr = avg_psnr / idx
    net.train()

    # log
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_parametrized.py model_yml_path train_yml_path")
        exit(1)
    model_yml_path = sys.argv[1]
    train_yml_path = sys.argv[2]
    model_opt = load(open(model_yml_path, 'r'), Loader=Loader)
    train_dataset_opt = model_opt['datasets']['train']
    val_dataset_opt = model_opt['datasets']['val']
    opt = objectify(load(open(train_yml_path, 'r'), Loader=Loader))
    net = create_model(model_opt)
    net.to(setting.device)
    logname = model_opt['logger']['log_path']
    logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    if opt.checkpoint is not None:
        net.load_state_dict(torch.load(opt.checkpoint))
    train_set = create_dataset(train_dataset_opt)
    train_loader = create_dataloader(train_set, train_dataset_opt, opt)
    val_set = create_dataset(train_dataset_opt)
    val_loader = create_dataloader(val_set, val_dataset_opt, opt)
    print("Start training")
    net.train()
    loss_func = CharbonnierLoss().to(setting.device)
    if not os.path.exists(setting.model_path):
        os.mkdir(setting.model_path)
    for epoch in tqdm(range(opt.epoch), total=opt.epoch, desc="Training ...", position=1, disable=opt.show_progress):
        logger.info("Epoch: {}".format(epoch))
        for current_step, train_data in enumerate(tqdm(train_loader)):

            lr = train_data['LQ'].to(setting.device)
            hr = train_data['GT'].to(setting.device)
            # print("lr.shape:", lr.shape)
            # print("hr.shape:", hr.shape)
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
            print(opt.val_freq)
            if (current_step + 1) % opt.val_freq == 0:
                compute_avg_psnr(val_loader, net)

        if epoch % opt.save_checkpoint_freq == 0 and opt.save_model:
            torch.save(net.state_dict(), os.path.join(setting.model_path, "sr_cb", "{}_{}.pth".format(model_opt['name'], epoch)))
    # compute_avg_psnr(val_loader, net)
    torch.save(net.state_dict(), os.path.join(setting.model_path, "sr_cb", "{}_{}.pth".format(model_opt['name'], opt.epoch)))
    print("\nTraining Complete\n")