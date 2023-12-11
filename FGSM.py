import sys
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import create_dataloader, create_dataset
from utils import objectify
from model.model_l_pa import UNet_Large_PA
from datasets import create_dataloader, create_dataset
import setting
import utils

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

def compute_avg_psnr(val_loader, net, fgsm=False, pgd=False):
    net.eval()
    avg_psnr = 0.
    avg_psnr_fgsm = 0.
    avg_psnr_pgd = 0.
    idx = 0
    img_save_path = ""
    for val_data in tqdm(val_loader):
        idx += 1
        if idx > 100:
            break
        lr = val_data['LQ'].to(setting.device)
        gt = val_data['GT'].to(setting.device)
        sr = net(lr)
        if fgsm:
            L = nn.MSELoss()
            lr_fgsm = FGSM(net, lr, gt, 0.1, L, targeted=False)
            sr_fgsm = net(lr_fgsm)
        if pgd:
            L = nn.MSELoss()
            lr_pgd = pgd_untargeted(net, lr, gt, 10, 0.1, 0.1, L=L)
            sr_pgd = net(lr_pgd)
        sr = sr.detach().cpu()
        gt = gt.detach().cpu()           
        lr = lr.detach().cpu() 
        sr_img = utils.tensor2img(sr)  # uint8
        gt_img = utils.tensor2img(gt)  # uint8
        lr_img = utils.tensor2img(lr)
        avg_psnr += utils.calculate_psnr(sr_img, gt_img)
        if fgsm:
            sr_fgsm = sr_fgsm.detach().cpu()
            sr_fgsm_img = utils.tensor2img(sr_fgsm)  # uint8
            avg_psnr_fgsm += utils.calculate_psnr(sr_fgsm_img, gt_img)
            img_save_path = "results/fgsm/"
        if pgd:
            sr_pgd = sr_pgd.detach().cpu()
            sr_pgd_img = utils.tensor2img(sr_pgd)
            avg_psnr_pgd += utils.calculate_psnr(sr_pgd_img, gt_img)
            img_save_path = "results/pgd/"
        if os.path.exists(img_save_path) == False:
                os.makedirs(img_save_path)
        if idx < 10:
            utils.save_img(lr_img, img_save_path + str(idx) + "_lr.png")
            utils.save_img(sr_img, img_save_path + str(idx) + "_sr.png")
            utils.save_img(gt_img, img_save_path + str(idx) + "_gt.png")
            if fgsm:
                utils.save_img(sr_fgsm_img, img_save_path + str(idx) + "_sr_fgsm.png")
            if pgd:
                utils.save_img(sr_pgd_img, img_save_path + str(idx) + "_sr_pgd.png")
    avg_psnr = avg_psnr / idx
    avg_psnr_fgsm = avg_psnr_fgsm / idx
    avg_psnr_pgd = avg_psnr_pgd / idx

    # log
    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr)) #22.986
    if fgsm:
        logger.info('# Validation # PSNR with FGSM: {:.4e}'.format(avg_psnr_fgsm)) #18.035
    if pgd:
        logger.info('# Validation # PSNR with PGD: {:.4e}'.format(avg_psnr_pgd))
    
def main():
    opt = {
        "train": False,
        "epoch": 1,
        "show_progress": True,
        "save_model": False,
        "print_freq": 10,
        "val_freq": 50,
        "checkpoint": "model_checkpoints/sr/model_l_pa_60.pth",
    }
    val_dataset_opt = {
        "dataroot_GT" : "data/DIV2K_valid_HR",
        "dataroot_LQ" : None,
        "batch_size" : 1,
        "mode" : "LQGT",
        "data_type": "img",
        "phase" : "train",
        "name" : "DIV2K",
        "n_workers" : 1,
        "scale" : 2, # for modcrop in the validation / test phase
        "GT_size" : 1024,
        "color" : "RGB",
        "use_flip": True,
        "use_rot": True,
    }
    opt = objectify(opt)
    net = UNet_Large_PA(3, 3, super_res=True)
    net.to(setting.device)
    if opt.checkpoint is not None:
        net.load_state_dict(torch.load(opt.checkpoint))
    val_set = create_dataset(val_dataset_opt)
    val_loader = create_dataloader(val_set, val_dataset_opt, opt)
    print("Start validation with FGSM attack")
    compute_avg_psnr(val_loader, net, fgsm=True)
    #compute_avg_psnr(val_loader, net, pgd=True)

# The last argument 'targeted' can be used to toggle between a targeted and untargeted attack.
def FGSM(model, features, labels, eps, L=nn.CrossEntropyLoss, targeted=False):
    model.train()
    if targeted:
        features.requires_grad_() # this is required so we can compute the gradient w.r.t x
        eps = eps - 1e-7 # small constant to offset floating-point erros
        targeted_classes = (labels + 1) % 10
        loss = L(model(features), targeted_classes)
        loss.backward()
        grad = features.grad.data 
        adv_features = features - eps * grad.sign()
        return adv_features
    else:
        features.requires_grad_() # this is required so we can compute the gradient w.r.t x
        eps = eps - 1e-7 # small constant to offset floating-point errors
        pred = model(features)
        loss = L(pred, labels)
        loss.backward()
        grad = features.grad.data
        adv_features = features + eps * grad.sign()
        return adv_features

def pgd_untargeted(model, x, labels, num_steps, eps_step, eps, clamp=(0,1), L=nn.CrossEntropyLoss):
    model.train()
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    for _ in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True).cuda()
        prediction = model(_x_adv)
        loss = L(prediction, labels)
        loss.backward()

        with torch.no_grad():
            gradients = _x_adv.grad.sign() * eps_step
            x_adv += gradients
        x_adv = torch.clamp(x_adv, x - eps, x + eps)
        x_adv = x_adv.clamp(*clamp)

    return x_adv.detach()

if __name__ == "__main__":
    main()