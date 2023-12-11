import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import torch.nn.functional
import sys
from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


import setting
from model import create_model
from datasets import create_dataloader, create_dataset
from utils import objectify
import logging


logger = logging.getLogger('base')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_seg_parametrized.py model_yml_path train_yml_path dataset_yml_path")
        exit(1)
    model_yml_path = sys.argv[1]
    train_yml_path = sys.argv[2]
    dataset_yml_path = sys.argv[3]
    model_opt = load(open(model_yml_path, 'r'), Loader=Loader)
    dataset_opt = load(open(dataset_yml_path, 'r'), Loader=Loader)
    train_dataset_opt = dataset_opt['train']
    train_opt = load(open(train_yml_path, 'r'), Loader=Loader)
    opt = objectify(train_opt)
    net = create_model(model_opt)
    net.to(setting.device)
    logname = train_opt['logger']['log_path']
    logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
    start_epoch = 0
    if opt.checkpoint is not None:
        net.load_state_dict(torch.load(opt.checkpoint))
        start_epoch = opt.checkpoint.split("_")[-1].split(".")[0]
    train_set = create_dataset(train_dataset_opt)
    train_loader = create_dataloader(train_set, train_dataset_opt)
    print("Start training")
    net.train()
    loss_func = nn.CrossEntropyLoss()
    if not os.path.exists(setting.model_path):
        os.mkdir(setting.model_path)
    for epoch in tqdm(range(start_epoch, opt.epoch + start_epoch), total=opt.epoch, desc="Training ...", position=1, disable=opt.show_progress):
        logger.info("Epoch: {}".format(epoch))
        for current_step, train_data in enumerate(tqdm(train_loader)):

            x = train_data['input'].to(setting.device)
            y = train_data['label'].to(setting.device)
            pred = net(x)
            
            loss = loss_func(pred, y)
            loss.backward()
            optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
            optimizer.step()
            optimizer.zero_grad()
                    
            #### log
            if current_step % opt.print_freq == 0:
                logger.info(
                    "Epoch: {} | Iteration: {} | Loss: {}".format(epoch, current_step, loss.item()))

        if epoch % opt.save_checkpoint_freq == 0 and opt.save_model:
            torch.save(net.state_dict(), os.path.join(setting.model_path, "segmentation", "{}_{}.pth".format(model_opt['name'], epoch)))
    # compute_avg_psnr(val_loader, net)
    torch.save(net.state_dict(), os.path.join(setting.model_path, "segmentation", "{}_{}.pth".format(model_opt['name'], opt.epoch)))
    print("\nTraining Complete\n")