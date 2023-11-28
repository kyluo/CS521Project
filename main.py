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

import setting
from model.model_l_base import UNet_Large_Basic


if __name__ == "__main__":
    net = UNet_Large_Basic(3, 3)
    net.to(setting.device)

    if setting.train:
        print("Start training")
        net.train()
        file_names = load_tensor(opt.train_name_path)
        train_data = DataHolder(
            data_path=opt.train_x_path, label_path=opt.train_y_path)

        train_loader = DataLoader(
            dataset=train_data, batch_size=opt.batch_size, shuffle=True)
        for epoch in tqdm(range(opt.epoch), total=opt.epoch, desc="Training ...", position=1, disable=opt.show_progress):

            loss_G, loss_D = train(train_loader, net, device)

            if (not opt.show_progress):
                print("\n")
                print(
                    f'Loss_G for Training on epoch {str(epoch)} is {str(loss_G)} \n')
                print(
                    f'Loss_D for Training on epoch {str(epoch)} is {str(loss_D)} \n')

            # if epoch % 100 == 0 and opt.save_model:
            #     save_model(net, opt)
        print("\nTraining Complete\n")
        if opt.save_model:
            save_model(net, opt)

    if opt.test:
        file_names = load_tensor(opt.test_name_path)
        test_data = DataHolder(data_path=opt.test_x_path,
                               label_path=opt.test_y_path)

        if opt.load_model:
            net = load_model(net, opt)

        net.eval()
        for i, (data, label) in enumerate(test_data):
            data = data.to(device)
            label = label.to(device)
            pred = net(data.view(1, opt.in_channel, 512, 512))
            save_result(label, pred[0], opt, file_names[i])

