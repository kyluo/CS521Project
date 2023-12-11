import torch
import torch.nn as nn

from .block.unet import UnetBlock

class UNet_Small_Base(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, dropout=0.5, filter_channel=64, super_res=False):
        super(UNet_Small_Base, self).__init__()
        
        # block -4 inner
        unet_block = UnetBlock(4*filter_channel, 4*filter_channel, 4*filter_channel, 
                               submodule=None, layer_mod = 1, name="inner")

        # block -3
        unet_block = UnetBlock(4*filter_channel, 4*filter_channel, 4*filter_channel, 
                               submodule=unet_block, name="block3")

        # block -2
        unet_block = UnetBlock(2*filter_channel, 2*filter_channel, 4*filter_channel, 
                               submodule=unet_block, name="block2")

        # block -1
        unet_block = UnetBlock(filter_channel, filter_channel, 2*filter_channel, 
                               submodule=unet_block, name="block1")

        downconv = nn.Conv2d(in_channel, filter_channel, kernel_size=4,
                             stride=2, padding=1)
        
        uprelu = nn.ReLU(True)
        upconv = nn.ConvTranspose2d(filter_channel*2, out_channel, kernel_size=4, 
                                      stride=2, padding=1)
        
        model = [downconv] + [unet_block] + [uprelu, upconv, nn.Sigmoid()]

        if super_res:
            upconv1 = nn.ConvTranspose2d(filter_channel*2, out_channel*2, kernel_size=4, 
                                      stride=2, padding=1)
            upnorm = nn.BatchNorm2d(out_channel*2)

            upconv2 = nn.ConvTranspose2d(out_channel*2, out_channel, kernel_size=4,
                                    stride=2, padding=1)
            model = [downconv] + [unet_block] + [uprelu, upconv1, upnorm, uprelu, upconv2, nn.Sigmoid()]
    
  
        self.model = nn.Sequential(*model)


    def forward(self, input):
        out = self.model(input)
        # print("Generator out shape ", out.shape)
        return out