import torch
import torch.nn as nn

from .block.unet import UnetBlock
# 2x upsample
class UNet_Large_Basic(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.5, filter_channel=64):
        super(UNet_Large_Basic, self).__init__()
        
        downconv = nn.Conv2d(in_channel, filter_channel, kernel_size=4,
                             stride=2, padding=1)

        uprelu = nn.ReLU(True)
        upconv = nn.ConvTranspose2d(filter_channel*2, out_channel, kernel_size=4, 
                                      stride=4, padding=0)
        
        # block -7 inner
        unet_block = UnetBlock(8*filter_channel, 8*filter_channel, 8*filter_channel, 
                               submodule=None, layer_mod = 1)
        
        # block -6
        unet_block = UnetBlock(8*filter_channel, 8*filter_channel, 8*filter_channel, 
                               submodule=unet_block, dropout=dropout)

        # block -5
        unet_block = UnetBlock(8*filter_channel, 8*filter_channel, 8*filter_channel, 
                               submodule=unet_block, dropout=dropout)
        
        # block -4
        unet_block = UnetBlock(8*filter_channel, 8*filter_channel, 8*filter_channel, 
                               submodule=unet_block, dropout=dropout)
        
        # block -3
        unet_block = UnetBlock(4*filter_channel, 4*filter_channel, 8*filter_channel, 
                               submodule=unet_block)
        
        # block -2
        unet_block = UnetBlock(2*filter_channel, 2*filter_channel, 4*filter_channel, 
                               submodule=unet_block)
        
        # block -1
        unet_block = UnetBlock(filter_channel, filter_channel, 2*filter_channel, 
                               submodule=unet_block)
        

        model = [downconv] + [unet_block] + [uprelu, upconv, nn.Sigmoid()]
        self.model = nn.Sequential(*model)


    def forward(self, input):
        out = self.model(input)
        # print("Generator out shape ", out.shape)
        return out
