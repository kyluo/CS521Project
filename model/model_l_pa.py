import torch
import torch.nn as nn

from .block_unet import UnetBlock
from .block_pixel import PA

class UNet_Large_PA(nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.5, filter_channel=64):
        super(UNet_Large_PA, self).__init__()
        
        downconv = nn.Conv2d(in_channel, filter_channel, kernel_size=4,
                             stride=2, padding=1)

        uprelu = nn.ReLU(True)
        upconv = nn.ConvTranspose2d(filter_channel*2, out_channel, kernel_size=4, 
                                      stride=2, padding=1)
        
        # (input_nc, outer_nc, filter_nc)
        # 512 to 512 - 512 to 512; inner channel, need dropout layer, does not cancate (no skip layer) layer_modiyier=1
        unet_block = UnetBlock(8*filter_channel, 8*filter_channel, 8*filter_channel, 
                               submodule=None, layer_mod = 1)
        unet_block = PA(8*filter_channel, unet_block)
        
        # 512 to 512 - (512 + 512) to 512
        unet_block = UnetBlock(8*filter_channel, 8*filter_channel, 8*filter_channel, 
                               submodule=unet_block, dropout=dropout)
        unet_block = PA(8*filter_channel, unet_block)

        # 512 to 512 - (512 + 512) to 512
        unet_block = UnetBlock(8*filter_channel, 8*filter_channel, 8*filter_channel, 
                               submodule=unet_block, dropout=dropout)
        unet_block = PA(8*filter_channel, unet_block)
        
        # 512 to 512 - (512 + 512) to 512
        unet_block = UnetBlock(8*filter_channel, 8*filter_channel, 8*filter_channel, 
                               submodule=unet_block, dropout=dropout)
        unet_block = PA(8*filter_channel, unet_block)

        # 256 to 512 - (512 + 512) to 256
        unet_block = UnetBlock(4*filter_channel, 4*filter_channel, 8*filter_channel, 
                               submodule=unet_block)
        unet_block = PA(8*filter_channel, unet_block)

        # 128 to 256 - (256 + 256) to 128
        unet_block = UnetBlock(2*filter_channel, 2*filter_channel, 4*filter_channel, 
                               submodule=unet_block)
        unet_block = PA(8*filter_channel, unet_block)

        # 64 to 128 - (128 + 128) to 64
        unet_block = UnetBlock(filter_channel, filter_channel, 2*filter_channel, 
                               submodule=unet_block)
        # # n to 64 - 64 to m; outter channel
        # unet_block = UnetBlock(in_channel, out_channel, 64, submodule=unet_block)

        model = [downconv] + [unet_block] + [uprelu, upconv, nn.Sigmoid()]
        self.model = nn.Sequential(*model)


    def forward(self, input):
        out = self.model(input)
        # print("Generator out shape ", out.shape)
        return out