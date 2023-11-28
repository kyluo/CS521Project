import torch
import torch.nn as nn

class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, filter_nc,
                 submodule=None, dropout=1, layer_mod=2, 
                 if_PA=False, name=" "):

        super(UnetBlock, self).__init__()

        downrelu = nn.LeakyReLU(0.1, inplace=True)
        downconv = nn.Conv2d(input_nc, filter_nc, kernel_size=4,
                             stride=2, padding=1)
        downnorm = nn.BatchNorm2d(filter_nc)

        uprelu = nn.ReLU(inplace=True)
        upconv = nn.ConvTranspose2d(filter_nc*layer_mod, outer_nc, kernel_size=4,
                                    stride=2, padding=1)
        upnorm = nn.BatchNorm2d(outer_nc)

        down = [downrelu, downconv, downnorm]
        up = [uprelu, upconv, upnorm]

        model = down + up
        if submodule != None:
            model = down + [submodule] + up
        if dropout < 1:
            model += [nn.Dropout(dropout)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = torch.cat([x, self.model(x)], 1)
        return out