import torch
import torch.nn as nn

class PA(nn.Module):
    '''
      Pixel attention mechanism, apply a 1*1 Conv and elementwise multiple with original matrix
    '''

    def __init__(self, num_feature, submodel):
        super(PA, self).__init__()

        conv = nn.Conv2d(num_feature, num_feature, 1)
        sigmoid = nn.Sigmoid()

        self.submodel = submodel
        self.pa_block = nn.Sequential(*[conv, sigmoid])
        

    def forward(self, x):
        conv_x = self.submodel(x)
        new_x = self.pa_block(conv_x)

        out = torch.mul(conv_x, new_x)
        return out