import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import setting
import math
import numpy as np

def loss_MSE():
     return nn.MSELoss()


def train(data_loader, model):
    model.train()
    it_train = tqdm(enumerate(data_loader), total=len(
        data_loader), desc="Training ...", position=0, disable=False)
    
    loss_func = loss_MSE()
    optimizer = optim.Adam(model.paramters(), lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))

    for _, (x, y) in it_train:
        x, y = x.to(setting.device), y.to(setting.device)

        optimizer.zero_grad()
        y_pred = model.forward(x)
        loss = loss_func(y, y_pred)
        loss.backward()
        optimizer.step()
        

def val(data_loader, model):
    loss_func = loss_MSE()
    model.eval()
    
    total_loss = 0
    for _, (x, y) in enumerate(data_loader):
        y_pred = model.forward(x)
        loss = loss_func(y, y_pred)
        total_loss += loss
    return total_loss

def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def test(x, model):
    x = x.to(setting.device)
    pred = model(x.unsqueeze_(0))
    return pred[0]

class objectify(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [objectify(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, objectify(v) if isinstance(v, dict) else v)