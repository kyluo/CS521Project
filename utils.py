import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import setting

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


def test(x, model):
	x = x.to(setting.device)
    pred = model(x.unsqueeze_(0))
	return pred[0]
    