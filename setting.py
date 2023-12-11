import torch


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
model_path = "model_checkpoints"