import torch
from PIL import Image
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import datasets.util as util

class SegmentationDataset(data.Dataset):
    def __init__(self, dataset_opt):
        super(SegmentationDataset, self).__init__()
        self.opt = dataset_opt
        self.GT_size = self.opt['GT_size']
        self.paths_input, _ = util.get_image_paths("img", self.opt['dataroot_input'])
        self.paths_label, _ = util.get_image_paths("img", self.opt['dataroot_label'])
        assert self.paths_input, 'Error: input path is empty.'
        assert self.paths_label, 'Error: label path is empty.'
        assert len(self.paths_input) == len(self.paths_label), 'input and label datasets have different number of images - {}, {}.'.format(len(self.paths_input), len(self.paths_label))
        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize((self.GT_size, self.GT_size), antialias=True),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
        ])
        
    
    def __getitem__(self, index):
        input_path = self.paths_input[index]
        label_path = self.paths_label[index]
        input = Image.open(input_path)
        label = Image.open(label_path)
        input = self.transform(input).float()
        input = input / 255.0
        label = self.transform(label).long().squeeze(0)
        return {'input': input, 'label': label}
    
    def __len__(self):
        return len(self.paths_input)