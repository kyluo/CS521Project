import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional
import shutil
import argparse


class DataHolder(Dataset):
    def __init__(self, data_path=None, label_path=None):
        self.data = load_tensor(data_path)
        self.label = load_tensor(label_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        data, label = None, None

        # transform = torch.nn.Sequential(
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomPerspective(),
        #     transforms.RandomRotation(90)
        # )

        if self.data != None:
            data = self.data[idx]
        if self.label != None:
            label = self.label[idx]
        return data, label
    
'''
This function saves tensor into .pt file
'''
def save_tensor(tensor, path, name):
    file_name = os.path.join(path, name)
    torch.save(tensor, file_name)
    print("Saved tensor to ", file_name)

'''
This function saves tensor into .pt file
'''
def load_tensor(path):
    try:
        print("Load tensor from file ", path)
        return torch.load(path)
    except:
        return None

'''
This function gets all the data in current channel
'''
def get_data_from_feature(path, file_names):
    data_arr = []
    img2tensor = transforms.ToTensor()

    for file in file_names:
        data_path = os.path.join(path, file)
        if data_path[-4:] == ".png":
            data_img = Image.open(data_path).convert('L')
            data = img2tensor(data_img)
            # data = data.type(torch.float64) # Prevent overflow and underflow
            data_img.close()
            data_arr.append(data)
    return torch.cat(data_arr, 0)

'''
This function retrives all feature data in dataset path
'''
def get_all_data(path, file_names):
    data_arr = []
    for feature in sorted(os.listdir(path)):
        feature_path = os.path.join(path, feature)
        feature_data = get_data_from_feature(feature_path, file_names)
        data_arr.append(feature_data)
    # we want the data in the shape (n, channel, h, w)
    data_arr = torch.stack(data_arr).permute(1, 0, 2, 3)
    return data_arr

'''
This function iterate through the data file for each channel (train and label),
and find the intersection of file names present in all folders.
'''
def data_validation(data_path, label_path):
    # Get chennels for both data and label
    label_channels = os.listdir(label_path)
    feature_channels = os.listdir(data_path)

    channel_name = []
    all_file_name = []
    # Get all filenames in the feature folder
    for feature in feature_channels:
        feature_path = os.path.join(data_path, feature)
        data = os.listdir(feature_path)
        all_file_name.append(data)
        channel_name.append(feature_path)

    # get all filenames in the label folder
    for feature in label_channels:
        feature_path = os.path.join(label_path, feature)
        data = os.listdir(feature_path)
        all_file_name.append(data)
        channel_name.append(feature_path)

    intersection = set.intersection(*map(set,all_file_name))
    return list(intersection)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save png into torch tensor')
    parser.add_argument('--load_path', required=True, type=str, help='path to dataset')
    parser.add_argument('--save_path', type=str, default="./datasets", help='path to dataset')
    parser.add_argument('--feature_file', type=str, default="feature_label_info.txt", help='file hold feature names')
    opt = parser.parse_args()
    file_type = ["train", "val", "test"]

    src = os.path.join(opt.load_path, opt.feature_file)
    dest = os.path.join(opt.save_path, opt.feature_file)
    shutil.copyfile(src, dest)

    for each_type in file_type:
        temp_path = os.path.join(opt.load_path, each_type)
        data_load_path = os.path.join(temp_path, "data")
        label_load_path = os.path.join(temp_path, "label")

        file_names = data_validation(data_load_path, label_load_path)

        data_data = get_all_data(data_load_path, file_names)
        print(each_type, " data shape: ", data_data.shape, "data type: ", data_data.dtype)
        label_data = get_all_data(label_load_path, file_names)
        print(each_type, " label shape: ", label_data.shape, "label type: ", data_data.dtype)

        if not os.path.exists(os.path.join(opt.save_path, each_type)):
            os.makedirs(os.path.join(opt.save_path, each_type))
    
        save_tensor(data_data, os.path.join(opt.save_path, each_type), "%s_x.pt" % (each_type))
        save_tensor(label_data, os.path.join(opt.save_path, each_type), "%s_y.pt" % (each_type))
        save_tensor(file_names, os.path.join(opt.save_path, each_type), "%s_name.pt" % (each_type))

    