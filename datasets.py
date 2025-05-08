import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image



class MaxNorm(object):
    def __call__(self, x):
        return x / x.max()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, kind, transform):
        assert kind in ['train', 'valid', 'test']
        
        master = pd.read_pickle('/home/intern_lhj/researches/retinal_atherosclerosis/label.pkl')
        
        if kind == 'train':
            self.df = master[master['dataset_group'] == 'train'].reset_index(drop = True)
        elif kind == 'valid':
            self.df = master[master['dataset_group'] == 'tune'].reset_index(drop = True)
        elif kind == 'test':
            self.df = master[master['dataset_group'] == 'validation'].reset_index(drop = True)

        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = '/home/DB/SNUHPC/RFI/IMAGESORIGINAL/' + self.df.iloc[index]['pngfilename']
        label = self.df.iloc[index]['carotid_atherosclerosis']
        age = self.df.iloc[index]['AGE'] / 100
        sex = (self.df.iloc[index]['SEX'] == 'M')*1.

        image = Image.open(image_path)
        image = self.transform(image)

        return image, torch.tensor(label).to(torch.float32), torch.tensor(age).to(torch.float32), torch.tensor(sex).to(torch.long)
    
    
def set_datasets(image_size=448):
    
    # train_transform = tfms.Compose([
    #     tfms.RandomResizedCrop(image_size, interpolation = tfms.InterpolationMode.NEAREST),
    #     tfms.RandomRotation(45),
    #     tfms.ToTensor(),
    #     MaxNorm(),
    #     tfms.RandomErasing(0.4, scale = (0.02, 0.2)),
    #     tfms.Normalize(mean = [0.5], std = [0.3]),
    #     tfms.RandomErasing(p=0.25),
    # ])
    
    # valid_transform = tfms.Compose([
    #     tfms.Resize((image_size, image_size), interpolation = tfms.InterpolationMode.NEAREST),
    #     tfms.ToTensor(),
    #     MaxNorm(),
    #     tfms.Normalize(mean = [0.5], std = [0.3]),
    # ])

    # https://github.com/rmaphoh/RETFound_MAE/blob/main/util/datasets.py
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # train transform
    train_transform = create_transform(
        input_size = image_size,
        is_training=True,
        color_jitter = None,
        auto_augment = 'rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob = 0.25,
        re_mode = 'pixel',
        re_count = 1,
        mean = mean,
        std = std,
    )

    valid_transform = []
    valid_transform.append(
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    valid_transform.append(transforms.CenterCrop(image_size))
    valid_transform.append(transforms.ToTensor())
    valid_transform.append(transforms.Normalize(mean, std))
    valid_transform = transforms.Compose(valid_transform)
    
    train_dataset = Dataset('train', train_transform)
    valid_dataset = Dataset('valid', valid_transform)
    test_dataset = Dataset('test', valid_transform)

    
    # return {
    #     'train' : train_dataset, 
    #     'valid' : val_dataset, 
    #     'test' : test_dataset
    # }
    return train_dataset, valid_dataset, test_dataset