# Rodrigo Caye Daudt
# rodrigo.cayedaudt@geod.baug.ethz.ch
# 04/2021

import torch
from glob import glob
from skimage import io
import os
from torchvision import transforms as tr
import warnings


class EuroSAT(torch.utils.data.Dataset):
    """Hight Resolution Semantic Change Detection dataset class, used for both training and test data."""

    def __init__(self, path, train=True):
        """Initialize the dataset.

        Args:
            path ([str]): Base path to EuroSAT dataset files.
            train (bool, optional): Flag to define whether to load train or test data. Defaults to True.
        """
        
        self.train = train
        self.path = path

        self.transform = tr.Compose([
            tr.Pad(5),
            tr.RandomAffine(5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=[-10, 10, -10, 10]),
            tr.RandomRotation(45),
            tr.RandomCrop(64),
            tr.RandomHorizontalFlip(),
            tr.RandomVerticalFlip(),
            tr.RandomErasing(p=0.2),
        ])

        self.CLASSES = {
            'AnnualCrop': 0,
            'Forest': 1,
            'HerbaceousVegetation': 2,
            'Highway': 3,
            'Industrial': 4,
            'Pasture': 5,
            'PermanentCrop': 6,
            'Residential': 7,
            'River': 8,
            'SeaLake': 9,
        }

        self.file_list = []
        self.label_list = []
        for c in self.CLASSES.keys():
            new_files = sorted(glob(os.path.join(self.path, 'train' if self.train else 'test', c, '*.jpg')))
            self.file_list += new_files
            self.label_list += [self.CLASSES[c] for _ in range(len(new_files))]

        
        if len(self.file_list) != 0:
            print('\nEuroSAT dataset ({}) created successfully with {} images.'.format('train' if self.train else 'test', len(self.file_list)))
        else:
            print('\nError creating dataset. Please ensure the path to the dataset folder is correct.\n')
            raise Exception
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        I = io.imread(self.file_list[idx]).transpose((2,0,1)) / 64.0 - 1.0
        I = torch.from_numpy(I).float()

        if self.train:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                I = self.transform(I)

        label = torch.tensor(self.label_list[idx]).long()
        

        sample = {'image': I, 'label': label}

        return sample