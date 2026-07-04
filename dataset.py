import os
import random
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


class GoProDataset(Dataset):
    def __init__(self, path, crop_size, subset, max_images=None, val_ratio=0.10): 
        ''' 
        Args:
            path (str): filepath to dataset
            crop_size (int): size of cropped images (must be a multiple of 8)
            subset (str): should be set to "train", "val" or "test" to specify whether to access the training or testing sets
        '''
        self.crop_size = crop_size
        self.to_tensor = transforms.ToTensor()

        split_dir = "train" if subset in ["train", "val"] else "test"
        self.subset = subset

        # searching through subfolders in the dataset
        search_path = os.path.join(path, split_dir, '*', 'blur', '*.png')
        self.blur_paths = sorted(glob(search_path))
        self.sharp_paths = [path.replace('blur', 'sharp') for path in self.blur_paths]

        if self.subset in ["train", "val"]:
            # deterministically shuffle the paths
            combined = list(zip(self.blur_paths, self.sharp_paths))
            random.Random(42).shuffle(combined) 
            self.blur_paths, self.sharp_paths = zip(*combined)
        
            self.blur_paths = list(self.blur_paths)
            self.sharp_paths = list(self.sharp_paths)
        
            # calculate index at the boundary of the training and validation sets
            split_index = int(len(self.blur_paths) * val_ratio)
            
            # slice the lists to isolate either the training or validation set
            if self.subset == "train":
                self.blur_paths = self.blur_paths[split_index:]
                self.sharp_paths = self.sharp_paths[split_index:]
            else:
                self.blur_paths = self.blur_paths[:split_index]
                self.sharp_paths = self.sharp_paths[:split_index]

        if max_images is not None:
            self.blur_paths = self.blur_paths[:max_images]
            self.sharp_paths = self.sharp_paths[:max_images]

    def __len__(self):
        return len(self.blur_paths)
    
    def __getitem__(self, i):
        # load images
        blur_img  = Image.open(self.blur_paths[i]).convert('RGB')
        sharp_img = Image.open(self.sharp_paths[i]).convert('RGB')

        if self.subset == "train":
            # crop images
            x, y, h, w = transforms.RandomCrop.get_params(blur_img, output_size = (self.crop_size, self.crop_size))
            blur_img = TF.crop(blur_img, x, y, h, w)
            sharp_img = TF.crop(sharp_img, x, y, h, w)

            # randomly flip image horizontally 
            if random.random() > 0.5:
                blur_img = TF.hflip(blur_img)
                sharp_img = TF.hflip(sharp_img)
            # randomly flip image vertically
            if random.random() > 0.5:
                blur_img = TF.vflip(blur_img)
                sharp_img = TF.vflip(sharp_img)
        else:
            blur_img = TF.center_crop(blur_img, self.crop_size)
            sharp_img = TF.center_crop(sharp_img, self.crop_size)
        
        return self.to_tensor(blur_img), self.to_tensor(sharp_img)
