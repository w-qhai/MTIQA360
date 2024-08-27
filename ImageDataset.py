import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms import ToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_loader(image_name):
    I = Image.open(image_name)
    return I.convert('RGB')

class ImageDataset(Dataset):
    def __init__(self, 
                 csv_file,
                 part,
                 img_dir,
                 test,):

        self.csv_file = csv_file
        self.part = part
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = image_loader


    def __getitem__(self, index):
        name, mos, distortion = self.csv_file[self.part[index]-1].split(',')
        I = []        
        for i in range(20):
            if "oiqa" in self.img_dir:
                image_name = os.path.join(self.img_dir, name+".png", f"{i}.png")
            elif "cviq" in self.img_dir:
                image_name = os.path.join(self.img_dir, name+".png", f"{i}.png")
                
            I.append(ToTensor()(self.loader(image_name)))
        I = torch.stack(I)
         
        mos = float(mos) / 10
        if "gaussian" in distortion:
            distortion = distortion[len("gaussian")+1: ]
            
        sample = {'I': I, 'mos': float(mos), 'distortion': distortion}
        return sample

    def __len__(self):
        return len(self.part)
    