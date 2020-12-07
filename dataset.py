from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from convert import *


class Data(Dataset):
  def __init__(self,Dir,df,transform=True,train=True):
    self.df =df
    self.dir = Dir
    self.transform = transform
    self.train=train
    self.len = df.shape[0]
  def __len__(self):
    return self.len;
  def __getitem__(self,idx):
    img_pth = os.path.join(self.dir,self.df["image_id"].loc[idx])
    hr_img = Image.open(img_pth,mode='r')
    lr_img = hr_img.resize((64,64),Image.BICUBIC)
    hr_img = hr_img.resize((256,256),Image.BICUBIC)
  
    if (self.transform):
      lr_img = convert_image(lr_img, source='pil', target='imagenet-norm')
      hr_img = convert_image(hr_img, source='pil', target='imagenet-norm')


    return lr_img,hr_img


