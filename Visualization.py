import torch
from convert import *
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some constants
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)

checkpoint = "/content/drive/MyDrive/model_SRGAN.pth"
srgan= torch.load(checkpoint)['generator'].to(device)
srgan.eval()


img = os.path.join("/content/tom.jpg")
hr_img = Image.open(img,mode='r')
hr_img = hr_img.convert('RGB')


lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                           Image.BICUBIC)
bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

sr_img = srgan(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
sr_img = sr_img.squeeze(0).cpu().detach()

sr_img *= imagenet_std
sr_img +=imagenet_mean

img2 = transforms.ToPILImage(mode='RGB')(sr_img) #Convert into PIL image
img2 #View generated high resolution image

bicubic_img #View low resolution image under bicubic interpolation

hr_img #View Original High resolution image
