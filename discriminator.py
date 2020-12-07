import torch
import torch.nn as nn
import torch.nn.functional as fun


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1,bias=False)
        self.conv2 = nn.Conv2d(64,64,3,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,3,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,3,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,256,3,padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,256,3,stride=2,padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256,512,3,padding=1,bias=False)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512,512,3,stride=2,padding=1,bias=False)
        self.bn8 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*16*16,1024)
        self.fc2 = nn.Linear(1024,1)
        self.drop = nn.Dropout2d(0.3)
        
    def forward(self,x):
        layer1 = fun.leaky_relu(self.conv1(x))
        layer2 = fun.leaky_relu(self.bn2(self.conv2(layer1)))
        layer3 = fun.leaky_relu(self.bn3(self.conv3(layer2)))
        layer4 = fun.leaky_relu(self.bn4(self.conv4(layer3)))
        layer5 = fun.leaky_relu(self.bn5(self.conv5(layer4)))
        layer6 = fun.leaky_relu(self.bn6(self.conv6(layer5)))
        layer7 = fun.leaky_relu(self.bn7(self.conv7(layer6)))
        layer8 = fun.leaky_relu(self.bn8(self.conv8(layer7)))
        layer8 = layer8.view(-1,layer8.size(1)*layer8.size(2)*layer8.size(3))
        layer9 = fun.leaky_relu(self.fc1(layer8))
        layer10 = torch.sigmoid(self.drop(self.fc2(layer9)))
        return layer10
