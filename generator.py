import torch
import torch.nn as nn



class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3,64,9,stride =1,padding=4, bias = False)
    self.conv2 = nn.Conv2d(64,64,3, stride=1,padding=1, bias =False)
    self.conv3_a = nn.Conv2d(64,256,3, stride=1, padding=1, bias=False)
    self.conv3_b = nn.Conv2d(64,256,3, stride=1, padding=1, bias=False)
    self.conv4 = nn.Conv2d(64,3,9, stride=1, padding=4, bias=False)
    self.ps = nn.PixelShuffle(2)
    self.bn = nn.BatchNorm2d(64)
    self.prelu = nn.PReLU()
  def forward(self,x):
    layer1 = self.conv1(x)
    layer1 = self.prelu(layer1)
    layer2 = torch.add(self.prelu(self.bn(self.conv2(self.prelu(self.bn(self.conv2(layer1)))))),layer1)
    layer3 = torch.add(self.prelu(self.bn(self.conv2(self.prelu(self.bn(self.conv2(layer2)))))),layer2)
    layer4 = torch.add(self.prelu(self.bn(self.conv2(self.prelu(self.bn(self.conv2(layer3)))))),layer3)
    layer5 = torch.add(self.prelu(self.bn(self.conv2(self.prelu(self.bn(self.conv2(layer4)))))),layer4)
    layer6 = torch.add(self.prelu(self.bn(self.conv2(self.prelu(self.bn(self.conv2(layer5)))))),layer5)
    layer7 = torch.add(self.bn(self.conv2(layer6)),layer1)
    layer8 = self.prelu(self.ps(self.conv3_a(layer7)))
    layer9 = self.prelu(self.ps(self.conv3_b(layer8)))
    layer10 = self.conv4(layer9)
    return layer10;
