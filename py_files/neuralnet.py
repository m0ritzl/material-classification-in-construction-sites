import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor, is_tensor


#Resnet originial implementation for imagenet
class BasicBlock(nn.Module):
     expansion = 1

     def __init__(self, in_planes, planes, stride=1):
         super(BasicBlock, self).__init__()
         self.conv1 = nn.Conv2d(
             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
         self.bn1 = nn.BatchNorm2d(planes)
         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=1, padding=1, bias=False)
         self.bn2 = nn.BatchNorm2d(planes)

         self.shortcut = nn.Sequential()
         if stride != 1 or in_planes != self.expansion*planes:
             self.shortcut = nn.Sequential(
                 nn.Conv2d(in_planes, self.expansion*planes,
                           kernel_size=1, stride=stride, bias=False),
                 nn.BatchNorm2d(self.expansion*planes)
             )

     def forward(self, x):
         out = F.relu(self.bn1(self.conv1(x)))
         out = self.bn2(self.conv2(out))
         out += self.shortcut(x)
         out = F.relu(out)
         return out


class Bottleneck(nn.Module):
     expansion = 4

     def __init__(self, in_planes, planes, stride=1):
         super(Bottleneck, self).__init__()
         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
         self.bn1 = nn.BatchNorm2d(planes)
         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                stride=stride, padding=1, bias=False)
         self.bn2 = nn.BatchNorm2d(planes)
         self.conv3 = nn.Conv2d(planes, self.expansion *
                                planes, kernel_size=1, bias=False)
         self.bn3 = nn.BatchNorm2d(self.expansion*planes)

         self.shortcut = nn.Sequential()
         if stride != 1 or in_planes != self.expansion*planes:
             self.shortcut = nn.Sequential(
                 nn.Conv2d(in_planes, self.expansion*planes,
                           kernel_size=1, stride=stride, bias=False),
                 nn.BatchNorm2d(self.expansion*planes)
             )

     def forward(self, x):
         out = F.relu(self.bn1(self.conv1(x)))
         out = F.relu(self.bn2(self.conv2(out)))
         out = self.bn3(self.conv3(out))
         out += self.shortcut(x)
         out = F.relu(out)
         return out


class ResNet(nn.Module):
     def __init__(self, block, num_blocks, num_classes=23):
         super(ResNet, self).__init__()
         self.in_planes = 64

         self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False)
         self.bn1 = nn.BatchNorm2d(64)
         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
         self.linear = nn.Linear(512*block.expansion, num_classes)

     def _make_layer(self, block, planes, num_blocks, stride):
         strides = [stride] + [1]*(num_blocks-1)
         layers = []
         for stride in strides:
             layers.append(block(self.in_planes, planes, stride))
             self.in_planes = planes * block.expansion
         return nn.Sequential(*layers)

     def forward(self, x):
         out = F.relu(self.bn1(self.conv1(x)))
         out = self.layer1(out)
         out = self.layer2(out)
         out = self.layer3(out)
         out = self.layer4(out)
         out = F.avg_pool2d(out, 4)
         out = out.view(out.size(0), -1)
         out = self.linear(out)
         return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


## ResNet adapted to CIFAR10 dataset
##code from: https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/
## 3x3 convolution
#def conv3x3(in_channels, out_channels, stride=1):
#    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                     stride=stride, padding=1, bias=False)
#
## Residual block
#class ResidualBlock(nn.Module):
#    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#        super(ResidualBlock, self).__init__()
#        self.conv1 = conv3x3(in_channels, out_channels, stride)
#        self.bn1 = nn.BatchNorm2d(out_channels)
#        self.relu = nn.ReLU(inplace=True)
#        self.conv2 = conv3x3(out_channels, out_channels)
#        self.bn2 = nn.BatchNorm2d(out_channels)
#        self.downsample = downsample
#
#    def forward(self, x):
#        residual = x
#        out = self.conv1(x)
#        out = self.bn1(out)
#        out = self.relu(out)
#        out = self.conv2(out)
#        out = self.bn2(out)
#        if self.downsample:
#            residual = self.downsample(x)
#        out += residual
#        out = self.relu(out)
#        return out
#
## ResNet
#class ResNet(nn.Module):
#    def __init__(self, block, layers, num_classes=8): #adapted the classes to 9
#        super(ResNet, self).__init__()
#        self.in_channels = 16
#        self.conv = conv3x3(3, 16)
#        self.bn = nn.BatchNorm2d(16)
#        self.relu = nn.ReLU(inplace=True)
#        self.layer1 = self.make_layer(block, 16, layers[0])
#        self.layer2 = self.make_layer(block, 32, layers[1], 2)
#        self.layer3 = self.make_layer(block, 64, layers[2], 2)
#        self.avg_pool = nn.AvgPool2d(8)
#        self.fc = nn.Linear(64, num_classes)
#
#    def make_layer(self, block, out_channels, blocks, stride=1):
#        downsample = None
#        if (stride != 1) or (self.in_channels != out_channels):
#            downsample = nn.Sequential(
#                conv3x3(self.in_channels, out_channels, stride=stride),
#                nn.BatchNorm2d(out_channels))
#        layers = []
#        layers.append(block(self.in_channels, out_channels, stride, downsample))
#        self.in_channels = out_channels
#        for i in range(1, blocks):
#            layers.append(block(out_channels, out_channels))
#        return nn.Sequential(*layers)
#
#    def forward(self, x):
#        out = self.conv(x)
#        out = self.bn(out)
#        out = self.relu(out)
#        out = self.layer1(out)
#        out = self.layer2(out)
#        out = self.layer3(out)
#        out = self.avg_pool(out)
#        out = out.view(out.size(0), -1)
#        out = self.fc(out)
#        return out

class SchmidhuberNet(nn.Module):
    def __init__(self, num_classes=16):
        super(SchmidhuberNet, self).__init__()
        # layer0 with 1 map of 95x95
        # layer1 with 48 maps of 92x92
        # layer2 with 48 maps of 46x46
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # layer3 with 48 maps of 42x42
        # layer4 with 48 maps auf 21x21
        self.layer2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # layer5 with 48 maps of 18x18
        # layer6 with 48 maps of 9x9
        self.layer3 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # layer7 with 48 maps of 6x6
        # layer8 with 48 maps of 3x3
        self.layer4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # layer9 with 200
        self.fc1 = nn.Linear(48*3*3, 200)
        # layer10 with 16
        self.fc2 = nn.Linear(200, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

#pytorch tutorial net
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class TutorialNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
