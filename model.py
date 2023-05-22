
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k=3, p=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=k,padding=p,  stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=k,padding=p,
                              stride=1, bias=False)        
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))            
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlockG(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, k=3, p=1):
        super(BasicBlockG, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=k, padding=p, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=k,padding=p, 
                               stride=1, bias=False)        
        self.bn2 = nn.BatchNorm2d(planes)
       
    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))         

        out = F.relu(out)
        return out




class LevelBlockGM(nn.Module):
    expansion = 1

    def __init__(self, planes, stride=1, n=1,  k=3, p=1, hw=32):
        super(LevelBlockGM, self).__init__()
        layers = []
        
        for i in range(n):
            layers.append(BasicBlock(planes, planes, stride, k, p)) 
            #pl= planes      
        self.layer = nn.Sequential(*layers)

        self.con0 = nn.Conv2d(planes, planes, kernel_size=1, padding=0, stride=1, bias=False)             
        self.bn0 = nn.BatchNorm2d(planes)

        self.conv1g = nn.Conv2d(planes, planes, kernel_size=1, padding=0, stride=1, bias=False)             
        self.bn1g = nn.BatchNorm2d(planes)
        self.conv2g = nn.Conv2d(planes, planes, kernel_size=1,padding=0, stride=1, bias=False)        
        self.bn2g = nn.BatchNorm2d(planes)

        self.fc1_1 = nn.Linear(planes, planes)
        self.fc2_1 = nn.Linear(planes, planes)
        self.ln11 = nn.LayerNorm(planes)
        self.ln21 = nn.LayerNorm(planes)
        

        self.fc1_2 = nn.Linear(planes, 2*planes, bias=True)
        self.fc1_3 = nn.Linear(planes, planes, bias=True)
        self.fc1_4 = nn.Linear(planes, 2*2, bias=True)
        self.fc1_5 = nn.Linear(planes, 2, bias=True)
        

        
        self.hw = hw
        self.df = planes
       

       
    def forward(self, x, m, xb, gridt, b):
        
        #resnet blocks image feature
        x = F.relu(self.bn0(self.con0(xb))+x)
        x = self.layer(x)
        
        #affine parameters
        xy1_ = F.relu(self.ln11(self.fc1_1(m)))
        xy1_ = F.relu(self.ln11(self.fc1_1(xy1_)))
        xy_b =  self.fc1_3(xy1_).view(-1, self.df, 1,1)
        xy_ = self.fc1_2(xy1_).view(-1,self.df , 2)
        xy_b1 =  self.fc1_5(xy1_).view(-1, 2, 1,1)
        xy_1 = self.fc1_4(xy1_).view(-1,2 , 2)
        
        #transform coordinate grid, first single affine, second for all channel
        gridt = torch.matmul(xy_1, gridt.view(-1, 2, self.hw*self.hw)).view(-1, 2, self.hw, self.hw) + xy_b1
        g1 = torch.matmul(xy_, gridt.view(-1, 2, self.hw*self.hw)).view(-1, self.df, self.hw, self.hw) + xy_b
        
        #generate bases
        b = F.relu(self.bn1g(self.conv1g(g1))+b)
        b = F.relu(self.bn2g(self.conv2g(b)))
        
        #compute moments
        xb = b*x 
        m =  torch.flatten(F.avg_pool2d(xb, x.shape[2]), 1)

        return x, xb, m, b, gridt


class DGMResNet(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(DGMResNet, self).__init__()

        self.hw = 32
        self.df=256

        h = (self.hw-1)
        a = (torch.Tensor(range(self.hw)))/(h)
        g = torch.meshgrid(a, a)
        self.gridt = nn.Parameter(torch.cat((g[0].view(1, 1, self.hw,self.hw), g[1].view(1, 1, self.hw,self.hw),),dim=1), requires_grad=False)

        self.layer01 = BasicBlockG( self.df, self.df, stride=1, k=1, p=0)
        self.conv11 = nn.Conv2d(2, self.df, kernel_size=1, stride=1, bias=False)
        self.layer02 = self._make_layer(block, self.df, 2, stride=1, k=3, p=1)
        self.conv02 = nn.Conv2d(3, self.df, kernel_size=5, stride=1, padding=2)

        self.lvl2 = LevelBlockGM(self.df, n=2, k=3, p=1, hw=self.hw)
        self.lvl3 = LevelBlockGM(self.df, n=2, k=3, p=1, hw=self.hw)
        self.lvl4 = LevelBlockGM(self.df, n=2, k=3, p=1, hw=self.hw)
        self.do = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.df, num_classes)
   
    def _make_layer(self, block, planes, num_blocks, stride, k=3, p=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(planes, planes, stride, k, p))
            #self.in_planes = planes 
        return nn.Sequential(*layers)

    def forward(self, x):
        #size = (x.shape[2], x.shape[3])
        #first level
        gridt = self.gridt
        bases = self.layer01(self.conv11(gridt))
        x = self.layer02(self.conv02(x))
        xb = bases*x
        m = torch.flatten(F.avg_pool2d(xb, x.shape[2]), 1)
        #second level onward
        x, xb, m, bases, gridt = self.lvl2(x, m, xb, gridt, bases)
        x, xb, m, bases, gridt = self.lvl3(x, m, xb, gridt, bases)
        x, xb, m, bases, gridt = self.lvl4(x, m, xb, gridt, bases)

        cl = self.linear(self.do(m))

        #visualization
        # imgr = torch.sum(xb*(m.view(-1, m.shape[1], 1, 1)), dim=1, keepdim=True)
        # imgr = imgr.view(imgr.size(0), -1)
        # imgr = imgr - imgr.min(1, keepdim=True)[0]
        # imgr = imgr/imgr.max(1, keepdim=True)[0]
        # imgr = (imgr.view(-1, 1, self.hw, self.hw))
        # imgr = nn.Upsample(size, mode='bilinear', align_corners=True)(imgr)

        return cl


def ResNet18(num_classes=100):
    return DGMResNet(BasicBlock, num_classes=num_classes)
