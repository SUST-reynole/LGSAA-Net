#  LGSAA-Net
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from model.unet.unet_parts import *

class DeepWise_PointWise_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeepWise_PointWise_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class depth_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(depth_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            depth_double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class depth_double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(depth_double_conv, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, 3, padding=1),
            DeepWise_PointWise_Conv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, 3, padding=1),
            DeepWise_PointWise_Conv(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class EA_embedding_patches(nn.Module):
    def __init__(self, insize, num,c,k):
        super(EA_embedding_patches, self).__init__()

        ## R
        self.insize = insize
        self.patchsize = int(insize/num)
        self.patchnum = num

        # self.head_esa = nn.ModuleList([External_attention(c) for i in range(patchsize) for j in range(patchsize)])

        self.sa = nn.ModuleList([SELayer_att(channel=c) for i in range(num*num)])

    def forward(self, x):
        stride = self.patchnum
        stride_size = self.patchsize
        id = 0

        att = []


        for i in range(stride):
            for j in range(stride):
                a = i*stride_size
                b = (i+1)*stride_size
                c = j*stride_size
                d = (j+1)*stride_size
                
                y = self.sa[id](x[...,stride_size:2*stride_size,stride_size:2*stride_size])
                att.append(y)
                id = id+1

        return att

class SELayer_att(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #self.squeeze
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

def patchestomlp(x, att, insize, num):

    id = 0
    stride_size = int(insize/num)
    y = torch.zeros_like(x)

    for i in range(num):
            for j in range(num):
                a = i*stride_size
                b = (i+1)*stride_size
                c = j*stride_size
                d = (j+1)*stride_size                
                # y[...,a:b,c:d] = x[...,a:b,c:d] * att[id] + x[...,a:b,c:d]
                y[...,a:b,c:d] = x[...,a:b,c:d] * att[id]
                id = id+1
    return x+y

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class LGSAA_Net(nn.Module): ## ok
    def __init__(self, n_channels, n_classes):
        super(LGSAA_Net, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.inc_p1 = inconv(n_channels, 64)

        self.down1 = depth_down(64, 128)
        self.down2 = depth_down(128, 256)
        self.down3 = depth_down(256, 512)
        self.down4 = depth_down(512, 512) 

        self.down_p1 = depth_down(64, 128)
        self.down_p2 = depth_down(128, 256)
        self.down_p3 = depth_down(256, 512)

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        
        self.outc = outconv(64, n_classes)        

        self.MLP1 = External_attention(64)
        self.MLP2 = External_attention(128)
        self.MLP3 = External_attention(256)
        self.MLP4 = External_attention(512)
        self.MLP5 = External_attention(512)

        self.head_esa1 = EA_embedding_patches(224,2,64,3)
        self.head_esa2 = EA_embedding_patches(112,2,128,3)
        self.head_esa3 = EA_embedding_patches(56,2,256,3)
        self.head_esa4 = EA_embedding_patches(28,2,512,3)
        self.head_esa5 = EA_embedding_patches(14,2,512,3)

        self.sigmoid = nn.Sigmoid()
        size = [224,112,56,28,14]
        self.CSA1 = Adapt_CSA(in_channels=64, size=size[0]) # 64+64
        self.CSA2 = Adapt_CSA(in_channels=2*128,size=size[1]) # 64+64*2
        self.CSA3 = Adapt_CSA(in_channels=2*256,size=size[2]) # 256+256*2
        self.CSA4 = Adapt_CSA(in_channels=2*512, size=size[3]) # 512+512*2

        self.diff_down1 = depth_down(64, 128)
        self.diff_down2 = depth_down(128, 256)
        self.diff_down3 = depth_down(256, 512)
        self.diff_down4 = depth_down(512, 512)

        self.diff2_conv = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.diff3_conv = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.diff4_conv = nn.Conv2d(1024, 512, kernel_size=1, stride=1)
        self.diff5_conv = nn.Conv2d(1024, 512, kernel_size=1, stride=1)

    def forward(self, x, m):  ## x post path; m, pre path
        x1 = self.inc(x)
        m1 = self.inc_p1(m)
        d1 = x1-m1
        fuse1=self.CSA1(d1)
        feature1 = self.diff_down1(fuse1)
        
        x2 = self.down1(x1)
        m2 = self.down_p1(m1)
        d2 = x2-m2
        feature2 = self.CSA2(d2)
        fuse2 = torch.cat((feature2,feature1),dim=1)
        fuse2 = self.diff2_conv(fuse2)
        feature2 = self.diff_down2(fuse2)

        x3 = self.down2(x2)
        m3 = self.down_p2(m2)
        d3 = x3-m3
        feature3 = self.CSA3(d3)
        fuse3 = torch.cat((feature3,feature2),dim=1)
        fuse3 = self.diff3_conv(fuse3)
        feature3 = self.diff_down3(fuse3)

        x4 = self.down3(x3)
        m4 = self.down_p3(m3)
        d4 = x4 - m4

        feature4 = self.CSA4(d4)
        fuse4 = torch.cat((feature4,feature3),dim=1)
        fuse4 = self.diff4_conv(fuse4)
        feature4 = self.diff_down4(fuse4)

        x5 = self.down4(d4)
        # d5 = x5
        fuse5 = torch.cat((x5,feature4), dim=1)
        fuse5 = self.diff5_conv(fuse5)

        # R        
        f1 = self.MLP1(fuse1)
        fsa1 = self.head_esa1(fuse1) #[att1,att2,att3,att4]
        DIFF1 = patchestomlp(f1, fsa1, 224, 2)

        f2 = self.MLP2(fuse2)
        fsa2 = self.head_esa2(fuse2)
        DIFF2 = patchestomlp(f2, fsa2, 112, 2)        

        f3 = self.MLP3(fuse3)
        fsa3 = self.head_esa3(fuse3)
        DIFF3 = patchestomlp(f3, fsa3, 56, 2)

        f4 = self.MLP4(fuse4)   
        fsa4 = self.head_esa4(fuse4)
        DIFF4 = patchestomlp(f4, fsa4, 28, 2)     

        f5 = self.MLP5(fuse5)
        fsa5 = self.head_esa5(fuse5)
        DIFF5 = patchestomlp(f5, fsa5, 14, 2)

        x = self.up1(DIFF5, DIFF4)
        x = self.up2(x, DIFF3)
        x = self.up3(x, DIFF2)
        x = self.up4(x, DIFF1)
        x = self.outc(x)  
        x = self.sigmoid(x)
        return x

class Adapt_CSA(nn.Module): # C,S fusion
    def __init__(self, in_channels,size):
        super(Adapt_CSA, self).__init__()
        gamma=2
        b=1
        # N,C,H,W = x.size()
        t1 = int(abs(np.math.log(in_channels,2)+b)/gamma)
        kc_size = t1 if t1 % 2 else t1 + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kc_size, padding=(kc_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

        self.sigmoid_s = nn.Sigmoid()
        delta=3 #
        b1=1
        # N,C,H,W = x.size()
        t = int(delta*abs(np.math.log(size,10))+b1)
        ks_size = t if t % 2 else t - 1
        self.SA_conv = SpatialAttention_msa(ks_size)
        ## here

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        yc = x * y.expand_as(x)
        yc = self.sigmoid(yc)
        ycs = yc*self.SA_conv(yc)
        z = ycs + x
        return z

class SpatialAttention_msa(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_msa, self).__init__()
 
        assert kernel_size in (3, 5, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        if kernel_size == 5:
            padding = 2

 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.register_buffer()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class eca_layer(nn.Module):
    def __init__(self, channel):        
        super(eca_layer, self).__init__()
        
        gamma=2
        b=1
        t = int(abs(np.math.log(channel,2)+b)/gamma)
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                c_in, c_out, kernel_size=kernel_size, stride=stride, 
                padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class External_attention(nn.Module):
    def __init__(self, c):
        super(External_attention, self).__init__()        
        self.conv1 = nn.Conv2d(c, c, 1)
        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))        

        self.relu = nn.ReLU()

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)   # b * c * n 
        attn = self.linear_0(x) # b, k, n
        attn = F.softmax(attn, dim=-1) # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True)) #  # b, k, n
        x = self.linear_1(attn) # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

