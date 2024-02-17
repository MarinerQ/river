import torch
from torch import nn
import torch.nn.functional as F

'''
class ResidualConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(residual)
        out = F.relu(out)
        return out
'''
class ResidualConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        out += self.downsample(residual)
        out = F.relu(out)
        return out

class EmbeddingConv1D(nn.Module):
    def __init__(self, ndet, nout, num_blocks, use_psd = True, middle_channel = 32, kernel_size=1, stride=1, padding=0, dilation=1, dropout=0.5, **kwargs):
        super().__init__()
        #self.ncomp = ncomp
        self.nout = nout
        if use_psd:
            self.nchannel = 3*ndet # strains(2) + PSD (1)
        else:
            self.nchannel = 2*ndet

        self.middle_channel = middle_channel
        self.layers = nn.ModuleList([self.make_layer(ResidualConvBlock1D, self.middle_channel, kernel_size, stride, padding, dilation, dropout) for _ in range(num_blocks)])
        self.linear = nn.Linear(self.middle_channel, self.nout)

    def make_layer(self, block, out_channels, kernel_size, stride, padding, dilation, dropout):
        layers = []
        layers.append(block(self.nchannel, out_channels, kernel_size, stride, padding, dilation, dropout))
        self.nchannel = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # x : [batch_size, channel (det_123, amp/phase) = 2*ndet, length (number of samples)]
        # bs,_,_  = x.shape 
        for layer in self.layers:
            x = layer(x)
        x = F.avg_pool1d(x, x.size()[2])
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        return output

class ResidualConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels or kernel_size!=1 or stride!=1 or padding!=0 or dilation!=1:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation) 
        else:
            self.downsample = nn.Identity()

        '''
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        '''

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(residual)
        out = F.relu(out)
        return out

class EmbeddingConv2D(nn.Module):
    def __init__(self, nout, num_blocks, use_psd = True, middle_channel = 16, kernel_size=1, stride=1, padding=0, dilation=1, **kwargs):
        super().__init__()
        #self.ncomp = ncomp
        #self.ndet=ndet # (ndet, ncomp) is the shape of each channel
        self.nout = nout
        if use_psd:
            self.nchannel = 3 # strains(2) + PSD (1)
        else:
            self.nchannel = 2

        self.middle_channel = middle_channel
        self.layers = nn.ModuleList([self.make_layer(ResidualConvBlock2D, self.middle_channel, kernel_size, stride, padding, dilation) for _ in range(num_blocks)])
        self.linear = nn.Linear(self.middle_channel, self.nout)

    def make_layer(self, block, out_channels, kernel_size, stride, padding, dilation):
        layers = []
        layers.append(block(self.nchannel, out_channels, kernel_size, stride, padding, dilation))
        self.nchannel = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # x : [batch_size, channel (det_123, amp/phase) = 2*ndet, length (number of samples)]
        # bs,_,_,_  = x.shape 
        for layer in self.layers:
            x = layer(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        return output

#I have a 2D input (or, a matrix) with shape (6, 512). There are 6 rows, let call them a1, a2, b1, b2, c1, c2. The features I want to extract are correlations between these rows, e.g, between a1 and a2, or a1 and b1, or a1 and c1 and so forth. As you can see, there are different "scales": a1 and a2 are close so we need a small conv kernel, but a1 and c1 are far way so we need a big conv kernel. How to design a CNN to extract features for this input? 


class MyEmbeddingConv2D(nn.Module):
    def __init__(self, nout, **kwargs):
        super().__init__()

        self.nout = nout
        self.layers = nn.ModuleList([self.make_layer()])
        self.linear = nn.Linear(3, self.nout)

    def make_layer(self):
        layers = []
        layers.append(ResidualConvBlock2D(in_channels=1, out_channels=1,
                                          kernel_size=(3,2), stride=(1,1), dilation=(1,1), padding = (2,0)))
        layers.append(ResidualConvBlock2D(in_channels=8, out_channels=16,
                                          kernel_size=(3,4), stride=(1,1), dilation=(1,1), padding = (2,0)))
        layers.append(ResidualConvBlock2D(in_channels=16, out_channels=16,
                                          kernel_size=(3,2), stride=(1,1), dilation=(1,1), padding = (0,0)))
        layers.append(ResidualConvBlock2D(in_channels=16, out_channels=3,
                                          kernel_size=(3,2), stride=(1,1), dilation=(1,1), padding = (0,0)))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x : [batch_size, channel (det_123, amp/phase) = 2*ndet, length (number of samples)]
        # bs,_,_,_  = x.shape
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        return output