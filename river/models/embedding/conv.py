import torch
from torch import nn
import torch.nn.functional as F
from glasflow.nflows.nn.nets import ResidualNet

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
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0, dilation=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        if in_channels != out_channels or kernel_size!=1 or stride!=1 or padding!=0 or dilation!=1:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation) 
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = F.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        out += self.downsample(residual)
        out = F.elu(out)
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


class MyEmbeddingConv1D(nn.Module):
    def __init__(self, ndet, nbasis, nout,  **kwargs):
        super().__init__()
        self.nout = nout
        self.nchannel = 2*ndet
        self.nbasis = nbasis
        
        self.conv_out_channel = 4
        self.conv = self.make_conv_layer()
        
        with torch.no_grad():
            testx = torch.randn((1, self.nchannel, self.nbasis))
            self.mlp_in_channel = self.conv(testx).shape[-1] * self.conv_out_channel
            print(f"Initialized MLP in channel: {self.mlp_in_channel}")

        self.mlp = self.make_mlp_layer(self.mlp_in_channel)

    def make_conv_layer(self):
        

        layers = []
        layers.append(ResidualConvBlock1D(self.nchannel, 64, kernel_size=32, stride=1, padding=2, dilation=1, dropout=0.))
        layers.append(ResidualConvBlock1D(64, 64, kernel_size=16, stride=1, padding=0, dilation=1, dropout=0.))
        layers.append(ResidualConvBlock1D(64, 64, kernel_size=16, stride=1, padding=0, dilation=1, dropout=0.))
        layers.append(ResidualConvBlock1D(64, 64, kernel_size=8, stride=2, padding=0, dilation=1, dropout=0.))
        layers.append(ResidualConvBlock1D(64, 64, kernel_size=8, stride=1, padding=0, dilation=1, dropout=0.))
        layers.append(ResidualConvBlock1D(64, 16, kernel_size=8, stride=1, padding=0, dilation=2, dropout=0.))
        layers.append(ResidualConvBlock1D(16, self.conv_out_channel, kernel_size=4, stride=1, padding=0, dilation=1, dropout=0.))
        #layers.append(ResidualConvBlock1D(4, self.conv_out_channel, kernel_size=2, stride=1, padding=0, dilation=1, dropout=0.1))
        
        return nn.Sequential(*layers)

    def make_mlp_layer(self, in_channel):
        layers = []
        #layers.append(torch.nn.Linear(self.nbasis * self.conv_out_channel, 256))
        #layers.append(torch.nn.ReLU())
        #layers.append(ResidualMLPBlock1D(in_channel,  self.nout, 3))
        layers.append(ResidualNet(in_features = in_channel,
                                  out_features = self.nout,
                                  hidden_features = 256,
                                  num_blocks=3,
                                  use_batch_norm=False
                                 ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        #x = F.avg_pool1d(x, x.size()[2])
        #x = x.view(x.size(0), -1)
        x = x.flatten(start_dim=1)
        out = self.mlp(x)

        return out

class EmbeddingResConv1DMLP(nn.Module):
    def __init__(self, nbasis, nout,conv_params, mlp_params, **kwargs):
        super().__init__()
        self.nout = nout
        #self.nchannel = 2*ndet
        self.nbasis = nbasis
        
        self.conv_params = conv_params        
        self.mlp_params = mlp_params
        
        self.conv = self.make_conv_layer()
        
        with torch.no_grad():
            testx = torch.randn((1, self.conv_params['in_channel'][0], self.nbasis))
            self.mlp_params['in_features'][0] = self.conv(testx).shape[-1] * self.conv_params['out_channel'][-1]
            print(f"Initialized MLP in channel: {self.mlp_params['in_features'][0]}")

        
        self.mlp = self.make_mlp_layer()

    def make_conv_layer(self):
        layers = []
        for i in range(len(self.conv_params['in_channel'])):
            layers.append(ResidualConvBlock1D(in_channels=self.conv_params['in_channel'][i], 
                                              out_channels=self.conv_params['out_channel'][i], 
                                              kernel_size=self.conv_params['kernel_size'][i], 
                                              stride=self.conv_params['stride'][i], 
                                              padding=self.conv_params['padding'][i], 
                                              dilation=self.conv_params['dilation'][i], 
                                              dropout=self.conv_params['dropout'][i]))
        
        return nn.Sequential(*layers)

    def make_mlp_layer(self):
        layers = []
        for i in range(len(self.mlp_params['in_features'])):
            layers.append(nn.Linear(self.mlp_params['in_features'][i], self.mlp_params['out_features'][i]))
            if i!= len(self.mlp_params['in_features'])-1:
                layers.append(nn.ReLU())
            
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        out = self.mlp(x)

        return out

class EmbeddingConv1DMLP(nn.Module):
    def __init__(self, nbasis, nout,conv_params, mlp_params, **kwargs):
        super().__init__()
        self.nout = nout
        #self.nchannel = 2*ndet
        self.nbasis = nbasis
        
        self.conv_params = conv_params        
        self.mlp_params = mlp_params
        
        self.conv = self.make_conv_layer()
        
        with torch.no_grad():
            testx = torch.randn((1, self.conv_params['in_channel'][0], self.nbasis))
            self.mlp_params['in_features'][0] = self.conv(testx).shape[-1] * self.conv_params['out_channel'][-1]
            print(f"Initialized MLP in channel: {self.mlp_params['in_features'][0]}")

        
        self.mlp = self.make_mlp_layer()

    def make_conv_layer(self):
        layers = []
        for i in range(len(self.conv_params['in_channel'])):
            layers.append(nn.Conv1d(in_channels=self.conv_params['in_channel'][i], 
                                    out_channels=self.conv_params['out_channel'][i], 
                                    kernel_size=self.conv_params['kernel_size'][i], 
                                    stride=self.conv_params['stride'][i], 
                                    padding=self.conv_params['padding'][i], 
                                    dilation=self.conv_params['dilation'][i]))
            if i!= len(self.conv_params['in_channel'])-1:
                layers.append(nn.ReLU())
                layers.append(nn.BatchNorm1d(num_features=self.conv_params['out_channel'][i]))
        
        return nn.Sequential(*layers)

    def make_mlp_layer(self):
        layers = []
        for i in range(len(self.mlp_params['in_features'])):
            layers.append(nn.Linear(self.mlp_params['in_features'][i], self.mlp_params['out_features'][i]))
            if i!= len(self.mlp_params['in_features'])-1:
                layers.append(nn.ReLU())
            
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        out = self.mlp(x)

        return out



class MyEmbeddingConv2D(nn.Module):
    def __init__(self, nout, **kwargs):
        super().__init__()

        self.nout = nout
        self.layers = nn.ModuleList([self.make_layer()])
        self.linear = nn.Linear(32, self.nout)

    def make_layer(self):
        layers = []
        layers.append(ResidualConvBlock2D(in_channels=1, out_channels=32,
                                          kernel_size=(3,4), stride=(1,1), dilation=(1,1), padding = (2,0)))
        layers.append(ResidualConvBlock2D(in_channels=32, out_channels=64,
                                          kernel_size=(3,8), stride=(1,1), dilation=(1,1), padding = (2,0)))
        layers.append(ResidualConvBlock2D(in_channels=64, out_channels=64,
                                          kernel_size=(3,16), stride=(1,1), dilation=(1,1), padding = (0,0)))
        layers.append(ResidualConvBlock2D(in_channels=64, out_channels=32,
                                          kernel_size=(3,8), stride=(1,1), dilation=(1,1), padding = (0,0)))
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


class MyTestEmbedding(nn.Module):
    def __init__(self, ndet, nout,  **kwargs):
        super().__init__()
        #self.ncomp = ncomp
        self.nout = nout
        self.nchannel = 2*ndet

        self.layers = nn.ModuleList([self.make_layer()])
        self.linear = nn.Linear(16, self.nout)

    def make_layer(self):
        layers = []
        layers.append(ResidualConvBlock1D(self.nchannel, 32, kernel_size=2, stride=1, padding=2, dilation=1, dropout=0.5))
        layers.append(ResidualConvBlock1D(32, 64, kernel_size=2, stride=2, padding=0, dilation=1, dropout=0.5))
        layers.append(ResidualConvBlock1D(64, 64, kernel_size=8, stride=1, padding=0, dilation=1, dropout=0.5))
        layers.append(ResidualConvBlock1D(64, 16, kernel_size=2, stride=1, padding=0, dilation=1, dropout=0.5))

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