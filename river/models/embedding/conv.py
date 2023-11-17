import torch
from torch import nn

class EmbeddingConv1D(nn.Module):
    def __init__(self, ndet, ncomp, nout, use_psd = True, middle_channel = 512):
        super().__init__()
        self.ncomp = ncomp
        self.nout = nout
        if use_psd:
            self.nchannel = 3*ndet # strains(2) + PSD (1)
        else:
            self.nchannel = 2*ndet

        self.middle_channel = middle_channel
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=self.nchannel, out_channels=self.middle_channel, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.middle_channel),
            #nn.MaxPool1d(kernel_size=2)
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=self.middle_channel, out_channels=self.middle_channel, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.middle_channel),
            #nn.MaxPool1d(kernel_size=2)
        )

        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=self.middle_channel, out_channels=self.middle_channel, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.middle_channel),
            #nn.MaxPool1d(kernel_size=2)
        )
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(self.middle_channel*self.ncomp, self.nout)


    def forward(self, x):
        # x : [batch_size, channel (det_123, amp/phase) = 2*ndet, length (number of samples)]
        bs,_,_  = x.shape 
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.dropout(x).reshape((bs,-1))
        output = self.linear(x)

        return output

class EmbeddingConv2D(nn.Module):
    def __init__(self, ndet, ncomp, nout, use_psd = True, middle_channel = 16):
        super().__init__()
        self.ncomp = ncomp
        self.nout = nout
        if use_psd:
            self.n1dchannel = 3*ndet # strains(2) + PSD (1)
        else:
            self.n1dchannel = 2*ndet

        self.middle_channel = middle_channel
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.middle_channel, kernel_size=1, stride=1, dilation=1),
            nn.BatchNorm2d(self.middle_channel),
            #nn.MaxPool1d(kernel_size=2)
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=self.middle_channel, out_channels=self.middle_channel, kernel_size=1, stride=1, dilation=1),
            nn.BatchNorm2d(self.middle_channel),
            #nn.MaxPool1d(kernel_size=2)
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=self.middle_channel, out_channels=self.middle_channel, kernel_size=1, stride=1, dilation=1),
            nn.BatchNorm2d(self.middle_channel),
            #nn.MaxPool1d(kernel_size=2)
        )
        self.dropout = nn.Dropout(0.5)
        #self.linear = nn.Linear(self.middle_channel*self.ncomp*self.n1dchannel, self.nout)
        self.linear = nn.Linear(771096, self.nout)


    def forward(self, x):
        # x : [batch_size, channel (det_123, amp/phase) = 2*ndet, length (number of samples)]
        bs,_,_,_  = x.shape 
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.dropout(x).reshape((bs,-1))
        #output = nn.Linear(len(x), self.nout, device=self.device)(x)
        output = self.linear(x)

        return output