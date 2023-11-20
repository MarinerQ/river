import torch
from torch import nn
import torch.nn.functional as F

class ResidualMLPBlock1D(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(in_features if i == 0 else out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        ) for i in range(num_blocks)])
        self.reshape = nn.Linear(in_features, out_features)  # Linear reshape for residual

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            residual = x
            if i == 0:
                out = block(x)
            else:
                out = block(out)  # Use the output from the previous block as input
            residual = self.reshape(residual)  # Apply reshape to residual
            out += residual
            out = F.relu(out)
        return out

class EmbeddingMLP1D(nn.Module):
    '''
    For a (batch_size, nchannel, length) input, perform ResidualMLP for each (:, channel, :). 
    '''
    def __init__(self, ndet, nout, num_blocks, datalength, use_psd = True, middle_features = 128, **kwargs):
        super().__init__()
        self.datalength = datalength
        self.nout = nout
        self.ndet = ndet
        self.use_psd = use_psd
        
        if use_psd:
            self.nstream = 3
            self.nchannel = 3*ndet # strains(2) + PSD (1)
        else:
            self.nstream = 2
            self.nchannel = 2*ndet

        self.middle_features = middle_features
        self.layers = nn.ModuleList([self.make_layer(ResidualMLPBlock1D, self.middle_features, num_blocks, iblock) for iblock in range(self.nstream)])
        self.linear = nn.Linear(self.middle_features * self.nchannel, self.nout)

    def make_layer(self, block, out_features, num_blocks, iblock):
        layers = []
        in_feature = self.datalength
        layers.append(block(in_features = in_feature, out_features = out_features, num_blocks = num_blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x : [batch_size, channel (det_123, amp/phase) = 2*ndet, length (number of samples)]
        # bs,_,_  = x.shape 
        outputs = []
        for i, layer in enumerate(self.layers):
            for j in range(self.nstream):
                outputs.append(layer(x[:, i*self.nstream + j, :]))
        x = torch.cat(outputs, dim=1)
        x = x.view(x.size(0), -1)
        output = self.linear(x)
        return output


