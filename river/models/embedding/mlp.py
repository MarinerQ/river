import torch
from torch import nn
import torch.nn.functional as F

class ResidualMLPBlock1D(nn.Module):
    def __init__(self, in_features, out_features, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features if i == 0 else out_features, out_features),
                nn.BatchNorm1d(out_features),
                nn.ReLU(),
                nn.Linear(out_features, out_features),
                nn.BatchNorm1d(out_features)
            ) for i in range(num_blocks)
        ])
        if in_features != out_features:
            self.reshape = nn.Linear(in_features, out_features)  # Conditional reshape for residual
        else:
            self.reshape = None

    def forward(self, x):
        out = x
        for block in self.blocks:
            residual = x if self.reshape is None else self.reshape(x)
            out = block(out)
            out += residual
            out = F.relu(out)
        return out

class MLPResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, depth=2):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(nn.Linear(in_features if i == 0 else out_features, out_features))
            self.layers.append(nn.BatchNorm1d(out_features))
            self.layers.append(nn.ReLU())
        if in_features != out_features:
            self.reshape = nn.Linear(in_features, out_features)
        else:
            self.reshape = None

    def forward(self, x):
        residual = x if self.reshape is None else self.reshape(x)
        out = x
        for layer in self.layers:
            out = layer(out)
        out = out+ residual
        return F.relu(out)


class MLPResNet(nn.Module):
    '''
    For (batch_size, C, L) input 
    '''
    def __init__(self, C, L, out_features, middle_features, depth_per_block=2, **kwargs):
        super().__init__()
        self.C = C
        self.L = L
        self.initial_in_features = C * L
        self.out_features = out_features
        self.middle_features = middle_features
        self.num_blocks = len(middle_features)
        self.depth_per_block = depth_per_block
        
        self.layers = nn.ModuleList([self.make_layer(MLPResNetBlock, iblock) for iblock in range(self.num_blocks)])
        self.linear = nn.Linear(self.middle_features[-1], self.out_features)

    def make_layer(self, block, iblock):
        layers = []
        if iblock == 0:
            block_in_features = self.initial_in_features
        else:
            block_in_features = self.middle_features[iblock - 1]
        block_out_features = self.middle_features[iblock]
        layers.append(block(in_features=block_in_features, out_features=block_out_features, depth=self.depth_per_block))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x : (batch_size, C, L)
        x = x.view(x.size(0), -1)  # Reshape x to (batch, C*L)
        for layer in self.layers:
            x = layer(x)
        output = self.linear(x)
        return output


'''

class ResnetMLP1D(nn.Module):
    def __init__(self, nout, num_blocks, in_feature, middle_features = 128, **kwargs):
        super().__init__()
        self.in_feature = in_feature
        self.nout = nout
        self.middle_features = middle_features
        self.num_blocks = num_blocks
        
        self.layers = nn.ModuleList([self.make_layer(ResidualMLPBlock1D, self.middle_features, num_blocks, iblock) for iblock in range(self.num_blocks)])
        self.linear = nn.Linear(self.middle_features, self.nout)

    def make_layer(self, block, out_features, num_blocks, iblock):
        layers = []
        if iblock == 0:
            in_feature = self.in_feature
        else:
            in_feature = self.middle_features
        layers.append(block(in_features = in_feature, out_features = out_features, num_blocks = num_blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        output = self.linear(x)
        return output



class EmbeddingMLP1D(nn.Module):
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
'''