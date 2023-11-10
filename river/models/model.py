import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 6, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(51200, 3)
        self._init_params()

    def _init_params(self):
        for module in [self.layer1[0], self.layer2[0], self.layer3[0], self.layer4[0], self.fc]:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)

    def forward(self, x):
        bs, _, _, _ = x.shape # [batch_size, channel, height, width]
        x = self.layer1(x) # [1, 64, 97, 97]
        x = self.layer2(x) # [1, 128, 47, 47]
        x = self.layer3(x) # [bs, 256, 22, 22]
        x = self.layer4(x) # [bs, 512, 10, 10]
        x = self.dropout(x).reshape((bs, -1)) # [bs, 51200]
        x = self.fc(x) 
        output = F.log_softmax(x, dim=1) # [bs, 3]
        return output

if __name__ == '__main__':
    x = torch.randn(128, 3, 200, 200)
    print(Net(x))