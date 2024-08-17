import torch.nn as nn
import torch.nn.functional as F

class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x
