import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block - maintaining spatial dimensions
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1),  # 28x28x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 16, 3, padding=1),  # 28x28x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05)
        )
        
        # First Block - first reduction
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 20, 3, padding=1),  # 28x28x20
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2, 2),  # 14x14x20
            nn.Dropout(0.05)
        )
        
        # Transition Block
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 24, 3, padding=1),  # 14x14x24
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Dropout(0.1)
        )
        
        # Second Block - second reduction
        self.conv4 = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1),  # 14x14x24
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.MaxPool2d(2, 2),  # 7x7x24
            nn.Dropout(0.1)
        )
        
        # Output Block
        self.conv5 = nn.Sequential(
            nn.Conv2d(24, 20, 3, padding=1),  # 7x7x20
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Conv2d(20, 10, 1),  # 7x7x10
            nn.AdaptiveAvgPool2d(1)  # 1x1x10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1) 