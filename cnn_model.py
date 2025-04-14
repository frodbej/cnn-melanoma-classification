
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Class with the CNN structure."""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64x64 -> 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32x32 -> 16x16
        x = x.view(-1, 16 * 16 * 16) # Flatten
        x = F.relu(self.fc1(x)) # 4096 -> 64
        x = self.fc2(x) # 64 -> 2
        return x


def get_model(num_classes=2):
    model = SimpleCNN(num_classes)
    return model