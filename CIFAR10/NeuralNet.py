import torch.nn as nn
import torch.nn.functional as F
import torch

class ZCATestTransform:
    def __init__(self, mean, W_zca):
        self.mean = mean
        self.W_zca = W_zca

    def __call__(self, img):
        x = np.array(img, dtype=np.float32).flatten() / 255.0
        x_centered = x - self.mean
        x_zca = x_centered @ self.W_zca
        return torch.tensor(x_zca.reshape(3,32,32), dtype=torch.float32)
    
class ZCADataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, add_noise_sigma=0.15):
        self.data = data
        self.labels = labels
        self.sigma = add_noise_sigma

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.sigma > 0:
            x = x + torch.randn_like(x) * self.sigma
        return x, y
    
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.15):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.sigma
            return x + noise
        return x
    
# Learning rate scheduler: linear decay after 100 epochs
def adjust_lr(optimizer, epoch, total_epochs=200):
    if epoch >= 100:
        lr = 1e-3 * (total_epochs - epoch) / 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
# -------------------------
# 9-layer CNN
# -------------------------
class CIFAR10CNN(nn.Module):
    def __init__(self, activation=F.relu):
        super().__init__()
        self.activation = activation
        # Block 1
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        
        # Block 2
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(192)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.5)
        
        # Block 3
        self.conv7 = nn.Conv2d(192, 192, 3)  # 3x3 conv without padding, 8x8 -> 6x6
        self.bn7 = nn.BatchNorm2d(192)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.bn8 = nn.BatchNorm2d(192)
        self.conv9 = nn.Conv2d(192, 192, 1)
        self.bn9 = nn.BatchNorm2d(192)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, 10)
        
    def forward(self, x):
        # Block 1
        act = self.activation  # shortcut

        # Block 1
        x = act(self.bn1(self.conv1(x)))
        x = act(self.bn2(self.conv2(x)))
        x = act(self.bn3(self.conv3(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = act(self.bn4(self.conv4(x)))
        x = act(self.bn5(self.conv5(x)))
        x = act(self.bn6(self.conv6(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = act(self.bn7(self.conv7(x)))
        x = act(self.bn8(self.conv8(x)))
        x = act(self.bn9(self.conv9(x)))

        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x