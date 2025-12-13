import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from scipy import linalg
import config as cfg

class ZCADataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, add_noise_sigma=0.15, training=True):
        self.data = data
        self.labels = labels
        self.sigma = add_noise_sigma
        self.training = training  # Important: noise only during training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        
        # Add Gaussian noise only during training
        if self.training and self.sigma > 0:
            x = x + torch.randn_like(x) * self.sigma
            
        return x, y
    
class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0],np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T,x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1./np.sqrt(S+self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S+self.regularization)))
        self.ZCA_mat = torch.tensor(np.dot(tmp, U.T), dtype=torch.float32)
        self.inv_ZCA_mat = torch.tensor(np.dot(tmp2, U.T), dtype=torch.float32)
        self.mean = torch.tensor(m, dtype=torch.float32)

    def apply(self, x):
        """
        x: torch.Tensor of shape (batch_size, ...)
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        s = x.shape
        x_flat = x.view(s[0], -1)  # flatten all dimensions except batch
        x_whitened = torch.matmul(x_flat - self.mean, self.ZCA_mat)
        return x_whitened.view(s)

    def invert(self, x):
        """
        x: torch.Tensor of shape (batch_size, ...)
        """
        
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        s = x.shape
        x_flat = x.view(s[0], -1)
        x_original = torch.matmul(x_flat, self.inv_ZCA_mat) + self.mean
        return x_original.view(s)
    
# Learning rate scheduler: linear decay after 100 epochs
def adjust_lr(optimizer, epoch, total_epochs=200):
    if epoch >= 100:
        lr = cfg.learning_rate * (total_epochs - epoch) / 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class WideResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, activation=nn.ReLU, drop_p=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.act1 = activation
        self.drop = nn.Dropout(p=drop_p)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.act2 = activation
        self.bn = nn.BatchNorm2d(out_ch)

        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
            if in_ch != out_ch or stride != 1 else nn.Identity()
        )

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.drop(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.bn(out)
        return out + self.shortcut(x)
# -------------------------
# 9-layer CNN
# -------------------------
class CIFAR100CNN(nn.Module):
    def __init__(self, activation=nn.ReLU, num_classes=100):
        super().__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)

        self.block1 = self._make_layer(16, 64, 6, stride=1, activation=activation)
        self.block2 = self._make_layer(64, 128, 6, stride=2, activation=activation)
        self.block3 = self._make_layer(128, 256, 6, stride=2, activation=activation)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_ch, out_ch, n, stride, activation):
        layers = [WideResBlock(in_ch, out_ch, stride, activation)]
        for _ in range(n - 1):
            layers.append(WideResBlock(out_ch, out_ch, 1, activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

def prepare_datasets(mode, val_ratio=0.1, data_root='./data'):
    """Prepare training, validation and test datasets"""
    # Load raw training data
    train_dataset_raw = datasets.CIFAR100(root=data_root, train=True, download=True, 
                                         transform=transforms.ToTensor())
    
    X_train = np.array([np.array(img) for img, _ in train_dataset_raw], dtype=np.float32)
    labels = np.array([label for _, label in train_dataset_raw])
    # Fit ZCA
    #whitener = ZCA(x=X_train)
    #trainx_white = whitener.apply(X_train)
    #print(trainx_white.shape)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Create full dataset
    full_train_dataset = ZCADataset(X_train, labels_tensor, add_noise_sigma=0.0, training=True)
    
    # Test dataset
    test_dataset_raw = datasets.CIFAR100(root=data_root, train=False, download=True,
                                        transform=transforms.ToTensor())
    X_test = np.array([np.array(img) for img, _ in test_dataset_raw], dtype=np.float32)
    labels_test = np.array([label for _, label in test_dataset_raw])
    #testx_white = whitener.apply(X_test)
    test_dataset = ZCADataset(X_test, torch.tensor(labels_test, dtype=torch.long),
                            add_noise_sigma=0.0, training=False)
    
    dummy1 = None
    dummy2 = None
    dummy3 = None
    dummy4 = None
    if mode == 1:
        print('CIFAR100 dataset restoration completed')
        return full_train_dataset, dummy1, test_dataset, dummy3, dummy4
    else:
        if val_ratio != 0.0:
            # Split train/validation
            total_size = len(full_train_dataset)
            val_size = int(total_size * val_ratio)
            train_size = total_size - val_size
            train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
            
            # Validation dataset without noise
            train_indices = train_dataset.indices
            val_indices = val_dataset.indices
            X_val_data = X_train[val_indices]
            val_labels = labels_tensor[val_indices]
            val_dataset = ZCADataset(X_val_data, val_labels, add_noise_sigma=0.0, training=False)

            #print(train_dataset.shape)
            print('CIFAR100 dataset preparation completed')
            return train_dataset, val_dataset, test_dataset, train_indices, val_indices
        else:
            train_dataset = full_train_dataset
            print('CIFAR100 dataset preparation completed')
            return train_dataset, None, test_dataset, None, None