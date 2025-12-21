# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
"""
Docstring for CIFAR10.NeuralNet
"""
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import random_split
import numpy as np
from torchvision import datasets, transforms
from scipy import linalg
import config as cfg

class ZCADataset(torch.utils.data.Dataset):
    """
    Docstring for ZCADataset
    """
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

class ZCA:
    """
    Docstring for ZCA
    """
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        """
        Docstring for fit
        """
        s = x.shape
        x = x.copy().reshape((s[0],np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = np.dot(x.T,x) / x.shape[0]
        uval, sval, _ = linalg.svd(sigma)
        tmp = np.dot(uval, np.diag(1./np.sqrt(sval+self.regularization)))
        tmp2 = np.dot(uval, np.diag(np.sqrt(sval+self.regularization)))
        self.zca_mat = torch.tensor(np.dot(tmp, uval.T), dtype=torch.float32)
        self.inv_zca_mat = torch.tensor(np.dot(tmp2, uval.T), dtype=torch.float32)
        self.mean = torch.tensor(m, dtype=torch.float32)

    def apply(self, x):
        """
        x: torch.Tensor of shape (batch_size, ...)
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        s = x.shape
        x_flat = x.view(s[0], -1)  # flatten all dimensions except batch
        x_whitened = torch.matmul(x_flat - self.mean, self.zca_mat)
        return x_whitened.view(s)

    def invert(self, x):
        """
        x: torch.Tensor of shape (batch_size, ...)
        """

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        s = x.shape
        x_flat = x.view(s[0], -1)
        x_original = torch.matmul(x_flat, self.inv_zca_mat) + self.mean
        return x_original.view(s)

# Learning rate scheduler: linear decay after 100 epochs
def adjust_lr(optimizer, epoch, total_epochs=200):
    """Adjust learning rate
    """
    if epoch >= 100:
        lr = cfg.learning_rate * (total_epochs - epoch) / 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
# -------------------------
# 9-layer CNN
# -------------------------
class CIFAR10CNN(nn.Module):
    """
    CIFAR10 CNN
    """
    def __init__(self, activation=F.gelu):
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

        self.bn_final = nn.BatchNorm1d(192)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        """
        Docstring for forward
        """
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
        x = self.bn_final(x)  # This makes it identical to the second architecture
        x = self.fc(x)
        return x

def prepare_datasets(mode, val_ratio=0.1, data_root='./data'):
    """Prepare training, validation and test datasets"""
    # Load raw training data
    train_dataset_raw = datasets.CIFAR10(root=data_root, train=True, download=True,
                                         transform=transforms.ToTensor())

    x_train = np.array([np.array(img) for img, _ in train_dataset_raw], dtype=np.float32)
    labels = np.array([label for _, label in train_dataset_raw])
    # Fit ZCA
    whitener = ZCA(x=x_train)
    trainx_white = whitener.apply(x_train)
    #print(trainx_white.shape)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Create full dataset
    full_train_dataset = ZCADataset(trainx_white, labels_tensor, \
                                    add_noise_sigma=0.15, training=True)

    # Test dataset
    test_dataset_raw = datasets.CIFAR10(root=data_root, train=False, \
                                        download=True,
                                        transform=transforms.ToTensor())
    x_test = np.array([np.array(img) for img, _ in test_dataset_raw], \
                      dtype=np.float32)
    labels_test = np.array([label for _, label in test_dataset_raw])
    testx_white = whitener.apply(x_test)
    test_dataset = ZCADataset(testx_white, torch.tensor(labels_test, \
                                dtype=torch.long),
                            add_noise_sigma=0.0, training=False)

    dummy1 = None
    dummy3 = None
    dummy4 = None
    if mode == 1:
        print('CIFAR10 dataset restoration completed')
        return full_train_dataset, dummy1, test_dataset, dummy3, dummy4
    if val_ratio != 0.0:
        # Split train/validation
        total_size = len(full_train_dataset)
        val_size = int(total_size * val_ratio)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

        # Validation dataset without noise
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        x_val_data = trainx_white[val_indices]
        val_labels = labels_tensor[val_indices]
        val_dataset = ZCADataset(x_val_data, val_labels, add_noise_sigma=0.0, training=False)

        #print(train_dataset.shape)
        print('CIFAR10 dataset preparation completed')
        return train_dataset, val_dataset, test_dataset, train_indices, val_indices
    train_dataset = full_train_dataset
    print('CIFAR10 dataset preparation completed')
    return train_dataset, None, test_dataset, None, None
