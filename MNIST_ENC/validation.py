# pylint:disable=too-many-locals
"""
Docstring for mnist_enc.validation
"""
import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import batch_size_autoenc
from mnist_enc.neuralnet import DeepAutoencoder
from mnist.utils import get_activation

def mnist_enc_validation(epochs, device, noise_level, activation_type='default'):
    """
    Validation for MNIST autoencoder
    
    :param epochs: total epochs
    :param device: device name
    :param noise_level: noise level
    :param activation_type: AF
    """
    activation = get_activation(activation_type, device)
    # Load MNIST
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, \
                                  transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_autoenc, \
                             shuffle=False)
    model = DeepAutoencoder(activation).to(device)
    test_collect = []
    criterion = nn.MSELoss()
    for _ in range(1, 2):
        torch.manual_seed(1234)
        checkpoint = torch.load('RESULTS/MNIST_ENC/' + \
                            activation_type + '/' + activation_type \
                            + '_' + str(epochs - 1) + '.pth',
                            map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # test_loss = checkpoint['test_loss']
        model.eval()
        test_loss_noisy = 0.0

        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                noise = torch.empty_like(data).uniform_(-noise_level, noise_level).to(device)
                x_noisy = data + noise
                outputs = model(x_noisy).to(device)
                test_loss_noisy += criterion(outputs, data).item() * data.size(0)
        test_loss_noisy /= len(test_loader.dataset)
        test_collect.append(test_loss_noisy)
    #test_collect = np.asarray(test_collect)
    test_acc_med = np.median(test_collect)
    print(f'{activation_type}, Test Error: {test_acc_med:.4f}')
    return test_acc_med
