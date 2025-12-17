import torch.nn as nn
import torch
from config import input_size, hidden_size, num_hidden_layers, num_classes, batch_size_autoenc
from MNIST_ENC.Neuralnet import DeepAutoencoder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model_lelu import LELU
from models.model_nele import NELE, NeleLutParam
import config as cfg
import numpy as np

def mnist_enc_validation(epochs, device, noise_level, activation_type='default'):
    if '=' in activation_type:
        base, param = activation_type.split('=')
        param = float(param)  # convert parameter to float if needed
    else:
        base = activation_type
        param = None
    if base == 'relu':
        activation = nn.ReLU()
    elif base == 'elu':
        activation = nn.ELU(alpha=1.0)
    elif base == 'leaky_relu' :
        activation = nn.LeakyReLU(negative_slope=0.1)
    elif base == 'gelu' :
        activation = nn.GELU()
    elif base == 'mish' :
        activation = nn.Mish()
    elif base == 'sigmoid' :
        activation = nn.Sigmoid()
    elif base == 'silu' :
        activation = nn.SiLU()
    elif base == 'softplus' :
        activation = nn.Softplus()
    elif base == 'tanh':
        activation = nn.Tanh()
    elif base == 'lelu' :
        activation = LELU()
    elif base == 'nele' :
        activation = NeleLutParam(device)
    else:
        raise ValueError("Invalid activation type")
    # Load MNIST
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_autoenc, shuffle=False)
    model = DeepAutoencoder(activation).to(device)
    test_collect = []
    criterion = nn.MSELoss()
    for i in range(1, 2):
        torch.manual_seed(1234)
        checkpoint = torch.load('RESULTS/MNIST_ENC/' + activation_type + '/' + activation_type + '_' + str(epochs - 1) + '.pth',
                            map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        epoch_loss = checkpoint['epoch_loss']
        # test_loss = checkpoint['test_loss']
        model.eval()
        test_loss_noisy = 0.0
        total = 0.0

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
