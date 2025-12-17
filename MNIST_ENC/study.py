import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MNIST_ENC.neuralnet import DeepAutoencoder
from file_operations import create_directory
from csv_operations import csv_write2
import numpy as np
from models.model_lelu import LELU
from models.model_nele import NELE, NeleLutParam
import config as cfg
from torch.utils.data import random_split

def save(model, optimizer, epoch_loss, activation_type, epoch, dir, test_loss = 0.0):
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch_loss': epoch_loss,
    'test_loss': test_loss
    }, dir + activation_type + '_' + str(epoch) + '.pth')


def mnist_enc_data(epochs, learn_rate, device, exec, activation_type='default'):
    activations =  ['Tanh', 'ReLU', 'ELU', 'GELU', 'Sigmoid', 'Leaky ReLU', 'SiLU', 'Softplus', 'LELU', 'BELU', 'Mish', 'NELE']
    colors = ["#1f77b4", "#aec7e8", 
            "#ff7f0e", "#ffbb78",
            "#2ca02c", "#98df8a",
            "#9467bd", "#c5b0d5",
            "#8c564b", "#c49c94",
            '#000000']
    loss_collect = np.zeros(len(activations))
    std_deviation_collect = np.zeros(len(activations))
    dir = 'RESULTS/MNIST_ENC/' + activation_type + '/'
    create_directory('RESULTS/MNIST_ENC/' + activation_type + '/')
    create_directory('PICS/MNIST_ENC/' + activation_type + '/')
    # Activation function selection
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

    train_dataset_full = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # Compute split sizes
    n_total = len(train_dataset_full)
    n_val = int(n_total * cfg.val_ratio)
    n_train = n_total - n_val

    # Split
    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [n_train, n_val]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size_autoenc, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size_autoenc, shuffle=False)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size_autoenc, shuffle=False)

    # Model, loss, optimizer
    device = device
    model = DeepAutoencoder(activation).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    criterion = criterion.to(device)

    loss_collect = []
    val_collect = []
    test_collect = []
    # Training loop
    correct_val, total_val = 0, 0
    correct_test, total_test = 0, 0
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
        epoch_loss /= len(train_loader.dataset)
        loss_collect.append(epoch_loss)

        # Validation
        val_loss = 0.0
        if cfg.val_ratio != 0:
            model.eval()
            with torch.no_grad():
                for x, _ in val_loader:  # labels not needed
                    x = x.to(device)
                    outputs = model(x)
                    loss = criterion(outputs, x)  # compare to input
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(val_loader.dataset)
        val_collect.append(val_loss)

        # Test
        test_loss_clean = 0.0
        test_loss_noisy = 0.0
        model.eval()
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                outputs = model(x)
                test_loss_clean += criterion(outputs, x).item() * x.size(0)
        test_loss_clean /= len(test_loader.dataset)
        
        # Test with noise
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                noise = torch.empty_like(x).uniform_(-cfg.noise_level, cfg.noise_level)
                x_noisy = x + noise
                outputs = model(x_noisy)
                test_loss_noisy += criterion(outputs, x).item() * x.size(0)
        test_loss_noisy /= len(test_loader.dataset)
        test_collect.append(test_loss_clean)
        if (epoch + 1) % cfg.save_every  == 0:
            save(model, optimizer, epoch_loss, activation_type, epoch, dir, test_loss_clean)
        print(f"Epoch {epoch+1}, loss: {epoch_loss:.4f}, Val Acc: {val_loss:.4f}, Test Acc 0: {test_loss_clean:.4f}, Test Acc 3: {test_loss_noisy:.4f}, lr : {lr:.5f}")
    
    # Evaluate
    
    loss_collect = torch.tensor(loss_collect)
    test_collect = torch.tensor(test_collect)
    val_collect = torch.tensor(val_collect)
    csv_write2(dir + '/loss_history_' + activation_type + '.csv', 
              torch.linspace(1, cfg.epochs+1, cfg.epochs+1), 
              loss_collect, 'epoch', 'loss', 'val', 'test', val_collect, test_collect, exec)
    