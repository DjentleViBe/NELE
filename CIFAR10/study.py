import torch
import config as cfg
import numpy as np
import torch.nn as nn
from file_operations import create_directory
from models.model_lelu import LELU
from models.model_nele import NELE
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from CIFAR10.NeuralNet import CIFAR10CNN, adjust_lr, GaussianNoise
from csv_operations import csv_write2

def save(model, optimizer, epoch_loss, activation_type, epoch, dir):
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch_loss': epoch_loss
    }, dir + activation_type + '_' + str(epoch) + '.pth')

def cifar10_data(epochs, learn_rate, device, activation_type='default'):
    activations = cfg.AF
    colors = cfg.colors
    loss_collect = []
    val_collect = []
    test_collect = []
    dir = 'RESULTS/CIFAR10/' + activation_type + '/'
    create_directory('RESULTS/CIFAR10/' + activation_type + '/')
    create_directory('PICS/CIFAR10/' + activation_type + '/')

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
        activation = nn.Softmax()
    elif base == 'tanh':
        activation = nn.Tanh()
    elif base == 'lelu' :
        activation = LELU()
    elif base == 'nele' :
        activation = NELE(1, 3, 2)
    else:
        raise ValueError("Invalid activation type")

    # CIFAR-10 transforms (ZCA whitening placeholder)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        GaussianNoise(0.15)  # Gaussian noise on input
    ])
    transform_test = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Example: split validation
    train_dataset, val_dataset = random_split(train_dataset, [45000, 5000])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # -------------------------
    # Training setup
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CIFAR10CNN(activation=activation).to(device)
    optimizer = Adam(model.parameters(), lr=learn_rate)
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Training loop skeleton
    # -------------------------
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        adjust_lr(optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        loss_collect.append(epoch_loss)
        
        # Optional: validation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = outputs.max(1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()
        val_acc = correct_val / total_val
        val_collect.append(val_acc)
        if epoch % 5 == 0:
            model.eval()
            correct_test, total_test = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += target.size(0)
                    correct_test += (predicted == target).sum().item()
        test_collect.append(100 * correct_test / total_test)
        if (epoch + 1) % cfg.save_every  == 0:
            save(model, optimizer, epoch_loss, activation_type, epoch, dir)    
        print(f"Epoch {epoch+1}, Val Acc: {val_acc:.4f}, Test Acc: {100 * correct_test / total_test:.4f}, lr : {lr}")

    loss_collect = torch.tensor(loss_collect)
    test_collect = torch.tensor(test_collect)
    val_collect = torch.tensor(val_collect)
    csv_write2(dir + '/loss_history_' + activation_type + '.csv', 
              torch.linspace(1, cfg.epochs, cfg.epochs), 
              loss_collect, 'epoch', 'loss', '', 'test', val_collect, test_collect)
    