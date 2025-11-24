import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MNIST.Neuralnet import DeepFCNet
from file_operations import create_directory
from csv_operations import csv_write2
import numpy as np
from models.model_lelu import LELU
from models.model_nele import NELE
import config as cfg
from torch.utils.data import random_split

def save(model, optimizer, epoch_loss, activation_type, epoch, dir):
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch_loss': epoch_loss
    }, dir + activation_type + '_' + str(epoch) + '.pth')


def mnist_data(epochs, learn_rate, device, exec, activation_type='default'):
    activations =  ['Tanh', 'ReLU', 'ELU', 'GELU', 'Sigmoid', 'Leaky ReLU', 'SiLU', 'Softplus', 'LELU', 'BELU', 'Mish', 'NELE']
    colors = ["#1f77b4", "#aec7e8", 
            "#ff7f0e", "#ffbb78",
            "#2ca02c", "#98df8a",
            "#9467bd", "#c5b0d5",
            "#8c564b", "#c49c94",
            '#000000']
    loss_collect = np.zeros(len(activations))
    std_deviation_collect = np.zeros(len(activations))
    dir = 'RESULTS/MNIST/' + activation_type + '/'
    create_directory('RESULTS/MNIST/' + activation_type + '/')
    create_directory('PICS/MNIST/' + activation_type + '/')
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
        activation = NELE(128, 3, 2)
    else:
        raise ValueError("Invalid activation type")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ])

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
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Model, loss, optimizer
    device = device
    model = DeepFCNet(cfg.input_size, cfg.hidden_size, cfg.num_hidden_layers, cfg.num_classes, activation, activation_type).to(device)
    criterion = nn.CrossEntropyLoss()
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
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
        epoch_loss /= len(train_loader.dataset)
        loss_collect.append(epoch_loss)

        # Optional: validation
        val_acc = 0.0
        if cfg.val_ratio != 0:
            model.eval()
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
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += target.size(0)
                    correct_test += (predicted == target).sum().item()
        test_collect.append(100 * correct_test / max(total_test, 1))
            # print(f'Test Accuracy: {100 * correct / total:.2f}%')
        if (epoch + 1) % cfg.save_every  == 0:
            save(model, optimizer, epoch_loss, activation_type, epoch, dir)
        print(f"Epoch {epoch+1}, loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {100 * correct_test / max(total_test, 1):.4f}, lr : {lr:.5f}")
    
    # Evaluate
    
    loss_collect = torch.tensor(loss_collect)
    test_collect = torch.tensor(test_collect)
    val_collect = torch.tensor(val_collect)
    csv_write2(dir + '/loss_history_' + activation_type + '.csv', 
              torch.linspace(1, cfg.epochs+1, cfg.epochs+1), 
              loss_collect, 'epoch', 'loss', 'val', 'test', val_collect, test_collect, exec)
    