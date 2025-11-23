import torch
import config as cfg
import numpy as np
import torch.nn as nn
from file_operations import create_directory
from models.model_lelu import LELU
from models.model_nele import NELE
from torch.optim import Adam
from torch.utils.data import DataLoader
from CIFAR10.NeuralNet import CIFAR10CNN, adjust_lr, prepare_datasets
from csv_operations import csv_write2
import time
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

    # Step 2: Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(
        val_ratio=0.1
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # -------------------------
    # Training setup
    # -------------------------
    model = CIFAR10CNN(activation=activation).to(device)
    optimizer = Adam(model.parameters(), lr=learn_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device=device)
    # -------------------------
    # Training loop skeleton
    # -------------------------
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        adjust_lr(optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        
        for i, (x, y) in enumerate(train_loader):
            start_epoch = time.time()
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * x.size(0)
            end_epoch = time.time()
            print(f"Batch : {i}, Time : {end_epoch - start_epoch:.2f} seconds")
            
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
        print(f"Epoch {epoch+1}, loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {100 * correct_test / total_test:.4f}, lr : {lr:.5f}")

    loss_collect = torch.tensor(loss_collect)
    test_collect = torch.tensor(test_collect)
    val_collect = torch.tensor(val_collect)
    csv_write2(dir + '/loss_history_' + activation_type + '.csv', 
              torch.linspace(1, cfg.epochs, cfg.epochs), 
              loss_collect, 'epoch', 'loss', '', 'test', val_collect, test_collect)
    