import torch
import config as cfg
import numpy as np
import torch.nn as nn
from file_operations import create_directory, getlatest
from models.model_lelu import LELU
from models.model_nele import NELE_LUT_PARAM_DIR, NELE_LUT_LEARN
from torch.optim import Adam
from torch.utils.data import DataLoader
from CIFAR10.neuralnet import CIFAR10CNN, adjust_lr, prepare_datasets
from csv_operations import csv_write2
import time
import pickle
import sys
import torch.nn.functional as F

def save(model, optimizer, epoch_loss, activation_type, epoch, dir):
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch_loss': epoch_loss
    }, dir + activation_type + '_' + str(epoch) + '.pth')

def cifar10_data(epochs, learn_rate, device, exec, activation_type='default'):
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
        activation = F.gelu
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
        activation = NELE_LUT_PARAM_DIR(device)
    else:
        raise ValueError("Invalid activation type")

    # -------------------------
    # Training setup
    # -------------------------
    model = CIFAR10CNN(activation=activation).to(device)
    optimizer = Adam(model.parameters(), lr=learn_rate)
    criterion = nn.CrossEntropyLoss()

    if exec == 1:
        # load the latest .pth file
        checkpoint_path = getlatest('RESULTS/CIFAR10/' + activation_type + '/')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] - 1
        with open('RESULTS/CIFAR10/' + activation_type + '/split_indices.pkl', 'rb') as f:
            train_indices, val_indices = pickle.load(f)
        full_train_set, _,test_dataset, _, _ = prepare_datasets(
            val_ratio=cfg.val_ratio,
            mode = 1
        )
        train_dataset = torch.utils.data.Subset(full_train_set, train_indices)
        if cfg.val_ratio != 0.0:
            val_dataset   = torch.utils.data.Subset(full_train_set, val_indices)

    # Step 2: Prepare datasets
    else:
        start_epoch = 0
        train_dataset, val_dataset, test_dataset, train_indices, val_indices = prepare_datasets(
            val_ratio=cfg.val_ratio,
            mode = 0
        )
        with open('RESULTS/CIFAR10/' + activation_type + '/split_indices.pkl', 'wb') as f:
            pickle.dump((train_indices, val_indices), f)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    if cfg.val_ratio != 0.0:
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
     
    # -------------------------
    # Training loop skeleton
    # -------------------------
    correct_val, total_val = 0, 0
    correct_test, total_test = 0, 0
    print(f'Activation : {activation_type}')
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0
        model.train()
        adjust_lr(optimizer, epoch)
        total_batches = len(train_loader)
        total_time = 0
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        
        for i, (x, y) in enumerate(train_loader):
            start_epoch = time.time()
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            end_epoch = time.time()
            batch_time = end_epoch - start_epoch
            total_time += batch_time
            # Progress bar
            bar_len = 30
            filled_len = int(round(bar_len * (i + 1) / total_batches))
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            
            # Print progress bar in-place
            sys.stdout.write(f'\rEpoch {epoch+1}/{epochs} |[{bar}]| '
                            f'Batch {i+1}/{total_batches} | Loss: {loss.item():.4f} | Time: {batch_time:.2f}s')
            sys.stdout.flush()
                
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
            val_acc = correct_val / max(total_val, 1.0)
        val_collect.append(val_acc)
        if epoch % 5 == 0:
            model.eval()
            correct_test, total_test = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs, 1)
                    total_test += target.size(0)
                    correct_test += (predicted == target).sum().item()
        test_collect.append(100 * correct_test / total_test)
        if (epoch + 1) % cfg.save_every  == 0:
            save(model, optimizer, epoch_loss, activation_type, epoch + 1, dir)    
        print(f"\nEpoch {epoch+1}, loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {100 * correct_test / max(total_test, 1):.4f}, lr : {lr:.5f}, Time : {total_time:.4f}")
    
    loss_collect = torch.tensor(loss_collect)
    test_collect = torch.tensor(test_collect)
    val_collect = torch.tensor(val_collect)
    csv_write2(dir + '/loss_history_' + activation_type + '.csv', 
              torch.linspace(1, cfg.epochs+1, cfg.epochs+1), 
              loss_collect, 'epoch', 'loss', 'val', 'test', val_collect, test_collect, exec)
    