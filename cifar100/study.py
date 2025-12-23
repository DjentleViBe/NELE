# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
"""
Docstring for CIFAR100.study
"""
import time
import pickle
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import optuna
from file_operations import create_directory, getlatest
from cifar100.neuralnet import CIFAR100CNN, prepare_datasets
from csv_operations import csv_write2
from cifar10.utils import save
from cifar10.utils import get_activation
import config as cfg

def cifar100_data(directory, device, exec_study, exec_type, config = None, activation_type='default', trial = None):
    """
    Docstring for cifar10_data
    
    :param epochs: total epochs
    :param learn_rate: learning rate
    :param device: device name
    :param exec_study: standalone or 7 runs
    :param exec_type: hyper param
    :param activation_type: AF
    """
    best_val = float('inf')
    activations =  ['Tanh', 'ReLU', 'ELU', 'GELU', 'Sigmoid', 'Leaky ReLU', \
                    'SiLU', 'Softplus', 'LELU', 'BELU', 'Mish', 'NELE']
    loss_collect = np.zeros(len(activations))
    loss_collect = []
    val_collect = []
    test_collect = []
    activation = get_activation(activation_type, directory, config, device)

    # -------------------------
    # Training setup
    # -------------------------
    model = CIFAR100CNN(activation=activation).to(device)
    optimizer = Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=cfg.epochs,
    eta_min=0.0
    )
    val_dataset = []
    if exec_study == 1:
        # load the latest .pth file
        checkpoint_path = getlatest('RESULTS/CIFAR100/' + activation_type + '/')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] - 1
        with open('RESULTS/CIFAR100/' + activation_type + '/split_indices.pkl', 'rb') as f:
            train_indices, val_indices = pickle.load(f)
        full_train_set, _,test_dataset, _, _ = prepare_datasets(
            val_ratio=cfg.val_ratio,
            mode = 1
        )
        train_dataset = torch.utils.data.Subset(full_train_set, train_indices)
        if cfg.val_ratio != 0.0:
            val_dataset = torch.utils.data.Subset(full_train_set, val_indices)

    # Step 2: Prepare datasets
    else:
        start_epoch = 0
        train_dataset, val_dataset, test_dataset, train_indices, val_indices = prepare_datasets(
            val_ratio=cfg.val_ratio,
            mode = 0
        )
        with open('RESULTS/CIFAR100/' + activation_type + '/split_indices.pkl', 'wb') as f:
            pickle.dump((train_indices, val_indices), f)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, \
                              shuffle=True, num_workers=2, pin_memory=True)
    if cfg.val_ratio != 0.0:
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # -------------------------
    # Training loop skeleton
    # -------------------------
    correct_val, total_val = 0, 0
    correct_test, total_test = 0, 0
    print(f'Activation : {activation_type}')
    for epoch in range(start_epoch, config["epochs"]):
        epoch_loss = 0
        model.train()
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
            barred = '=' * filled_len + '-' * (bar_len - filled_len)

            # Print progress bar in-place
            sys.stdout.write(f'\rEpoch {epoch+1}/{config["epochs"]} |[{barred}]| '
                            f'Batch {i+1}/{total_batches} | \
                                Loss: {loss.item():.4f} | Time: {batch_time:.2f}s')
            sys.stdout.flush()

        epoch_loss /= len(train_loader.dataset)
        loss_collect.append(epoch_loss)
        scheduler.step()
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
            val_err = 1 - val_acc
        val_collect.append(val_acc)

        if exec_type == 2:
            if trial is not None:
                trial.report(val_err, epoch + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if val_err < best_val:
                best_val = val_acc
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= cfg.PATIENCE:
                loss_collect = torch.tensor(loss_collect)
                test_collect = torch.tensor(test_collect)
                val_collect = torch.tensor(val_collect)
                csv_write2(directory + '/loss_history_' + activation_type + '.csv',
                            torch.linspace(1, cfg.epochs+1, cfg.epochs+1),
                            loss_collect, 'epoch', 'loss', 'val', 'test', val_collect, test_collect, exec_type)
                return best_val, trial
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
            save(model, optimizer, epoch_loss, activation_type, epoch + 1, directory)
        print(f"\nEpoch {epoch+1}, loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}, \
              Test Acc: {100 * correct_test / max(total_test, 1):.4f}, \
                lr : {lr:.5f}, Time : {total_time:.4f}")

    loss_collect = torch.tensor(loss_collect)
    test_collect = torch.tensor(test_collect)
    val_collect = torch.tensor(val_collect)
    csv_write2(directory + '/loss_history_' + activation_type + '.csv',
              torch.linspace(1, cfg.epochs+1, cfg.epochs+1),
              loss_collect, 'epoch', 'loss', 'val', 'test', val_collect, test_collect, exec_study)
    if exec_type == 2:
        return best_val, trial
