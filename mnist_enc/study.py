# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
"""
Docstring for mnist autoencoder study
"""
import sys
import time
import numpy as np
import optuna
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from mnist_enc.neuralnet import DeepAutoencoder
from csv_operations import csv_write2
import config as cfg
from mnist.utils import get_activation
from mnist.utils import save

def mnist_enc_data(directory, device, exec_type, config=None, activation_type='default', trial=None):
    """
    Docstring for mnist_enc_data
    
    :param epochs: total epochs
    :param learn_rate: learning rate
    :param device: device name
    :param exec_type: study or standalone
    :param activation_type: Description
    """
    best_val = float('inf')
    activations =  ['Tanh', 'ReLU', 'ELU', 'GELU', 'Sigmoid', 'Leaky ReLU', \
                    'SiLU', 'Softplus', 'LELU', 'BELU', 'Mish', 'NELE']
    loss_collect = np.zeros(len(activations))
    # Activation function selection
    activation = get_activation(activation_type, directory, config, device)

    # Load MNIST
    transform = transforms.ToTensor()

    train_dataset_full = datasets.MNIST(root='./data', \
                            train=True, transform=transform, download=True)
    # Compute split sizes
    n_total = len(train_dataset_full)
    n_val = int(n_total * config["val_ratio"])
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
    model = DeepAutoencoder(activation).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = criterion.to(device)

    loss_collect = []
    val_collect = []
    test_collect = []
    # Training loop
    if exec_type == 2:
        config["epochs"] = cfg.HYPER_EPOCHS
    for epoch in range(config["epochs"]):
        total_batches = len(train_loader)
        total_time = 0
        epoch_loss = 0
        model.train()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        for i, (data, target) in enumerate(train_loader):
            start_epoch = time.time()
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
            end_epoch = time.time()
            batch_time = end_epoch - start_epoch
            total_time += batch_time
            # Progress bar
            bar_len = 30
            filled_len = int(round(bar_len * (i + 1) / total_batches))
            barred = '=' * filled_len + '-' * (bar_len - filled_len)
            # Print progress bar in-place
            sys.stdout.write(f"\rEpoch {epoch+1}/{config["epochs"]} |[{barred}]|"
                            f"Batch {i+1}/{total_batches} | Loss: {loss.item():.4f}"
                            f"| Time: {batch_time:.2f}s")
            sys.stdout.flush()
        epoch_loss /= len(train_loader.dataset)
        loss_collect.append(epoch_loss)

        # Validation
        val_loss = 0.0
        if config["val_ratio"] != 0:
            model.eval()
            with torch.no_grad():
                for x, _ in val_loader:  # labels not needed
                    x = x.to(device)
                    outputs = model(x)
                    loss = criterion(outputs, x)  # compare to input
                    val_loss += loss.item() * x.size(0)
            val_loss /= len(val_loader.dataset)
        val_collect.append(val_loss)
    
        if exec_type == 2:
            if trial is not None:
                trial.report(val_loss, epoch + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if val_loss < best_val:
                best_val = val_loss
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
            save(model, optimizer, epoch_loss, activation_type, epoch, directory, test_loss_clean)
        print(f"\nEpoch {epoch+1}, loss: {epoch_loss:.4f}, Val Acc: {val_loss:.4f},"
              f"Test Acc 0: {test_loss_clean:.4f}, Test Acc 3: {test_loss_noisy:.4f}, lr : {lr:.5f}, "
              f"Time : {total_time:.4f}")

    # Evaluate
    loss_collect = torch.tensor(loss_collect)
    test_collect = torch.tensor(test_collect)
    val_collect = torch.tensor(val_collect)
    csv_write2(directory + '/loss_history_' + activation_type + '.csv',
              torch.linspace(1, cfg.epochs+1, cfg.epochs+1),
              loss_collect, 'epoch', 'loss', 'val', 'test', val_collect, test_collect, exec_type)
    if exec_type == 2:
        return best_val, trial
