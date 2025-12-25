# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
"""
Docstring for mnist.study
"""
import numpy as np
import optuna
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets, transforms
from file_operations import create_directory
from csv_operations import csv_write2
from mnist.utils import get_activation
from mnist.neuralnet import DeepFCNet
from mnist.utils import save
import config as cfg

def mnist_data(directory, device, exec_type, config=None, activation_type='default', trial = None):
    """
    MNIST training
    
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
    activation = get_activation(activation_type, directory, config, device)
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ])

    train_dataset_full = datasets.MNIST(root='./data', train=True, \
                                        transform=transform, download=True)
    # Compute split sizes
    n_total = len(train_dataset_full)
    n_val = int(n_total * config["val_ratio"])
    n_train = n_total - n_val

    # Split
    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model, loss, optimizer
    model = DeepFCNet(config["input_size"], config["hidden_size"], config["num_hidden_layers"], \
                      config["num_classes"], activation, activation_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = criterion.to(device)

    loss_collect = []
    val_collect = []
    test_collect = []
    # Training loop
    correct_val, total_val = 0.0, 0.0
    correct_test, total_test = 0.0, 0.0
    test_loss_0, test_loss_3 = 0.0, 0.0
    if exec_type == 2:
        config["epochs"] = cfg.HYPER_EPOCHS
    for epoch in range(config["epochs"]):
        epoch_loss = 0
        model.train()
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        for _, (data, target) in enumerate(train_loader):
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
        val_err = 0.0
        if config["val_ratio"] != 0:
            model.eval()
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    _, predicted = outputs.max(1)
                    total_val += y.size(0)
                    correct_val += (predicted == y).sum().item()
            val_acc = correct_val / total_val
            val_err = 1 - val_acc
        val_collect.append(val_acc)
        if exec_type == 2:
            if trial is not None:
                trial.report(val_err, epoch + 1)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if val_err < best_val:
                best_val = val_err
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

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # torch.manual_seed(1234)
            correct_test, total_test = 0.0, 0.0
            model.eval()
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += target.size(0)
                    correct_test += (predicted == target).sum().item()
            test_loss_0 = 100 * correct_test / max(total_test, 1)
            correct_test, total_test = 0.0, 0.0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    noise = torch.empty_like(data).uniform_(-cfg.noise_level, cfg.noise_level)
                    x_noisy = data + noise
                    outputs = model(x_noisy)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += target.size(0)
                    correct_test += (predicted == target).sum().item()
            test_loss_3 = 100 * correct_test / max(total_test, 1)
        test_collect.append(test_loss_0)
        if (epoch + 1) % cfg.save_every  == 0:
            save(model, optimizer, epoch_loss, activation_type, epoch, directory, test_loss_0)
        print(f"Epoch {epoch+1}, loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}," 
              f"Test Acc 0: {test_loss_0:.4f}, Test Acc 3: {test_loss_3:.4f}, lr : {lr:.5f}")

    # Evaluate
    loss_collect = torch.tensor(loss_collect)
    test_collect = torch.tensor(test_collect)
    val_collect = torch.tensor(val_collect)
    csv_write2(directory + '/loss_history_' + activation_type + '.csv',
              torch.linspace(1, cfg.epochs+1, cfg.epochs+1),
              loss_collect, 'epoch', 'loss', 'val', 'test', val_collect, test_collect, exec_type)
    if exec_type == 2:
        return best_val, trial
