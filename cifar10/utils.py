# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
utils for CIFAR10
"""
import numpy as np
import torch
from torch import nn
from models.model_lelu import LELU
from models.model_nele import NeleUniversalRead

def get_activation(activation_type, directory, proparray, device=None):
    """
    Return PyTorch activation module based on a string.
    
    Supports optional parameter: e.g., 'elu=0.5'
    """
    if '=' in activation_type:
        base, param = activation_type.split('=')
        param = float(param)
    else:
        base = activation_type
        param = None
    config = {}
    directory_order = directory + '/'
    with open(directory_order + "config.py") as f:
        exec(f.read(), config)
    dispatch = {
        'relu': nn.ReLU(),
        'elu' : nn.ELU(alpha=param or 1.0),
        'leaky_relu' : nn.LeakyReLU(negative_slope=0.1),
        'gelu' : nn.GELU(),
        'mish' : nn.Mish(),
        'sigmoid' : nn.Sigmoid(),
        'silu' : nn.SiLU(),
        'softplus' : nn.Softplus(),
        'tanh': nn.Tanh(),
        'lelu' : LELU(),
        'nele' : NeleUniversalRead(device, proparray),
    }

    return dispatch[base]

def getselectedlosses(epochs, losses_test):
    """
    Docstring for getselectedlosses
    """
    selected_epochs = []
    selected_losses = []
    for e, l in zip(epochs, losses_test):
        if (e - 1) % 5 == 0:   # 1,6,11,...
            selected_epochs.append(e)
            selected_losses.append(100-l)
    selected_epochs.append(epochs[-1])
    selected_losses.append(100 - losses_test[-1])
    return selected_epochs, selected_losses

def getselectedlosses2(epochs, losses_test):
    """
    Docstring for getselectedlosses
    """
    selected_epochs = []
    selected_losses = []
    for e, l in zip(epochs, losses_test):
        if (e - 1) % 5 == 0:   # 1,6,11,...
            selected_epochs.append(e)
            selected_losses.append(100-l)
    selected_epochs.append(epochs[-1])
    selected_losses.append(losses_test[-1])
    return selected_epochs, selected_losses

def save(model, optimizer, epoch_loss, activation_type, epoch, directory, test_loss = 0.0):
    """
    Docstring for saving the model
    
    :param model: model
    :param optimizer: model optimizer
    :param epoch_loss: training loss value
    :param activation_type: AF
    :param epoch: total epoch at save state
    :param directory: directory location
    :param test_loss: test score
    """
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch_loss': epoch_loss,
    'test_loss': test_loss
    }, directory + activation_type + '_' + str(epoch) + '.pth')


def take_median(*args):
    """
    Compute median across multiple runs for train and test losses/accuracies.
    
    Arguments:
        args: losses/accuracies arrays in the order:
              train1, test1, train2, test2, train3, test3, ...
              Number of args must be even (train/test pairs).
              
    Returns:
        median_train: np.array of median training values per epoch
        median_test: np.array of median test values per epoch
    """
    if len(args) % 2 != 0:
        raise ValueError("Number of arguments must be even (train/test pairs).")

    # Separate train and test arrays
    train_arrays = args[::2]
    test_arrays  = args[1::2]

    # Stack along new axis and take median along runs
    median_train = np.median(np.stack(train_arrays, axis=0), axis=0)
    median_test  = np.median(np.stack(test_arrays, axis=0), axis=0)

    return median_train, median_test
