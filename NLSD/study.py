# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
"""
Non Linear Synthetic Data
"""

import torch
from torch import nn
from torch import optim
import config as cfg
from models.model_lelu import LELU
from models.model_nele import NeleUniversal
from csv_operations import csv_write
from file_operations import create_directory
from NLSD.neuralnet import Net

def nlsd_data(x, y, af, device='cpu', study_type='default'):
    """
    Docstring for nlsd_data
    
    :param x: X values
    :param y: Y values
    :param af: Activation function type
    :param device: device name
    :param study_type: Study name
    """
    loss_collect = []
    directory = 'RESULTS/NLSD/' + study_type + '/'
    learning_rate = cfg.learning_rate
    if '=' in af:
        base, param = af.split('=')
        param = float(param)  # convert parameter to float if needed
    else:
        base = af
        param = None

    dispatch = {
        'relu': nn.ReLU(),
        'elu' : nn.ELU(alpha=1.0),
        'leaky_relu' : nn.LeakyReLU(negative_slope=0.1),
        'gelu' : nn.GELU(),
        'mish' : nn.Mish(),
        'sigmoid' : nn.Sigmoid(),
        'silu' : nn.SiLU(),
        'softplus' : nn.Softplus(),
        'tanh': nn.Tanh(),
        'lelu' : LELU(),
        'nele' : NeleUniversal()
    }
    activation = dispatch[base]

    directory = directory + '/' + af
    create_directory('RESULTS/NLSD/' + study_type + '/' + base + '/')
    create_directory('PICS/NLSD/' + study_type + '/' + base + '/')

    model = Net(x.shape[1], activation=activation)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    loss_collect = []
    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_collect.append(loss.item())
        if (epoch+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{cfg.epochs}], Loss: {loss.item():.4f}')

    # Evaluate model
    model.eval()
    predicted = model(x).detach()
    loss_collect = torch.tensor(loss_collect)

    # Write to CSV
    pred_file = directory + '/predictions_' + af + '.csv'
    loss_file = directory + '/loss_history_' + af + '.csv'
    csv_write(pred_file, x, predicted, 'x' , 'y_pred', 'y_actual', y)
    csv_write(loss_file, torch.linspace(1, cfg.epochs, cfg.epochs), \
              loss_collect,  'epoch', 'loss', '', torch.linspace(1, cfg.epochs, cfg.epochs))
    sigma_est = torch.std(y - predicted)
    return loss.item(), sigma_est.item()
