from plot_pred import plot_pred
from plot_loss import plot_loss
from matplotlib import pyplot as plt
from file_operations import create_directory
import numpy as np
import config as cfg
from models.model_lelu import LELU
from models.model_nele import NELE
import torch.nn as nn
import torch
from csv_operations import csv_write
import torch.optim as optim
from NLSD.NeuralNet import Net

def nlsd_data(x, y, af, device='cpu', study_type='default'):
    loss_collect = []
    val_collect = []
    test_collect = []
    dir = 'RESULTS/NLSD/' + study_type + '/'
    learning_rate = cfg.learning_rate
    if '=' in af:
        base, param = af.split('=')
        param = float(param)  # convert parameter to float if needed
    else:
        base = af
        param = None
    if base == 'relu':
        activation = nn.ReLU()
        learning_rate = cfg.learning_rate_array[6]
    elif base == 'elu':
        activation = nn.ELU(alpha=1.0)
        learning_rate = cfg.learning_rate_array[3]
    elif base == 'leaky_relu' :
        activation = nn.LeakyReLU(negative_slope=0.1)
        learning_rate = cfg.learning_rate_array[7]
    elif base == 'gelu' :
        activation = nn.GELU()
        learning_rate = cfg.learning_rate_array[5]
    elif base == 'mish' :
        activation = nn.Mish()
        learning_rate = cfg.learning_rate_array[9]
    elif base == 'sigmoid' :
        activation = nn.Sigmoid()
        learning_rate = cfg.learning_rate_array[1]
    elif base == 'silu' :
        activation = nn.SiLU()
        learning_rate = cfg.learning_rate_array[4]
    elif base == 'softplus' :
        activation = nn.Softplus()
        learning_rate = cfg.learning_rate_array[2]
    elif base == 'tanh':
        activation = nn.Tanh()
        learning_rate = cfg.learning_rate_array[0]
    elif base == 'lelu' :
        activation = LELU()
        learning_rate = cfg.learning_rate_array[8]
    elif base == 'nele' :
        activation = NELE(1, 3, 2)
        learning_rate = cfg.learning_rate_array[10]
    else:
        raise ValueError("Invalid activation type")
    dir = dir + '/' + af
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
    pred_file = dir + '/predictions_' + af + '.csv'
    loss_file = dir + '/loss_history_' + af + '.csv'
    csv_write(pred_file, x, predicted, 'x' , 'y_pred', 'y_actual', y)
    csv_write(loss_file, torch.linspace(1, cfg.epochs, cfg.epochs), loss_collect,  'epoch', 'loss', '', torch.linspace(1, cfg.epochs, cfg.epochs))
    sigma_est = torch.std(y - predicted)
    return loss.item(), sigma_est.item() 
