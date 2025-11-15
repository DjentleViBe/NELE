from csv_operations import csv_read
import torch 
import numpy as np
import matplotlib.pyplot as plt

activations =  ['Tanh', 'ReLU', 'ELU', 'GELU', 'Sigmoid', 'Leaky ReLU', 'SiLU', 'Softplus', 'LELU', 'BELU', 'NELU']
activations_file =  ['tanh', 'relu', 'elu', 'gelu', 'sigmoid', 'lrelu', 'silu', 'softplus', 'lelu', 'belu', 'nelu']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 
           '#d62728', '#9467bd', '#8c564b', 
           '#e377c2', '#7f7f7f', '#17becf',
           '#bcbd22', '#000000',
            ]

def plot_only(x, study_type):
    loss_collect = np.zeros(len(activations))
    std_deviation_collect = np.zeros(len(activations))
    for i, act in enumerate(activations_file):
        dir = 'RESULTS/' + study_type + '/'
        y, predicted, y_actual = csv_read(dir + '/predictions_' + act + '.csv', 'x', 'y_pred', 'y_actual')
        y_actual = torch.tensor(y_actual)
        predicted = torch.tensor(predicted)
        sigma_est = torch.std(y_actual - predicted)

        _, loss, _ = csv_read(dir + '/loss_history_' + act + '.csv', 'epoch', 'loss', '')
        loss_collect[i] = loss[-1]
        std_deviation_collect[i] = sigma_est.item()
    plt.figure(figsize=(8,4))
    plt.bar(activations, std_deviation_collect, yerr=loss_collect, capsize=5, color=colors, alpha=0.7)
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(study_type)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./PICS/' + study_type + '/Loss_Bar_Chart_with_Error_Bars.pdf')

x = torch.linspace(-5, 5, 200).unsqueeze(1)

################### SINE NOISE ##########################
plot_only(x, 'sine_noise')

################### TRIG NOISE ##########################
plot_only(x, 'trig_noise')

################### EXP NOISE ##########################

plot_only(x, 'exp_noise')

################### HYP NOISE ##########################

plot_only(x, 'hyp_noise')

################### QUAD NOISE ##########################

y = x**2 + 0.2 * torch.randn(x.size())
plot_only(x, 'quad_noise')

################### EXP-POLY NOISE ##########################
x = torch.linspace(0, 10, 200).unsqueeze(1)
y = x**3 / (torch.exp(x) - 1 + 1E-6) + 0.2 * torch.randn(x.size())
plot_only(x, 'exppoly_noise')
    
