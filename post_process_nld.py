from csv_operations import csv_read
import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
activations =  ['Tanh', 'Sigmoid', 'Softplus', 'ELU', 'SiLU', 'GELU', 'ReLU', 'Leaky ReLU', 'LeLU', 'Mish', 'NELE']
activations_file =  ['tanh', 'sigmoid', 'softplus', 'elu', 'silu', 'gelu', 'relu', 'lrelu', 'lelu', 'mish', 'nele']
colors = ["#490092", "#006ddb", 
          "#b66dff", "#ff6db6",
          "#920000", "#db6d00",
          "#ffdf4d", "#004949",
          "#009999", "#22cf22",
          '#000000']

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
    mpl.rcParams['pdf.use14corefonts'] = False
    mpl.rcParams['pdf.fonttype'] = 42  # keeps colors in RGB
    plt.style.use("tableau-colorblind10")
    plt.figure(figsize=(8,4))
    plt.bar(activations, std_deviation_collect, color=colors)
    plt.errorbar(activations, std_deviation_collect, yerr=loss_collect, fmt='none', ecolor="black", elinewidth=3, capsize=5)
    plt.errorbar(activations, std_deviation_collect, yerr=loss_collect, fmt='none', ecolor="white", elinewidth=0, capsize=3)
    # plt.bar(activations, std_deviation_collect, yerr=loss_collect, capsize=5, , ecolor="#c7c7c7", error_kw={"elinewidth": 2})
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./PICS/' + study_type + '/Loss_Bar_Chart_with_Error_Bars.pdf', transparent=False)
    plt.cla()
    plt.close()

    plt.figure(figsize=(8,5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    for i, act in enumerate(activations):
        epochs, losses_tanh, _ = csv_read(dir + '/loss_history_' + activations_file[i] + '.csv', 'epoch', 'loss', '')
        plt.plot(epochs, losses_tanh, colors[i], label=activations[i], linewidth = 0.8)
    # Plot
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linewidth=0.1)
    plt.tight_layout()
    plt.savefig('PICS/' + study_type + '/training_loss.pdf', transparent=False)

    plt.cla()
    plt.close()

    plt.figure(figsize=(8,5))
    for i, act in enumerate(activations):
        x_vals, y_preds, y = csv_read(dir + '/predictions_' + activations_file[i] + '.csv', 'x', 'y_pred','y_actual')
        plt.plot(x_vals, y_preds, colors[i], label=activations[i], linewidth=0.7)
    
    # Plot
    plt.scatter(x, y, label='Data', color = 'k', s=10)
    # plt.scatter(x_vals, y_preds, s=10, alpha=0.5)  # optional: scatter for points
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('PICS/' + study_type + '/curve_fitting.pdf', transparent=False)

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
    
