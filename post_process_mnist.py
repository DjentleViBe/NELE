from csv_operations import csv_read
import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#activations =  ['Tanh', 'Sigmoid', 'Softplus', 'ELU', 'SiLU', 'GELU', 'ReLU', 'Leaky ReLU', 'LeLU', 'Mish', 'NELE']
#activations_file =  ['tanh', 'sigmoid', 'softplus', 'elu', 'silu', 'gelu', 'relu', 'lrelu', 'lelu', 'mish', 'nele']
activations_file =  ['lelu', 'nele']
activations =  ['LeLU', 'NELE']
colors = ["#0000FF", "#aec7e8", 
            "#ff0000", "#ffbb78",
            "#2ca02c", "#98df8a",
            "#8000FF", "#d57dc1",
            "#99342f", "#c7c7c7",
            '#000000']

def plot_only(study_type):
    
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

    for i, act in enumerate(activations_file):
        epochs, losses_tanh, _ = csv_read('RESULTS/' + study_type  + '/' + act + '/loss_history_' + activations_file[i] + '.csv', 'epoch', 'loss', '')
        plt.plot(epochs, losses_tanh, colors[i], label=activations[i], linewidth = 0.7)
    # Plot
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('PICS/' + study_type + '/training_loss.pdf', transparent=False)

    plt.cla()
    plt.close()

plot_only('MNIST')

    
