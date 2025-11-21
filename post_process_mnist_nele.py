from csv_operations import csv_read
import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import config as cfg
activations =  cfg.AF_plot
activations_file =  cfg.AF
colors = ["#0000FF",
            "#ff0000",
            "#2ca02c",
            "#99342f", "#c7c7c7",
            '#000000']

def plot_only(study_type):
    fig, (ax1) = plt.subplots(1, 1, figsize = (5,3))
    ax1.set_ylabel('Training Loss')
    ax1.set_ylabel('Test Loss (%)')
    ax1.set_yscale('log')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Training loss')
    ax1.set_xlabel('epochs')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Loss (%)')
    for i, act in enumerate(activations_file):
        epochs, losses_train, losses_test = csv_read('RESULTS/' + study_type  + '/' + act + '/loss_history_' + activations_file[i] + '.csv', 'epoch', 'loss', '')
        print(f'{act} : {round(max(losses_test), 2)}, index : {losses_test.index(max(losses_test))}')
        ax1.plot(epochs, losses_train, color = colors[i], label=activations[i], linewidth = 0.7)
        selected_epochs = []
        selected_losses = []

        for e, l in zip(epochs, losses_test):
            if (e - 1) % 5 == 0:   # 1,6,11,...
                selected_epochs.append(e)
                selected_losses.append(l)
        selected_epochs.append(epochs[-1])
        selected_losses.append(losses_test[-1])
        ax2.plot(selected_epochs, selected_losses, color = colors[i], label=activations[i], linewidth = 0.7, linestyle = '--')

    ax1.set_xlim(1, cfg.epochs)
    ax1.set_xticks(range(1, cfg.epochs, cfg.epochs//3))
    ax1.grid(True, linewidth = 0.1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('PICS/' + study_type + '/nele_training_loss.pdf', transparent=False)
    
    plt.cla()
    plt.close()

def mnist_nele():
    plot_only('MNIST')
