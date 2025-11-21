from csv_operations import csv_read
import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import config as cfg
activations =  cfg.AF_plot
activations_file =  cfg.AF
colors = cfg.colors

def plot_only(study_type):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 3))
    fig.subplots_adjust(wspace = 0.2, right = 0.82, bottom=0.14) 
    # plt.bar(activations, std_deviation_collect, yerr=loss_collect, capsize=5, , ecolor="#c7c7c7", error_kw={"elinewidth": 2})
    ax1.set_ylabel('Training Loss')
    ax2.set_ylabel('Test Loss (%)')
   
    ax1.set_yscale('log')
    ax1.set_xlabel('epochs')
    ax3 = ax2.twinx()
    ax4 = ax1.twiny()
    ax5 = ax2.twiny()
    ax3.set_ylabel('Test Loss (%)')
    ax2.set_xlabel('epochs')
    for i, act in enumerate(activations_file):
        epochs, losses_train, losses_test = csv_read('RESULTS/' + study_type  + '/' + act + '/loss_history_' + activations_file[i] + '.csv', 'epoch', 'loss', '')
        print(f'{act} : {round(max(losses_test), 2)}, index : {losses_test.index(max(losses_test))}')
        if act == 'nele' or act =='lelu':
            ax4.plot(epochs, losses_train, color = colors[i], label=activations[i], linewidth = 0.7)
        else:
            ax1.plot(epochs, losses_train, colors[i], label=activations[i], linewidth = 0.7)
        selected_epochs = []
        selected_losses = []

        for e, l in zip(epochs, losses_test):
            if (e - 1) % 5 == 0:   # 1,6,11,...
                selected_epochs.append(e)
                selected_losses.append(l)
        selected_epochs.append(epochs[-1])
        selected_losses.append(losses_test[-1])
        if act == 'sigmoid' or act == 'softplus':
            ax3.plot(selected_epochs, selected_losses, color = colors[i], label=activations[i], linewidth = 0.7)
        elif act == 'nele' or act =='lelu':
            ax5.plot(selected_epochs, selected_losses, color = colors[i], label=activations[i], linewidth = 0.7)
        else:
            ax2.plot(selected_epochs, selected_losses, color = colors[i], label=activations[i], linewidth = 0.7)
    
    ax1.set_xlim(1, 20)
    ax2.set_xlim(1, 20)
    ax3.set_xlim(1, 20)
    ax4.set_xlim(1, int(max(epochs)))
    ax5.set_xlim(1, int(max(epochs)))
    ax1.set_xticks(range(1, 20, 5))
    ax2.set_xticks(range(1, 20, 5))
    ax3.set_xticks(range(1, 20, 5))
    ax1.grid(True, linewidth = 0.1)
    ax2.grid(True, linewidth = 0.1)
    lines = []
    labels = []

    for ax in [ax1, ax4]:
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab
    
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor = (0.945, 0.1))
    # fig.tight_layout()
    plt.savefig('PICS/' + study_type + '/training_loss.pdf', transparent=False)

    plt.cla()
    plt.close()

def process_mnist():
    plot_only('MNIST')
