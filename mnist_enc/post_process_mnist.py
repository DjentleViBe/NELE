# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Docstring for mnist_enc.post_process_mnist
"""
import numpy as np
import matplotlib.pyplot as plt
from csv_operations import csv_read
import config as cfg
activations =  cfg.AF_plot
activations_file =  cfg.AF
colors = cfg.colors

def plot_only(study_type):
    """
    Docstring for plot_only
    
    :param study_type: Description
    """
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize = (11, 3))
    fig2.subplots_adjust(wspace = 0.2, right = 0.82, bottom=0.14)
    ax3.set_ylabel('Training Loss')
    ax4.set_ylabel('Test Error (%)')

    ax3.set_yscale('log')
    ax3.set_xlabel('epochs')
    ax4.set_xlabel('epochs')

    for i, act in enumerate(activations_file):
        act = act.split('=')[0]
        limit = cfg.epochs
        epochs, losses_train, losses_test = csv_read('RESULTS/' + \
                                        study_type  + '/' + act \
                                        + '/loss_history_' + act \
                                        + '.csv', 'epoch', 'loss', 'test')

        max_test_loss = round(np.max(losses_test), 2)
        max_test_index = np.argmax(losses_test)
        min_train_loss = round(np.min(losses_train), 4)
        min_train_index = np.argmin(losses_train)

        print(f'{act} : Train = {min_train_loss}, \
              index : {min_train_index}, Test error = {round(100 - max_test_loss, 2)}, \
                index : {max_test_index}')

        ax3.plot(epochs[:limit], losses_train[:limit], \
                 colors[i], label=activations[i], linewidth = 0.7)
        selected_epochs = []
        selected_losses = []
        for e, l in zip(epochs, losses_test):
            if (e - 1) % 5 == 0:   # 1,6,11,...
                selected_epochs.append(e)
                selected_losses.append(l)
        selected_epochs.append(epochs[-1])
        selected_losses.append(losses_test[-1])
        ax4.plot(selected_epochs, selected_losses, \
                 color = colors[i], label=activations[i], linewidth = 0.7)

    ax3.set_xlim(1, cfg.epochs)
    ax4.set_xlim(1, cfg.epochs)
    ax4.set_yscale("log")
    ax3.set_xticks(range(1, cfg.epochs, 50))
    ax4.set_xticks(range(1, cfg.epochs, 50))
    ax3.grid(True, linewidth = 0.1)
    ax4.grid(True, linewidth = 0.1)
    lines = []
    labels = []

    for ax in [ax3]:
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab

    fig2.legend(lines, labels, loc='lower center', \
               bbox_to_anchor = (0.915, 0.1))
    plt.savefig('PICS/' + study_type + '/training_loss.pdf', \
                transparent=False)

    plt.cla()
    plt.close()

def process_mnist_enc():
    """
    Docstring for process_mnist_enc
    """
    plot_only('mnist_enc')
