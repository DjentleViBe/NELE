# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Docstring for CIFAR10.post_process_cifar10
"""
import matplotlib.pyplot as plt
import numpy as np
from csv_operations import csv_read
import config as cfg
from cifar10.utils import getselectedlosses, take_median, \
                            getselectedlosses2
activations =  cfg.AF_plot
activations_file =  cfg.AF
colors = cfg.colors

def plot_only(study_type, ylabelname):
    """
    Docstring for plot_only
    
    :param study_type: AF
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 3))
    fig.subplots_adjust(wspace = 0.2, right = 0.82, bottom=0.14)
    ax1.set_ylabel('Training Loss')
    ax2.set_ylabel(ylabelname)
    ax1.set_yscale('log')
    ax1.set_xlabel('epochs')
    ax2.set_xlabel('epochs')
    for i, act in enumerate(activations_file):
        act = act.split('=')[0]
        limit = cfg.epochs
        epochs, losses_train1, losses_test1 = csv_read('RESULTS/' \
                                            + study_type  + '/' + act \
                                            + '=1/loss_history_' + act \
                                            + '=1.csv', 'epoch', 'loss', 'test')
        epochs, losses_train2, losses_test2 = csv_read('RESULTS/' \
                                            + study_type  + '/' + act \
                                            + '=2/loss_history_' + act \
                                            + '=2.csv', 'epoch', 'loss', 'test')
        epochs, losses_train3, losses_test3 = csv_read('RESULTS/' \
                                            + study_type  + '/' + act \
                                            + '=3/loss_history_' + act \
                                            + '=3.csv', 'epoch', 'loss', 'test')
        losses_train, losses_test, std_train, std_test = take_median(losses_train1[:limit],
                                                losses_test1[:limit],
                                                losses_train2[:limit],
                                                losses_test2[:limit],
                                                losses_train3[:limit],
                                                losses_test3[:limit])
        max_test_loss = round(np.max(losses_test), 2)
        max_test_index = np.argmax(losses_test)
        min_train_loss = round(np.min(losses_train), 4)
        min_train_index = np.argmin(losses_train)
        print(f'{act} : {round(max_test_loss, 2)}, \
              index : {max_test_index}')
        ax1.plot(epochs[:limit], losses_train[:limit], colors[i], \
                 label=activations[i], linewidth = 0.7)
        selected_epochs, selected_losses = getselectedlosses(epochs, losses_test)
        selected_epochs, std_losses = getselectedlosses(epochs, std_test)
        std_losses = np.asarray(std_losses)
        std_losses = 100 - std_losses
        ax2.plot(selected_epochs, selected_losses, color = colors[i],\
                  label=activations[i], linewidth = 0.7)
        if 'nele' in activations_file[i]:
            plt.fill_between(selected_epochs,
                    np.asarray(selected_losses) - np.asarray(std_losses),
                    np.asarray(selected_losses) + np.asarray(std_losses),
                    alpha=0.15, color = 'k')
    ax1.set_xlim(1, cfg.epochs)
    ax2.set_xlim(1, cfg.epochs)
    ax1.set_xticks(range(1, cfg.epochs, cfg.epochs//4))
    ax2.set_xticks(range(1, cfg.epochs, cfg.epochs//4))
    ax1.grid(True, linewidth = 0.1)
    ax2.grid(True, linewidth = 0.1)
    lines = []
    labels = []

    for ax in [ax1]:
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab

    fig.legend(lines, labels, loc='lower center', bbox_to_anchor = (0.91, 0.1))
    plt.savefig('PICS/' + study_type + '/training_loss.pdf', transparent=False)
    plt.tight_layout()
    plt.cla()
    plt.close()

def process_cifar_study(study, ylabelname):
    """
    Docstring for process_cifar10
    """
    plot_only(study, ylabelname)
