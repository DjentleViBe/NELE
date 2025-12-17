# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Docstring for CIFAR10.post_process_cifar10
"""
import matplotlib.pyplot as plt
from csv_operations import csv_read2
import config as cfg
activations =  cfg.AF_plot
activations_file =  cfg.AF
colors = cfg.colors

def plot_only(study_type):
    """
    Docstring for plot_only
    
    :param study_type: AF
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 3))
    fig.subplots_adjust(wspace = 0.2, right = 0.82, bottom=0.14)
    ax1.set_ylabel('Training Loss')
    ax2.set_ylabel('Test Accuracy (%)')
    ax1.set_yscale('log')
    ax1.set_xlabel('epochs')
    ax2.set_xlabel('epochs')
    for i, act in enumerate(activations_file):
        epochs, losses_train, _, losses_test = csv_read2('RESULTS/' \
                                        + study_type  + '/' + act + '/loss_history_' \
                                            + act + '.csv', 'epoch', \
                                            'loss', 'val', 'test')
        print(f'{act} : {round(max(losses_test), 2)}, \
              index : {losses_test.index(max(losses_test))}')
        ax1.plot(epochs, losses_train, colors[i], label=act, linewidth = 0.7)
        selected_epochs = []
        selected_losses = []

        for e, l in zip(epochs, losses_test):
            if e % 5 == 0:   # 1,6,11,...
                selected_epochs.append(e)
                selected_losses.append(100 - l)
        selected_epochs.append(epochs[-1])
        selected_losses.append(losses_test[-1])
        ax2.plot(selected_epochs, selected_losses, color = colors[i],\
                  label=act, linewidth = 0.7)

    ax1.set_xlim(1, 200)
    ax2.set_xlim(101, 200)
    ax2.set_ylim(8, 15)
    ax1.set_xticks(range(1, 200, 50))
    ax2.set_xticks(range(101, 200, 50))
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

def process_cifar10():
    """
    Docstring for process_cifar10
    """
    plot_only('CIFAR10')
