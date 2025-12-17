# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Docstring for mnist.post_process_mnist
"""
import matplotlib.pyplot as plt
from csv_operations import csv_read
import config as cfg
from mnist.utils import getselectedlosses
activations =  cfg.AF_plot
activations_file =  cfg.AF
colors = cfg.colors

def plot_only(study_type):
    """
    Docstring for plot_only
    
    :param study_type: AF name
    """
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize = (11, 3))
    fig2.subplots_adjust(wspace = 0.2, right = 0.82, bottom=0.14)
    ax3.set_ylabel('Training Loss')
    ax4.set_ylabel('Test Accuracy (%)')

    ax3.set_yscale('log')
    ax3.set_xlabel('epochs')
    ax3 = ax4.twinx()
    ax4 = ax3.twiny()
    ax5 = ax4.twiny()
    ax3.set_ylabel('Test Accuracy (%)')
    ax4.set_xlabel('epochs')
    for i, act in enumerate(activations_file):
        epochs, losses_train, losses_test = csv_read('RESULTS/' + \
                                            study_type  + '/' + act + '/loss_history_' \
                                            + act + '.csv', 'epoch', 'loss', '')
        print(f'{act} : {round(max(losses_test), 2)}, \
              index : {losses_test.index(max(losses_test))}')
        if 'nele' in act:
            ax4.plot(epochs, losses_train, color = colors[i], label=act, linewidth = 0.7)
        else:
            ax3.plot(epochs, losses_train, colors[i], label=act, linewidth = 0.7)
        selected_losses, selected_epochs = getselectedlosses(epochs, losses_test)
        if act in ('sigmoid', 'softplus'):
            ax3.plot(selected_epochs, selected_losses, \
                     color = colors[i], label=act, linewidth = 0.7)
        elif 'nele' in act or 'lelu' in act:
            ax5.plot(selected_epochs, selected_losses, \
                     color = colors[i], label=act, linewidth = 0.7)
        else:
            ax4.plot(selected_epochs, selected_losses, \
                     color = colors[i], label=act, linewidth = 0.7)

    ax3.set_xlim(1, 20)
    ax4.set_xlim(1, 20)
    ax3.set_xlim(1, 20)
    ax4.set_xlim(1, int(max(epochs)))
    ax5.set_xlim(1, int(max(epochs)))
    ax3.set_xticks(range(1, 20, 5))
    ax4.set_xticks(range(1, 20, 5))
    ax3.set_xticks(range(1, 20, 5))
    ax3.grid(True, linewidth = 0.1)
    ax4.grid(True, linewidth = 0.1)
    lines = []
    labels = []

    for ax in [ax3, ax4]:
        l, lab = ax.get_legend_handles_labels()
        lines += l
        labels += lab

    fig2.legend(lines, labels, loc='lower center', bbox_to_anchor = (0.945, 0.1))
    plt.savefig('PICS/' + study_type + '/training_loss.pdf', transparent=False)

    plt.cla()
    plt.close()

def process_mnist():
    """
    Docstring for process_mnist
    """
    plot_only('MNIST')
