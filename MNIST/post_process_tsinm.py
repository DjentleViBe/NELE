from csv_operations import csv_read
import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import config as cfg
activations =  cfg.AF_plot
activations_file =  cfg.AF
colors = cfg.colors

def take_median(*args):
    """
    Compute median across multiple runs for train and test losses/accuracies.
    
    Arguments:
        args: losses/accuracies arrays in the order:
              train1, test1, train2, test2, train3, test3, ...
              Number of args must be even (train/test pairs).
              
    Returns:
        median_train: np.array of median training values per epoch
        median_test: np.array of median test values per epoch
    """
    if len(args) % 2 != 0:
        raise ValueError("Number of arguments must be even (train/test pairs).")
    
    # Separate train and test arrays
    train_arrays = args[::2]
    test_arrays  = args[1::2]
    
    # Stack along new axis and take median along runs
    median_train = np.median(np.stack(train_arrays, axis=0), axis=0)
    median_test  = np.median(np.stack(test_arrays, axis=0), axis=0)
    
    return median_train, median_test

def plot_only(study_type):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 3))
    fig.subplots_adjust(wspace = 0.2, right = 0.82, bottom=0.14) 
    # plt.bar(activations, std_deviation_collect, yerr=loss_collect, capsize=5, , ecolor="#c7c7c7", error_kw={"elinewidth": 2})
    ax1.set_ylabel('Training Loss')
    ax2.set_ylabel('Test Error (%)')
   
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.set_xlabel('epochs')
    #ax3 = ax2.twinx()
    #ax4 = ax1.twiny()
    #ax5 = ax2.twiny()
    #ax3.set_ylabel('Test Accuracy (%)')
    ax2.set_xlabel('epochs')

    for i, act in enumerate(activations_file):
        act = act.split('=')[0]
        epochs, losses_train1, losses_test1 = csv_read('RESULTS/' + study_type  + '/' + act + '=1/loss_history_' + act + '=1.csv', 'epoch', 'loss', 'test')
        epochs, losses_train2, losses_test2 = csv_read('RESULTS/' + study_type  + '/' + act + '=2/loss_history_' + act + '=2.csv', 'epoch', 'loss', 'test')
        epochs, losses_train3, losses_test3 = csv_read('RESULTS/' + study_type  + '/' + act + '=3/loss_history_' + act + '=3.csv', 'epoch', 'loss', 'test')
        losses_train, losses_test = take_median(losses_train1, losses_test1, losses_train2, losses_test2, losses_train3, losses_test3)
        max_loss = round(np.max(losses_test), 2)
        max_index = np.argmax(losses_test)

        print(f'{act} : {max_loss}, index : {max_index}')

        ax1.plot(epochs, losses_train, colors[i], label=activations[i], linewidth = 0.7)
        selected_epochs = []
        selected_losses = []
        for e, l in zip(epochs, losses_test):
            if (e - 1) % 5 == 0:   # 1,6,11,...
                selected_epochs.append(e)
                selected_losses.append(100-l)
        selected_epochs.append(epochs[-1])
        selected_losses.append(100 - losses_test[-1])
        ax2.plot(selected_epochs, selected_losses, color = colors[i], label=activations[i], linewidth = 0.7)
    
    ax1.set_xlim(1, cfg.epochs)
    ax2.set_xlim(1, cfg.epochs)
    #ax3.set_xlim(1, cfg.epochs)
    #ax4.set_xlim(1, int(max(epochs)))
    #ax5.set_xlim(1, int(max(epochs)))
    ax1.set_xticks(range(1, cfg.epochs, 10))
    ax2.set_xticks(range(1, cfg.epochs, 10))
    #ax3.set_xticks(range(1, cfg.epochs, 10))
    ax1.grid(True, linewidth = 0.1)
    ax2.grid(True, linewidth = 0.1)
    lines = []
    labels = []

    for ax in [ax1]:
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
