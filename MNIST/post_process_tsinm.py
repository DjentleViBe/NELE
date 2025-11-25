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
        if act == 'lelu':
            limit = cfg.epochs
        else:
            limit = cfg.epochs
        epochs, losses_train1, losses_test1 = csv_read('RESULTS/' + study_type  + '/' + act + '=1/loss_history_' + act + '=1.csv', 'epoch', 'loss', 'test')
        epochs, losses_train2, losses_test2 = csv_read('RESULTS/' + study_type  + '/' + act + '=2/loss_history_' + act + '=2.csv', 'epoch', 'loss', 'test')
        epochs, losses_train3, losses_test3 = csv_read('RESULTS/' + study_type  + '/' + act + '=3/loss_history_' + act + '=3.csv', 'epoch', 'loss', 'test')
        epochs, losses_train4, losses_test4 = csv_read('RESULTS/' + study_type  + '/' + act + '=4/loss_history_' + act + '=4.csv', 'epoch', 'loss', 'test')
        epochs, losses_train5, losses_test5 = csv_read('RESULTS/' + study_type  + '/' + act + '=5/loss_history_' + act + '=5.csv', 'epoch', 'loss', 'test')
        epochs, losses_train6, losses_test6 = csv_read('RESULTS/' + study_type  + '/' + act + '=6/loss_history_' + act + '=6.csv', 'epoch', 'loss', 'test')
        epochs, losses_train7, losses_test7 = csv_read('RESULTS/' + study_type  + '/' + act + '=7/loss_history_' + act + '=7.csv', 'epoch', 'loss', 'test')
        losses_train, losses_test = take_median(losses_train1[:limit], 
                                                losses_test1[:limit], 
                                                losses_train2[:limit], 
                                                losses_test2[:limit], 
                                                losses_train3[:limit], 
                                                losses_test3[:limit],
                                                losses_train4[:limit], 
                                                losses_test4[:limit],
                                                losses_train5[:limit], 
                                                losses_test5[:limit],
                                                losses_train6[:limit], 
                                                losses_test6[:limit],
                                                losses_train7[:limit], 
                                                losses_test7[:limit])
        max_test_loss = round(np.max(losses_test), 2)
        max_test_index = np.argmax(losses_test)
        min_train_loss = round(np.min(losses_train), 4)
        min_train_index = np.argmin(losses_train)

        print(f'{act} : Train = {min_train_loss}, index : {min_train_index}, Test = {max_test_loss}, index : {max_test_index}')

        ax1.plot(epochs[:limit], losses_train[:limit], colors[i], label=activations[i], linewidth = 0.7)
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
