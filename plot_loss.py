import matplotlib.pyplot as plt
from csv_operations import csv_read

def plot_loss(lossfiles, labels):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 
              '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#000000',
              '#bcbd22', 
              '#17becf']
    plt.figure(figsize=(8,5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    for lossfile, label, color in zip(lossfiles, labels, colors):
        epochs, losses_tanh = csv_read(lossfile, 'epoch', 'loss')
        plt.plot(epochs, losses_tanh, color, label=label, linewidth = 0.7)
    # Plot
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('PICS/training_loss.pdf')