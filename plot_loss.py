import matplotlib.pyplot as plt
from csv_operations import csv_read

def plot_loss(lossfiles, labels):
    colors = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9',
              '#74add1', '#4575b4', '#313695']
    plt.figure(figsize=(8,5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    for lossfile, label, color in zip(lossfiles, labels, colors):
        epochs, losses_tanh = csv_read(lossfile, 'epoch', 'loss')
        plt.plot(epochs, losses_tanh, color, label=label)
    # Plot
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('PICS/training_loss.pdf')