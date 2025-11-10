import matplotlib.pyplot as plt
from csv_operations import csv_read

def plot_loss(lossfiles, labels):
    plt.figure(figsize=(8,5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    for lossfile, label in zip(lossfiles, labels):
        epochs, losses_tanh = csv_read(lossfile, 'epoch', 'loss')
        plt.plot(epochs, losses_tanh, 'k', label=label)
    # Plot
    
    plt.legend()
    plt.grid(True)
    plt.savefig('PICS/training_loss.pdf')