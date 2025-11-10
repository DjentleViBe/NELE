import matplotlib.pyplot as plt
import torch
from csv_operations import csv_read

def plot_pred(predfiles, labels):
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    y = torch.sin(x) + 0.2 * torch.randn(x.size())
    plt.figure(figsize=(8,5))
    # Read the CSV file
    for predfile, label in zip(predfiles, labels):
        x_vals_tanh, y_preds_tanh = csv_read(predfile, 'x', 'y_pred')
        plt.plot(x_vals_tanh, y_preds_tanh, 'r', label=label)
    
    # Plot
    plt.scatter(x, y, label='Data', s=10)
    # plt.scatter(x_vals, y_preds, s=10, alpha=0.5)  # optional: scatter for points
    plt.xlabel('x')
    plt.ylabel('y_pred')
    plt.title('Curve fitting')
    plt.legend()
    plt.savefig('PICS/curve_fitting.pdf')