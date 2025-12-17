# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""
Plot predictions
"""
import matplotlib.pyplot as plt
from csv_operations import csv_read

def plot_pred(x, y, predfiles, labels, study_type='default'):
    """
    Docstring for plot_pred
    
    :param x: X values
    :param y: Y values
    :param predfiles: files to be read
    :param labels: x labels
    :param study_type: analysis type
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
              '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#17becf',
              '#bcbd22', '#000000',
              ]
    plt.figure(figsize=(8,5))
    # Read the CSV file
    for predfile, label, color in zip(predfiles, labels, colors):
        x_vals_tanh, y_preds_tanh, _ = csv_read(predfile, 'x', 'y_pred','y_actual')
        plt.plot(x_vals_tanh, y_preds_tanh, color, label=label, linewidth=0.7)

    # Plot
    plt.scatter(x, y, label='Data', color = 'k', s=10)
    # plt.scatter(x_vals, y_preds, s=10, alpha=0.5)  # optional: scatter for points
    plt.xlabel('x')
    plt.ylabel('y_pred')
    plt.title(study_type)
    plt.legend()
    plt.tight_layout()
    plt.savefig('PICS/' + study_type + '/curve_fitting.pdf')
