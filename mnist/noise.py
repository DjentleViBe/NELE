"""
Noise range evaluation
"""
import shutil
import matplotlib.pyplot as plt
import numpy as np
from mnist.validation import mnist_validation
import config as cfg

def noise_eval(device, af, i, j):
    """
    Evaluate MNIST for a range of noise values
    
    :param device: device name
    :param af: Activation function
    :param i: iterator
    """
    x = np.linspace(0, 3, 10)
    colors = cfg.colors
    afplot = cfg.AF_plot
    test_acc = []
    for noise_level in x:
        directory = './RESULTS/MNIST/' + af + '/'
        source_file = 'config.py'
        shutil.copy(source_file, './RESULTS/MNIST/')
        config = {}
        with open('./RESULTS/MNIST/' + "config.py") as f:
            exec(f.read(), config)
        test_acc_med = mnist_validation(directory, cfg.epochs, device, noise_level, af, config)
        test_acc.append(test_acc_med)
    plt.plot(x, test_acc, label = afplot[j], color = colors[j])
    plt.xlabel('Noise strength')
    plt.ylabel('Test accuracy (%)')
    plt.legend()
    plt.xticks(np.arange(0, 4, 1))
    plt.tight_layout()
    plt.grid(True, linewidth = 0.1)
    plt.savefig('PICS/MNIST/Noise_study.pdf')
