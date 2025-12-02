from csv_operations import csv_read
import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import config as cfg
from MNIST.validation import mnist_validation
activations =  cfg.AF_plot
activations_file =  cfg.AF
colors = ["#0000FF",
            "#ff0000",
            "#2ca02c",
            "#99342f", "#c7c7c7",
            '#000000']

def mnist_nele(device, reset):
    for i, af in enumerate(cfg.AF):
        mnist_validation(cfg.epochs, device, cfg.noise_level, af)
