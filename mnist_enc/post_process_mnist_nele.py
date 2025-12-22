"""
Docstring for MNIST_ENC.post_process_mnist_nele
"""
import config as cfg
from mnist_enc.validation import mnist_enc_validation
activations =  cfg.AF_plot
activations_file =  cfg.AF
colors = ["#0000FF",
            "#ff0000",
            "#2ca02c",
            "#99342f", "#c7c7c7",
            '#000000']

def mnist_enc_nele(device):
    """
    Docstring for mnist_enc_nele
    
    :param device: device name
    """
    for af in cfg.AF:
        mnist_enc_validation(cfg.epochs, device, cfg.noise_level, af)
