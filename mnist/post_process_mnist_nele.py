"""
Docstring for mnist.post_process_mnist_nele
"""
import config as cfg
from mnist.validation import mnist_validation

def mnist_nele(device):
    """
    Docstring for mnist_nele
    
    :param device: device name
    :param reset: Description
    """
    for af in cfg.AF:
        mnist_validation(cfg.epochs, device, cfg.noise_level, af)
