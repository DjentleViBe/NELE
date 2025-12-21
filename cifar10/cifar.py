"""
Docstring for CIFAR10.cifar
"""
from cifar10.study import cifar10_data
from file_operations import reset_directory, create_directory
import config as cfg

def cifar10_train(device, reset, exec_study):
    """
    Docstring for cifar10_train
    
    :param device: device name
    :param reset: reset== 1 clears folder
    :param exec_study: standalone or 7 runs
    """
    print("Starting CIFAR10")
    if reset == 1:
        reset_directory('./RESULTS/CIFAR10')
        reset_directory('./PICS/CIFAR10')
    else:
        create_directory('./RESULTS/CIFAR10')
        create_directory('./PICS/CIFAR10')
    for af in cfg.AF_CIFAR10:
        cifar10_data(cfg.epochs, cfg.learning_rate, device, exec_study, af)

def cifar10_train_nele(device, reset, exec_study):
    """
    Docstring for cifar10_train_nele

    :param device: device name
    :param reset: reset== 1 clears folder
    :param exec_study: standalone or 7 runs
    """
    print("Starting CIFAR10")
    if reset == 1:
        reset_directory('./RESULTS/CIFAR10')
        reset_directory('./PICS/CIFAR10')
    else:
        create_directory('./RESULTS/CIFAR10')
        create_directory('./PICS/CIFAR10')
    for af in cfg.AF_CIFAR10_NELE:
        cifar10_data(cfg.epochs, cfg.learning_rate, device, exec_study, af)
