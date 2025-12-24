"""
Docstring for CIFAR10.cifar
"""
import shutil
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
        directory = 'RESULTS/CIFAR10/' + af + '/'
        create_directory('RESULTS/CIFAR10/' + af + '/')
        create_directory('PICS/CIFAR10/' + af + '/')
        print(f'Activation : {af}')
        source_file = 'config.py'
        shutil.copy(source_file, directory)
        config = {}
        with open(directory + "config.py") as f:
            exec(f.read(), config)
        cifar10_data(directory, device, exec_study, 0, config, af)

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
        directory = 'RESULTS/CIFAR10/' + af + '/'
        create_directory('RESULTS/CIFAR10/' + af + '/')
        create_directory('PICS/CIFAR10/' + af + '/')
        print(f'Activation : {af}')
        source_file = 'config.py'
        shutil.copy(source_file, directory)
        config = {}
        with open(directory + "config.py") as f:
            exec(f.read(), config)
        cifar10_data(directory, device, exec_study, 0, config, af)

def cifar10_train_study(device, reset, exec_study):
    """
    Docstring for cifar10_train
    
    :param device: device name
    :param reset: reset== 1 clears folder
    :param exec_study: standalone or 7 runs
    """
    print("Starting CIFAR10 study")
    if reset == 1:
        reset_directory('./RESULTS/CIFAR10')
        reset_directory('./PICS/CIFAR10')
    else:
        create_directory('./RESULTS/CIFAR10')
        create_directory('./PICS/CIFAR10')
    for af in cfg.STUDY_3:
        directory = 'RESULTS/CIFAR10/' + af + '/'
        create_directory('RESULTS/CIFAR10/' + af + '/')
        create_directory('PICS/CIFAR10/' + af + '/')
        print(f'Activation : {af}')
        source_file = 'config.py'
        shutil.copy(source_file, directory)
        config = {}
        with open(directory + "config.py") as f:
            exec(f.read(), config)
        cifar10_data(directory, device, exec_study, 0, config, af)
