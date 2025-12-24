"""
Docstring for CIFAR100.cifar
"""
import shutil
from cifar100.study import cifar100_data
from file_operations import reset_directory, create_directory
import config as cfg

def cifar100_train(device, reset, exec_study):
    """
    Docstring for cifar100_train
    
    :param device: device name
    :param reset: reset== 1 clears folder
    :param exec_study: standalone or 7 runs
    """
    print("Starting CIFAR100")
    if reset == 1:
        reset_directory('./RESULTS/CIFAR100')
        reset_directory('./PICS/CIFAR100')
    else:
        create_directory('./RESULTS/CIFAR100')
        create_directory('./PICS/CIFAR100')
    for af in cfg.AF_CIFAR100:
        directory = 'RESULTS/CIFAR100/' + af + '/'
        create_directory('RESULTS/CIFAR100/' + af + '/')
        create_directory('PICS/CIFAR100/' + af + '/')
        print(f'Activation : {af}')
        source_file = 'config.py'
        shutil.copy(source_file, directory)
        config = {}
        with open(directory + "config.py") as f:
            exec(f.read(), config)
        cifar100_data(directory, device, exec_study, 0, config, af)

def cifar100_train_nele(device, reset, exec_study):
    """
    Docstring for cifar100_train_nele

    :param device: device name
    :param reset: reset== 1 clears folder
    :param exec_study: standalone or 7 runs
    """
    print("Starting CIFAR100")
    if reset == 1:
        reset_directory('./RESULTS/CIFAR100')
        reset_directory('./PICS/CIFAR100')
    else:
        create_directory('./RESULTS/CIFAR100')
        create_directory('./PICS/CIFAR100')
    for af in cfg.AF_CIFAR100_NELE:
        directory = 'RESULTS/CIFAR100/' + af + '/'
        create_directory('RESULTS/CIFAR100/' + af + '/')
        create_directory('PICS/CIFAR100/' + af + '/')
        print(f'Activation : {af}')
        source_file = 'config.py'
        shutil.copy(source_file, directory)
        config = {}
        with open(directory + "config.py") as f:
            exec(f.read(), config)
        cifar100_data(directory, device, exec_study, 0, config, af)

def cifar100_train_study(device, reset, exec_study):
    """
    Docstring for cifar100_train
    
    :param device: device name
    :param reset: reset== 1 clears folder
    :param exec_study: standalone or 7 runs
    """
    print("Starting CIFAR100")
    if reset == 1:
        reset_directory('./RESULTS/CIFAR100')
        reset_directory('./PICS/CIFAR100')
    else:
        create_directory('./RESULTS/CIFAR100')
        create_directory('./PICS/CIFAR100')
    for af in cfg.STUDY_3:
        directory = 'RESULTS/CIFAR100/' + af + '/'
        create_directory('RESULTS/CIFAR100/' + af + '/')
        create_directory('PICS/CIFAR100/' + af + '/')
        print(f'Activation : {af}')
        source_file = 'config.py'
        shutil.copy(source_file, directory)
        config = {}
        with open(directory + "config.py") as f:
            exec(f.read(), config)
        cifar100_data(directory, device, exec_study, 0, config, af)
