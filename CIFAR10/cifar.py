from CIFAR10.study import cifar10_data
from file_operations import reset_directory, create_directory
import config as cfg

def cifar10_train(device, reset, exec):
    print("Starting CIFAR10")
    if reset == 1:
        reset_directory('./RESULTS/CIFAR10')
        reset_directory('./PICS/CIFAR10')
    else:
        create_directory('./RESULTS/CIFAR10')
        create_directory('./PICS/CIFAR10')
    for af in cfg.AF:
        cifar10_data(cfg.epochs, cfg.learning_rate, device, exec, af)
                  