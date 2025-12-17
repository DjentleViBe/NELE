from CIFAR100.study import cifar100_data
from file_operations import reset_directory, create_directory
import config as cfg

def cifar100_train(device, reset, exec):
    print("Starting CIFAR100")
    if reset == 1:
        reset_directory('./RESULTS/CIFAR100')
        reset_directory('./PICS/CIFAR100')
    else:
        create_directory('./RESULTS/CIFAR100')
        create_directory('./PICS/CIFAR100')
    for af in cfg.AF_CIFAR100:
        cifar100_data(cfg.epochs, cfg.learning_rate, device, exec, af)

def cifar100_train_nele(device, reset, exec):
    print("Starting CIFAR100")
    if reset == 1:
        reset_directory('./RESULTS/CIFAR100')
        reset_directory('./PICS/CIFAR100')
    else:
        create_directory('./RESULTS/CIFAR100')
        create_directory('./PICS/CIFAR100')
    for af in cfg.AF_CIFAR100_NELE:
        cifar100_data(cfg.epochs, cfg.learning_rate, device, exec, af)
                  