from MNIST.study import mnist_data
from file_operations import reset_directory
from MNIST.validation import mnist_validation
import config as cfg

def mnist_train(device):
    print("Starting MNIST")
    #reset_directory('./RESULTS/MNIST')
    #reset_directory('./PICS/MNIST')
    for af in cfg.AF:
        mnist_data(cfg.epochs, cfg.learning_rate, device, af)

def mnist_eval(device):
    for af in cfg.AF:
        mnist_validation(cfg.epochs, device, af)
