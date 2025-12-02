from MNIST.study import mnist_data
from file_operations import reset_directory, create_directory
from MNIST.validation import mnist_validation
from MNIST.noise import noise_eval
import config as cfg

def mnist_train(device, reset, exec):
    print("Starting MNIST")
    if reset == 1:
        reset_directory('./RESULTS/MNIST')
        reset_directory('./PICS/MNIST')
    else:
        create_directory('./RESULTS/MNIST')
        create_directory('./PICS/MNIST')
    for af in cfg.AF:
        print(f'Activation : {af}')
        mnist_data(cfg.epochs, cfg.learning_rate, device, exec, af)
    
def mnist_train_nele(device, reset, exec):
    print("Starting MNIST")
    if reset == 1:
        reset_directory('./RESULTS/MNIST')
        reset_directory('./PICS/MNIST')
    else:
        create_directory('./RESULTS/MNIST')
        create_directory('./PICS/MNIST')
    for af in cfg.AF_nele:
        print(f'Activation : {af}')
        mnist_data(cfg.epochs, cfg.learning_rate, device, exec, af)

def mnist_eval(device, reset):
    if reset == 1:
        reset_directory('./RESULTS/MNIST')
        reset_directory('./PICS/MNIST')
    else:
        create_directory('./RESULTS/MNIST')
        create_directory('./PICS/MNIST')
    for i, af in enumerate(cfg.AF):
        # print(f'Activation : {af}')
        # mnist_validation(cfg.epochs, device, cfg.noise_level, af)
        noise_eval(device, af, i)
