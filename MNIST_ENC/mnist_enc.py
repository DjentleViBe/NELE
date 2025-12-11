from MNIST_ENC.study import mnist_enc_data
from file_operations import reset_directory, create_directory
from MNIST_ENC.validation import mnist_enc_validation
from MNIST_ENC.noise import noise_eval
import config as cfg

def mnist_enc_train(device, reset, exec):
    print("Starting MNIST autoencoder")
    if reset == 1:
        reset_directory('./RESULTS/MNIST_ENC')
        reset_directory('./PICS/MNIST_ENC')
    else:
        create_directory('./RESULTS/MNIST_ENC')
        create_directory('./PICS/MNIST_ENC')
    for af in cfg.AF:
        print(f'Activation : {af}')
        mnist_enc_data(cfg.epochs, cfg.learning_rate, device, exec, af)
    
def mnist_enc_train_nele(device, reset, exec):
    print("Starting MNIST autoencoder")
    if reset == 1:
        reset_directory('./RESULTS/MNIST_ENC')
        reset_directory('./PICS/MNIST_ENC')
    else:
        create_directory('./RESULTS/MNIST_ENC')
        create_directory('./PICS/MNIST_ENC')
    for af in cfg.AF_nele:
        print(f'Activation : {af}')
        mnist_enc_data(cfg.epochs, cfg.learning_rate, device, exec, af)

def mnist_enc_eval(device, reset):
    if reset == 1:
        reset_directory('./RESULTS/MNIST_ENC')
        reset_directory('./PICS/MNIST_ENC')
    else:
        create_directory('./RESULTS/MNIST_ENC')
        create_directory('./PICS/MNIST_ENC')
    for i, af in enumerate(cfg.AF):
        # print(f'Activation : {af}')
        # mnist_validation(cfg.epochs, device, cfg.noise_level, af)
        noise_eval(device, af, i)
    
def mnist_enc_train_study(device, reset, exec):
    print("Starting MNIST autoencoder")
    if reset == 1:
        reset_directory('./RESULTS/MNIST_ENC')
        reset_directory('./PICS/MNIST_ENC')
    else:
        create_directory('./RESULTS/MNIST_ENC')
        create_directory('./PICS/MNIST_ENC')
    for af in cfg.STUDY:
        print(f'Activation : {af}')
        mnist_enc_data(cfg.epochs, cfg.learning_rate, device, exec, af)
    
