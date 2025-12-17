"""
MNIST main
"""
from mnist_enc.study import mnist_enc_data
from mnist_enc.noise import noise_eval
from file_operations import reset_directory, create_directory
import config as cfg

def mnist_enc_train(device, reset, exec_study):
    """
    Docstring for mnist autoencoder train
    
    :param device: device name
    :param reset: reset==1 clears folder
    :param exec_study: study or singular nanalysis
    """
    print("Starting MNIST autoencoder")
    if reset == 1:
        reset_directory('./RESULTS/MNIST_ENC')
        reset_directory('./PICS/MNIST_ENC')
    else:
        create_directory('./RESULTS/MNIST_ENC')
        create_directory('./PICS/MNIST_ENC')
    for af in cfg.AF:
        print(f'Activation : {af}')
        mnist_enc_data(cfg.epochs, cfg.learning_rate, device, exec_study, af)

def mnist_enc_train_nele(device, reset, exec_study):
    """
    Docstring for mnist autoencoder train with NELE AF
    
    :param device: device name
    :param reset: reset==1 clears folder
    :param exec: study or singular nanalysis
    """
    print("Starting MNIST autoencoder")
    if reset == 1:
        reset_directory('./RESULTS/MNIST_ENC')
        reset_directory('./PICS/MNIST_ENC')
    else:
        create_directory('./RESULTS/MNIST_ENC')
        create_directory('./PICS/MNIST_ENC')
    for af in cfg.AF_nele:
        print(f'Activation : {af}')
        mnist_enc_data(cfg.epochs, cfg.learning_rate, device, exec_study, af)

def mnist_enc_eval(device, reset):
    """
    Docstring for mnist autoencoder validation with various noise levels
    
    :param device: device name
    :param reset: reset==1 clears folder
    """
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

def mnist_enc_train_study(device, reset, exec_study):
    """
    Docstring for mnist autoencoder train with 7 runs
    
    :param device: device name
    :param reset: reset==1 clears folder
    :param exec: study or singular nanalysis
    """
    print("Starting MNIST autoencoder")
    if reset == 1:
        reset_directory('./RESULTS/MNIST_ENC')
        reset_directory('./PICS/MNIST_ENC')
    else:
        create_directory('./RESULTS/MNIST_ENC')
        create_directory('./PICS/MNIST_ENC')
    for af in cfg.STUDY:
        print(f'Activation : {af}')
        mnist_enc_data(cfg.epochs, cfg.learning_rate, device, exec_study, af)
