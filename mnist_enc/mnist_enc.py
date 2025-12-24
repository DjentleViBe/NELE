"""
MNIST main
"""
import shutil
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
        directory = 'RESULTS/MNIST_ENC/' + af + '/'
        create_directory('RESULTS/MNIST_ENC/' + af + '/')
        create_directory('PICS/MNIST_ENC/' + af + '/')
        source_file = 'config.py'
        shutil.copy(source_file, directory)
        config = {}
        with open(directory + "config.py") as f:
            exec(f.read(), config)
        mnist_enc_data(directory, device, exec_study, config, af)

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
        directory = 'RESULTS/MNIST_ENC/' + af + '/'
        create_directory('RESULTS/MNIST_ENC/' + af + '/')
        create_directory('PICS/MNIST_ENC/' + af + '/')
        print(f'Activation : {af}')
        source_file = 'config.py'
        shutil.copy(source_file, directory)
        config = {}
        with open(directory + "config.py") as f:
            exec(f.read(), config)
        mnist_enc_data(directory, device, exec_study, config, af)

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
        directory = 'RESULTS/MNIST_ENC/' + af + '/'
        create_directory('RESULTS/MNIST_ENC/' + af + '/')
        create_directory('PICS/MNIST_ENC/' + af + '/')
        source_file = 'config.py'
        shutil.copy(source_file, directory)
        config = {}
        with open(directory + "config.py") as f:
            exec(f.read(), config)
        mnist_enc_data(directory, device, exec_study, config, af)
