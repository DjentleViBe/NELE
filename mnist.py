from MNIST.study import mnist_data
from file_operations import reset_directory

def mnist():
    reset_directory('./RESULTS/MNIST')
    reset_directory('./PICS/MNIST')
    mnist_data(20, 0.01, 'elu')
