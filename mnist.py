from MNIST.study import mnist_data
from file_operations import reset_directory
from MNIST.validation import mnist_validation

def mnist():
    print("Starting MNIST")
    #reset_directory('./RESULTS/MNIST')
    #reset_directory('./PICS/MNIST')
    '''
    mnist_data(50, 0.01, 'elu')
    
    mnist_data(50, 0.01, 'gelu')
    mnist_data(50, 0.01, 'relu')
    mnist_data(50, 0.01, 'leaky_relu')
    mnist_data(50, 0.01, 'sigmoid')
    mnist_data(50, 0.01, 'softplus')
    mnist_data(50, 0.01, 'tanh')
    mnist_data(50, 0.01, 'mish')
    mnist_data(50, 0.01, 'silu')
    mnist_data(50, 0.01, 'lelu')
    mnist_data(50, 0.001, 'nele')
    
    mnist_validation(50,'tanh')
    mnist_validation(50, 'softplus')
    mnist_validation( 50,'sigmoid')
    mnist_validation( 50,'elu')
    mnist_validation( 50,'silu')
    mnist_validation( 50,'gelu')
    mnist_validation( 50,'relu')
    mnist_validation( 50,'leaky_relu')
    mnist_validation( 50,'mish')
    
    mnist_validation( 50,'lelu')
    mnist_validation(30, 'nele')
    '''
    mnist_data(100, 0.01, 'nele')
    #mnist_data(100, 0.01, 'lelu')
    #mnist_validation(100,'lelu')