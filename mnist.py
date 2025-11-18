from MNIST.study import mnist_data
from file_operations import reset_directory
from MNIST.validation import mnist_validation

def mnist():
    print("Starting MNIST")
    #reset_directory('./RESULTS/MNIST')
    #reset_directory('./PICS/MNIST')
    #mnist_data(300, 0.01, 'elu')
    #mnist_data(300, 0.01, 'gelu')
    #mnist_data(300, 0.01, 'relu')
    #mnist_data(300, 0.01, 'leaky_relu')
    #mnist_data(300, 0.01, 'sigmoid')
    #mnist_data(300, 0.01, 'softplus')
    #mnist_data(300, 0.01, 'tanh')
    #mnist_data(300, 0.01, 'mish')
    #mnist_data(300, 0.01, 'silu')

    mnist_validation( 'tanh')
    mnist_validation( 'softplus')
    mnist_validation( 'sigmoid')
    mnist_validation( 'elu')
    mnist_validation( 'silu')
    mnist_validation( 'gelu')
    mnist_validation( 'relu')
    mnist_validation( 'leaky_relu')
    mnist_validation( 'mish')
