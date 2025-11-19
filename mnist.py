from MNIST.study import mnist_data
from file_operations import reset_directory
from MNIST.validation import mnist_validation

def mnist(device):
    print("Starting MNIST")
    #reset_directory('./RESULTS/MNIST')
    #reset_directory('./PICS/MNIST')
    mnist_data(20, 0.01, device, 'elu')
    mnist_data(20, 0.01, device, 'gelu')
    mnist_data(20, 0.01, device, 'relu')
    mnist_data(20, 0.01, device, 'leaky_relu')
    mnist_data(20, 0.01, device, 'sigmoid')
    mnist_data(20, 0.01, device, 'softplus')
    mnist_data(20, 0.01, device, 'tanh')
    mnist_data(20, 0.01, device, 'mish')
    mnist_data(20, 0.01, device, 'silu')
    mnist_data(20, 0.01, device, 'lelu')
    # mnist_data(20, 0.001, 'nele')
    
    mnist_validation(20,'tanh')
    mnist_validation(20, 'softplus')
    mnist_validation( 20,'sigmoid')
    mnist_validation( 20,'elu')
    mnist_validation( 20,'silu')
    mnist_validation( 20,'gelu')
    mnist_validation( 20,'relu')
    mnist_validation( 20,'leaky_relu')
    mnist_validation( 20,'mish')