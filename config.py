################### NLSD ######################
learning_rate = 0.001
################### MNIST #####################
input_size = 28 * 28
hidden_size = 128
num_hidden_layers = 8
num_classes = 10
batch_size = 64
epochs = 20
save_every = 20
learning_rate = 0.001
# AF =  ['tanh', 'relu', 'elu', 'gelu', 
#       'sigmoid', 'leaky_relu', 'silu', 
#       'softplus', 'lelu', 'mish', 'nele']
AF =  ['nele=0.002', 'nele=0.001', 'nele=0.0001', 
       'nele=0.00001']
