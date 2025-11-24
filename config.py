epochs = 2000
save_every = 20
learning_rate = 0.01
################### NLSD ######################
AF_NLSD = ['softplus']
# ['sin', 'trig', 'exp', 'hyp', 'quad', 'exppoly']
FUNC_NLSD = ['quad']
################### MNIST #####################
input_size = 28 * 28
hidden_size = 128
num_hidden_layers = 8
num_classes = 10
batch_size = 64
#AF = ['nele=0.00001']
#AF_plot = ['lr=0.00001']
#AF =  ['tanh', 'relu', 'elu', 'gelu', 
#       'sigmoid', 'leaky_relu', 'silu', 
#       'softplus', 'lelu', 'mish', 'nele=0.002']
#AF_plot =  ['Tanh', 'ReLU', 'ELU', 'GELU', 
#       'Sigmoid', 'Leaky_ReLU', 'SiLU', 
#       'Softplus', 'LeLU', 'Mish', 'NELE']
#AF =  ['nele=0.003', 'nele=0.002', 'nele=0.001', 'nele=0.0001', 
#       'nele=0.00001']
#AF_plot =  ['lr=0.003', 'lr=0.002', 'lr=0.001', 'lr=0.0001', 
#       'lr=0.00001']
AF = ['nele=0.0003']
AF_plot = ['nele=0.0003']
################# CIFAR-10 ####################

################# PLOTS #######################
colors = ["#0072B2", "#56B4E9", 
            "#E69F00", "#F0B775",
            "#009E73", "#63C6A8",
            "#CC79A7", "#DDA5C9",
            "#8E6B55", "#C7B2A8",
            '#000000']
