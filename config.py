epochs = 50
save_every = 50
learning_rate = 0.001
################### NLSD ######################
AF_NLSD = ['nele']
# ['sin', 'trig', 'exp', 'hyp', 'quad', 'exppoly']
FUNC_NLSD = ['quad']
################### MNIST #####################
input_size = 28 * 28
hidden_size = 128
num_hidden_layers = 8
num_classes = 10
batch_size = 128
#AF = ['nele=0.00001']
#AF_plot = ['lr=0.00001']
AF =  ['tanh=1', 'relu=1', 'elu=1', 'gelu=1', 
       'sigmoid=1', 'leaky_relu=1', 'silu=1', 
       'softplus=1', 'lelu=1', 'mish=1']
AF_plot =  ['Tanh', 'ReLU', 'ELU', 'GELU', 
       'Sigmoid', 'Leaky_ReLU', 'SiLU', 
       'Softplus', 'LeLU', 'Mish']
#AF =  ['nele=0.003', 'nele=0.002', 'nele=1', 'nele=1', 
#       'nele=0.00001']
#AF_plot =  ['lr=0.003', 'lr=0.002', 'lr=1', 'lr=1', 
#       'lr=0.00001']
#AF = ['tanh', 'relu', 'elu', 'gelu', 
#       'sigmoid', 'leaky_relu', 'silu', 
#       'softplus', 'lelu', 'mish', 'nele=0.002']
#AF_plot = ['Tanh', 'ReLU', 'ELU', 'GELU', 
#       'Sigmoid', 'Leaky_ReLU', 'SiLU', 
#       'Softplus', 'LeLU', 'Mish', 'NELE']
#AF = ['nele=17']
#AF_plot = ['nele=17']
################# CIFAR-10 ####################
val_ratio = 0.0
################# PLOTS #######################
colors = ["#0072B2", "#56B4E9", 
            "#E69F00", "#F0B775",
            "#009E73", "#63C6A8",
            "#CC79A7", "#DDA5C9",
            "#8E6B55", "#C7B2A8",
            '#000000']
