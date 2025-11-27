epochs = 75
save_every = 75
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
noise_level = 3.0
#AF = ['nele=0.002']
#AF_plot = ['lr=0.002']
"""
AF =  ['tanh=1', 'relu=1', 'elu=1', 'gelu=1', 
       'sigmoid=1', 'leaky_relu=1', 'silu=1', 
       'softplus=1', 'nele=1', 'mish=1', 'nele=4',
       'tanh=2', 'relu=2', 'elu=2', 'gelu=2', 
       'sigmoid=2', 'leaky_relu=2', 'silu=2', 
       'softplus=2', 'nele=2', 'mish=2', 'nele=5',
       'tanh=3', 'relu=3', 'elu=3', 'gelu=3', 
       'sigmoid=3', 'leaky_relu=3', 'silu=3', 
       'softplus=3', 'nele=3', 'mish=3', 'nele=6',
       'tanh=4', 'relu=4', 'elu=4', 'gelu=4', 
       'sigmoid=4', 'leaky_relu=4', 'silu=4', 
       'softplus=4', 'nele=4', 'mish=4', 'nele=7',
       'tanh=5', 'relu=5', 'elu=5', 'gelu=5', 
       'sigmoid=5', 'leaky_relu=5', 'silu=5', 
       'softplus=5', 'nele=5', 'mish=5', 
       'tanh=6', 'relu=6', 'elu=6', 'gelu=6', 
       'sigmoid=6', 'leaky_relu=6', 'silu=6', 
       'softplus=6', 'nele=6', 'mish=6', 
       'tanh=7', 'relu=7', 'elu=7', 'gelu=7', 
       'sigmoid=7', 'leaky_relu=7', 'silu=7', 
       'softplus=7', 'nele=7', 'mish=7'
       ]
AF_plot =  ['tanh=1', 'relu=1', 'elu=1', 'gelu=1', 
       'sigmoid=1', 'leaky_relu=1', 'silu=1', 
       'softplus=1', 'nele=1', 'mish=1', 'nele=4',
       'tanh=2', 'relu=2', 'elu=2', 'gelu=2', 
       'sigmoid=2', 'leaky_relu=2', 'silu=2', 
       'softplus=2', 'nele=2', 'mish=2', 'nele=5',
       'tanh=3', 'relu=3', 'elu=3', 'gelu=3', 
       'sigmoid=3', 'leaky_relu=3', 'silu=3', 
       'softplus=3', 'nele=3', 'mish=3', 'nele=6',
       'tanh=4', 'relu=4', 'elu=4', 'gelu=4', 
       'sigmoid=4', 'leaky_relu=4', 'silu=4', 
       'softplus=4', 'nele=4', 'mish=4', 'nele=7',
       'tanh=5', 'relu=5', 'elu=5', 'gelu=5', 
       'sigmoid=5', 'leaky_relu=5', 'silu=5', 
       'softplus=5', 'nele=5', 'mish=5', 
       'tanh=6', 'relu=6', 'elu=6', 'gelu=6', 
       'sigmoid=6', 'leaky_relu=6', 'silu=6', 
       'softplus=6', 'nele=6', 'mish=6', 
       'tanh=7', 'relu=7', 'elu=7', 'gelu=7', 
       'sigmoid=7', 'leaky_relu=7', 'silu=7', 
       'softplus=7', 'nele=7', 'mish=7']
       """
AF_nele =  ['nele=1', 'nele=2', 'nele=3', 'nele=4', 'nele=5', 'nele=6', 'nele=7']
AF_plot_nele =  ['nele', 'nele', 'nele', 'nele', 'nele', 'nele', 'nele']
AF = [ 'tanh', 'sigmoid', 'softplus', 'elu', 'silu', 'gelu', 'relu',
     'leaky_relu', 
      'lelu', 'mish', 'nele']
AF_plot = ['Tanh', 'Sigmoid', 'Softplus',
           'ELU', 'SiLU', 'GELU', 'ReLU',
       'Leaky_ReLU',  
       'LeLU', 'Mish', 'NELE']
# AF = ['nele']
# AF_plot = ['nele']
################# CIFAR-10 ####################
val_ratio = 0.0
################# PLOTS #######################
colors = ['#000000',"#0072B2", "#56B4E9", 
            "#E69F00", "#F0B775",
            "#009E73", "#63C6A8",
            "#CC79A7", "#DDA5C9",
            "#8E6B55", "#C7B2A8"
            ]
