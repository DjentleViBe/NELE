from non_linear.non_linear_tanh import tanh_net
from non_linear.non_linear_relu import relu_net
from non_linear.non_linear_elu import elu_net
from non_linear.non_linear_gelu import gelu_net
from non_linear.non_linear_sigmoid import sigmoid_net
from non_linear.non_linear_lrelu import lrelu_net
from non_linear.non_linear_silu import silu_net
from non_linear.non_linear_softplus import softplus_net
from non_linear.non_linear_belu import belu_net
from non_linear.non_linear_lelu import lelu_net
from plot_pred import plot_pred
from plot_loss import plot_loss
from matplotlib import pyplot as plt
import numpy as np

def study_data(x, y, epochs, learn_rate, study_type='default'):
    activations =  ['Tanh', 'ReLU', 'ELU', 'GELU', 'Sigmoid', 'Leaky ReLU', 'SiLU', 'Softplus', 'LELU', 'BELU']
    loss_collect = np.zeros(len(activations))
    std_deviation_collect = np.zeros(len(activations))
    dir =  'RESULTS/' + study_type + '/'
    print("Running analysis - Tanh Activation Function")
    loss_collect[0], std_deviation_collect[0] = tanh_net(x, y, dir + '/predictions_tanh.csv', 
                                                         dir + '/loss_history_tanh.csv', learn_rate, epochs)
    print("Running analysis - ReLU Activation Function")
    loss_collect[1], std_deviation_collect[1] = relu_net(x, y, dir + '/predictions_relu.csv', 
                                                         dir + '/loss_history_relu.csv', learn_rate, epochs)
    print("Running analysis - ELU Activation Function")
    loss_collect[2], std_deviation_collect[2] = elu_net(x, y, dir + '/predictions_elu.csv', 
                                                        dir + '/loss_history_elu.csv', learn_rate, epochs)
    print("Running analysis - GELU Activation Function")
    loss_collect[3], std_deviation_collect[3] = gelu_net(x, y, dir + '/predictions_gelu.csv', 
                                                         dir + '/loss_history_gelu.csv', learn_rate, epochs)
    print("Running analysis - Sigmoid Activation Function")
    loss_collect[4], std_deviation_collect[4] = sigmoid_net(x, y, dir + '/predictions_sigmoid.csv', 
                                                            dir + '/loss_history_sigmoid.csv', learn_rate, epochs)
    print("Running analysis - Leaky ReLU Activation Function")
    loss_collect[5], std_deviation_collect[5] = lrelu_net(x, y, dir + '/predictions_lrelu.csv', 
                                                          dir + '/loss_history_lrelu.csv', learn_rate, epochs)
    print("Running analysis - SiLU Activation Function")
    loss_collect[6], std_deviation_collect[6] = silu_net(x, y, dir + '/predictions_silu.csv', 
                                                         dir + '/loss_history_silu.csv', learn_rate, epochs)
    print("Running analysis - softplus Activation Function")
    loss_collect[7], std_deviation_collect[7] = softplus_net(x, y, dir + '/predictions_softplus.csv', 
                                                             dir + '/loss_history_softplus.csv', learn_rate, epochs)
    print("Running analysis - LELU Activation Function")
    loss_collect[8], std_deviation_collect[8] = lelu_net(x, y, dir + '/predictions_lelu.csv', 
                                                             dir + '/loss_history_lelu.csv', learn_rate, epochs)
    
    print("Running analysis - BELU Activation Function")
    loss_collect[9], std_deviation_collect[9] = belu_net(x, y, dir + '/predictions_belu.csv', 
                                                         dir + '/loss_history_belu.csv', 20, learn_rate, epochs)
    
    print("Plotting results")
    plot_pred(x, y,
              [dir + '/predictions_tanh.csv',
               dir + '/predictions_relu.csv',
               dir + '/predictions_elu.csv',
               dir + '/predictions_gelu.csv',
               dir + '/predictions_sigmoid.csv',
               dir + '/predictions_lrelu.csv',
               dir + '/predictions_silu.csv',
               dir + '/predictions_softplus.csv',
               dir + '/predictions_lelu.csv',
               dir + '/predictions_belu.csv'], 
               ['Tanh',
                'ReLU',
                'ELU',
                'GELU',
                'Sigmoid',
                'Leaky ReLU',
                'SiLU',
                'Softplus',
                'LELU',
                'BELU'],
                study_type)
    plot_loss([dir + '/loss_history_tanh.csv',
               dir + '/loss_history_relu.csv',
               dir + '/loss_history_elu.csv',
               dir + '/loss_history_gelu.csv',
               dir + '/loss_history_sigmoid.csv',
               dir + '/loss_history_lrelu.csv',
               dir + '/loss_history_silu.csv',
               dir + '/loss_history_softplus.csv',
               dir + '/loss_history_lelu.csv',
                dir + '/loss_history_belu.csv'], 
              ['Tanh',
                'ReLU',
                'ELU',
                'GELU',
                'Sigmoid',
                'Leaky ReLU',
                'SiLU',
                'Softplus',
                'LELU',
                'BELU'],
                study_type)
    plt.figure(figsize=(8,5))
    plt.bar(activations, loss_collect, color='k')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(study_type)
    plt.tight_layout()
    # plt.title('Final Loss for Different Activation Functions')
    plt.savefig('./PICS/' + study_type +'/Loss_Bar_Chart.pdf')

    plt.cla()
    plt.close()
    plt.figure(figsize=(8,5))
    plt.bar(activations, std_deviation_collect, color='k')
    plt.ylabel('Standard Deviation')
    plt.yscale('log')
    # plt.title('Standard Deviation for Different Activation Functions')
    plt.title(study_type)
    plt.tight_layout()
    plt.savefig('./PICS/' + study_type +'/Stddev_Bar_Chart.pdf')