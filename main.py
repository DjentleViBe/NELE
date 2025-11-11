import torch
from non_linear_tanh import tanh_net
from non_linear_relu import relu_net
from non_linear_elu import elu_net
from non_linear_gelu import gelu_net
from non_linear_sigmoid import sigmoid_net
from non_linear_lrelu import lrelu_net
from non_linear_silu import silu_net
from non_linear_softplus import softplus_net
from plot_pred import plot_pred
from plot_loss import plot_loss
from matplotlib import pyplot as plt
import numpy as np
if __name__ == "__main__":

    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    y = torch.sin(x) + 0.2 * torch.randn(x.size())
    
    activations =  ['Tanh', 'ReLU', 'ELU', 'GELU', 'Sigmoid', 'Leaky ReLU', 'SiLU', 'Softplus']
    loss_collect = np.zeros(len(activations))
    std_deviation_collect = np.zeros(len(activations))

    print("Running analysis - Tanh Activation Function")
    loss_collect[0], std_deviation_collect[0] = tanh_net(x, y, 'RESULTS/predictions_tanh.csv', 'RESULTS/loss_history_tanh.csv')
    print("Running analysis - ReLU Activation Function")
    loss_collect[1], std_deviation_collect[1] = relu_net(x, y, 'RESULTS/predictions_relu.csv', 'RESULTS/loss_history_relu.csv')
    print("Running analysis - ELU Activation Function")
    loss_collect[2], std_deviation_collect[2] = elu_net(x, y, 'RESULTS/predictions_elu.csv', 'RESULTS/loss_history_elu.csv')
    print("Running analysis - GELU Activation Function")
    loss_collect[3], std_deviation_collect[3] = gelu_net(x, y, 'RESULTS/predictions_gelu.csv', 'RESULTS/loss_history_gelu.csv')
    print("Running analysis - Sigmoid Activation Function")
    loss_collect[4], std_deviation_collect[4] = sigmoid_net(x, y, 'RESULTS/predictions_sigmoid.csv', 'RESULTS/loss_history_sigmoid.csv')
    print("Running analysis - Leaky ReLU Activation Function")
    loss_collect[5], std_deviation_collect[5] = lrelu_net(x, y, 'RESULTS/predictions_lrelu.csv', 'RESULTS/loss_history_lrelu.csv')
    print("Running analysis - SiLU Activation Function")
    loss_collect[6], std_deviation_collect[6] = silu_net(x, y, 'RESULTS/predictions_silu.csv', 'RESULTS/loss_history_silu.csv')
    print("Running analysis - softplus Activation Function")
    loss_collect[7], std_deviation_collect[7] = softplus_net(x, y, 'RESULTS/predictions_softplus.csv', 'RESULTS/loss_history_softplus.csv')
    print("Plotting results")
    plot_pred(['RESULTS/predictions_tanh.csv',
               'RESULTS/predictions_relu.csv',
               'RESULTS/predictions_elu.csv',
               'RESULTS/predictions_gelu.csv',
               'RESULTS/predictions_sigmoid.csv',
               'RESULTS/predictions_lrelu.csv',
               'RESULTS/predictions_silu.csv',
               'RESULTS/predictions_softplus.csv'], 
               ['Tanh',
                'ReLU',
                'ELU',
                'GELU',
                'Sigmoid',
                'Leaky ReLU',
                'SiLU',
                'Softplus'])
    plot_loss(['RESULTS/loss_history_tanh.csv',
               'RESULTS/loss_history_relu.csv',
               'RESULTS/loss_history_elu.csv',
               'RESULTS/loss_history_gelu.csv',
               'RESULTS/loss_history_sigmoid.csv',
               'RESULTS/loss_history_lrelu.csv',
               'RESULTS/loss_history_silu.csv',
               'RESULTS/loss_history_softplus.csv'], 
              ['Tanh',
                'ReLU',
                'ELU',
                'GELU',
                'Sigmoid',
                'Leaky ReLU',
                'SiLU',
                'Softplus'])
    plt.figure(figsize=(8,5))
    plt.bar(activations, loss_collect, color='k')
    plt.ylabel('Loss')
    plt.title('Final Loss for Different Activation Functions')
    plt.savefig('./PICS/Loss_Bar_Chart.pdf')

    plt.cla()
    plt.close()
    plt.figure(figsize=(8,5))
    plt.bar(activations, std_deviation_collect, color='k')
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation for Different Activation Functions')
    plt.savefig('./PICS/Stddev_Bar_Chart.pdf')
