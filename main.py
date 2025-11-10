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

if __name__ == "__main__":
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    y = torch.sin(x) + 0.2 * torch.randn(x.size())
    print("Running analysis - Tanh Activation Function")
    tanh_net(x, y, 'RESULTS/predictions_tanh.csv', 'RESULTS/loss_history_tanh.csv')
    print("Running analysis - ReLU Activation Function")
    relu_net(x, y, 'RESULTS/predictions_relu.csv', 'RESULTS/loss_history_relu.csv')
    print("Running analysis - ELU Activation Function")
    elu_net(x, y, 'RESULTS/predictions_elu.csv', 'RESULTS/loss_history_elu.csv')
    print("Running analysis - GELU Activation Function")
    gelu_net(x, y, 'RESULTS/predictions_gelu.csv', 'RESULTS/loss_history_gelu.csv')
    print("Running analysis - Signoid Activation Function")
    sigmoid_net(x, y, 'RESULTS/predictions_sigmoid.csv', 'RESULTS/loss_history_sigmoid.csv')
    print("Running analysis - Leaky ReLU Activation Function")
    lrelu_net(x, y, 'RESULTS/predictions_lrelu.csv', 'RESULTS/loss_history_lrelu.csv')
    print("Running analysis - SiLU Activation Function")
    silu_net(x, y, 'RESULTS/predictions_silu.csv', 'RESULTS/loss_history_silu.csv')
    print("Running analysis - softplus Activation Function")
    softplus_net(x, y, 'RESULTS/predictions_softplus.csv', 'RESULTS/loss_history_softplus.csv')
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