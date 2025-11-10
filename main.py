import torch
from non_linear_tanh import tanh_net
from non_linear_relu import relu_net
from non_linear_elu import elu_net
from non_linear_gelu import gelu_net
from plot_pred import plot_pred
from plot_loss import plot_loss

if __name__ == "__main__":
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    y = torch.sin(x) + 0.2 * torch.randn(x.size())
    print("Running analysis - Tanh Activation Function")
    tanh_net(x, y)

    print("Running analysis - ReLU Activation Function")
    relu_net(x, y)

    print("Running analysis - ELU Activation Function")
    elu_net(x, y)

    print("Running analysis - GELU Activation Function")
    gelu_net(x, y)

    print("Plotting results")
    plot_pred(['RESULTS/predictions_tanh.csv',
               'RESULTS/predictions_relu.csv',
               'RESULTS/predictions_elu.csv',
               'RESULTS/predictions_gelu.csv'], 
               ['Tanh',
                'ReLU',
                'ELU',
                'GELU'])
    plot_loss(['RESULTS/loss_history_tanh.csv',
               'RESULTS/loss_history_relu.csv',
               'RESULTS/loss_history_elu.csv',
               'RESULTS/loss_history_gelu.csv'], 
              ['Tanh',
                'ReLU',
                'ELU',
                'GELU'])