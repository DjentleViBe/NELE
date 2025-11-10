import torch
from non_linear_tanh import tanh_net
from plot_pred import plot_pred
from plot_loss import plot_loss

if __name__ == "__main__":
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    y = torch.sin(x) + 0.2 * torch.randn(x.size())
    print("Running analysis - Tanh Activation Function")
    tanh_net(x, y)

    plot_pred(['RESULTS/predictions_tanh.csv'], ['Tanh'])
    plot_loss(['RESULTS/loss_history_tanh.csv'], ['Tanh'])