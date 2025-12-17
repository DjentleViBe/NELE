"""
Contains the neural network architecture for MNIST
"""
import torch.nn as nn
# Fully connected deep neural network
class DeepFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, num_classes, activation, activation_type):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        layers.append(nn.Linear(hidden_size, num_classes))
        self.net = nn.Sequential(*layers)

        # He initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                if activation_type in ['relu', 'leaky_relu']:
                    nn.init.kaiming_normal_(m.weight, nonlinearity=activation_type)
                else:
                    nn.init.kaiming_normal_(m.weight) 
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)