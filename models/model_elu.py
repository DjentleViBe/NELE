# model.py
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        layers = []
        input_size = 1
        hidden_neurons = 240
        hidden_layers = 7
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_neurons))
        layers.append(nn.ELU())
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.ELU())
        
        # Output layer (linear)
        layers.append(nn.Linear(hidden_neurons, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)