import torch.nn as nn
import torch
from MNIST.study import input_size, hidden_size, num_hidden_layers, num_classes, batch_size
from MNIST.Neuralnet import DeepFCNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model_lelu import LELU
from models.model_nele import NELE

def mnist_validation(activation_type='default'):
    if activation_type == 'relu':
        activation = nn.ReLU()
    elif activation_type == 'elu':
        activation = nn.ELU(alpha=1.0)
    elif activation_type == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.1)
    elif activation_type == 'gelu':
        activation = nn.GELU()
    elif activation_type == 'mish':
        activation = nn.Mish()
    elif activation_type == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation_type == 'silu':
        activation = nn.SiLU()
    elif activation_type == 'softplus':
        activation = nn.Softmax()
    elif activation_type == 'tanh':
        activation = nn.Tanh()
    elif activation_type == 'lelu':
        activation = LELU()
    elif activation_type == 'nele':
        activation = NELE(1, 4, 3)
    else:
        raise ValueError("Invalid activation type")
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = DeepFCNet(input_size, hidden_size, num_hidden_layers, num_classes, activation, activation_type)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'{activation_type}, Test Accuracy: {100 * correct / total:.2f}%')
