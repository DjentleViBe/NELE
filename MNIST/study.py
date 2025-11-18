import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from MNIST.Neuralnet import DeepFCNet
from file_operations import create_directory
from csv_operations import csv_write
import numpy as np
from models.model_lelu import LELU
from models.model_nele import NELE
# Hyperparameters
input_size = 28 * 28
hidden_size = 128
num_hidden_layers = 8
num_classes = 10
batch_size = 64
epochs = 20

def mnist_data(epochs, learn_rate, activation_type='default'):
    activations =  ['Tanh', 'ReLU', 'ELU', 'GELU', 'Sigmoid', 'Leaky ReLU', 'SiLU', 'Softplus', 'LELU', 'BELU', 'Mish', 'NELE']
    colors = ["#1f77b4", "#aec7e8", 
            "#ff7f0e", "#ffbb78",
            "#2ca02c", "#98df8a",
            "#9467bd", "#c5b0d5",
            "#8c564b", "#c49c94",
            '#000000']
    loss_collect = np.zeros(len(activations))
    std_deviation_collect = np.zeros(len(activations))
    dir = 'RESULTS/MNIST/' + activation_type + '/'
    create_directory('RESULTS/MNIST/' + activation_type + '/')
    create_directory('PICS/MNIST/' + activation_type + '/')
    # Activation function selection
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
        activation = NELE(1, 3, 2)
    else:
        raise ValueError("Invalid activation type")

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = DeepFCNet(input_size, hidden_size, num_hidden_layers, num_classes, activation, activation_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learn_rate)

    loss_collect = []
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * data.size(0)
        epoch_loss /= len(train_loader.dataset)
        loss_collect.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch_loss': epoch_loss
    }, dir + activation_type + '_' + str(epoch) + '.pth')
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    loss_collect = torch.tensor(loss_collect)
    csv_write(dir + '/loss_history_' + activation_type + '.csv', torch.linspace(1, epochs, epochs), loss_collect,  'epoch', 'loss', '', torch.linspace(1, epochs, epochs))
    