import torch.nn as nn
import torch
from config import input_size, hidden_size, num_hidden_layers, num_classes, batch_size
from MNIST.Neuralnet import DeepFCNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model_lelu import LELU
from models.model_nele import NELE

def mnist_validation(epochs, device, activation_type='default'):
    if '=' in activation_type:
        base, param = activation_type.split('=')
        param = float(param)  # convert parameter to float if needed
    else:
        base = activation_type
        param = None
    if base == 'relu':
        activation = nn.ReLU()
    elif base == 'elu':
        activation = nn.ELU(alpha=1.0)
    elif base == 'leaky_relu' :
        activation = nn.LeakyReLU(negative_slope=0.1)
    elif base == 'gelu' :
        activation = nn.GELU()
    elif base == 'mish' :
        activation = nn.Mish()
    elif base == 'sigmoid' :
        activation = nn.Sigmoid()
    elif base == 'silu' :
        activation = nn.SiLU()
    elif base == 'softplus' :
        activation = nn.Softmax()
    elif base == 'tanh':
        activation = nn.Tanh()
    elif base == 'lelu' :
        activation = LELU()
    elif base == 'nele' :
        activation = NELE(1, 3, 2)
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
    checkpoint = torch.load('RESULTS/MNIST/' + activation_type + '/' + activation_type + '_' + str(epochs - 1) + '.pth',
                            map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    epoch_loss = checkpoint['epoch_loss']
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'{activation_type}, Test Accuracy: {100 * correct / total:.2f}%, Training loss : {round(epoch_loss, 4)}')
