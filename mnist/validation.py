# pylint:disable=too-many-locals
"""
Docstring for mnist.validation
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import input_size, hidden_size, num_hidden_layers, num_classes, batch_size
from mnist.neuralnet import DeepFCNet
from mnist.utils import get_activation

def mnist_validation(directory, epochs, device, noise_level, activation_type='default', config=None):
    """
    Validation for MNIST
    
    :param epochs: total epochs
    :param device: device name
    :param noise_level: noise level
    :param activation_type: AF
    """
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, \
                                transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, \
                             shuffle=False)
    activation = get_activation(activation_type.lower() + '=1', directory, config, device)
    model = DeepFCNet(input_size, hidden_size, num_hidden_layers, \
                        num_classes, activation, activation_type).to(device)
    test_collect = []
    for i in range(1, 8):
        torch.manual_seed(1234)
        checkpoint = torch.load('RESULTS/MNIST/' + \
                            activation_type + '=' + str(i) \
                            + '/' + activation_type + '=' \
                            + str(i) + '_' + str(epochs - 1) + '.pth',
                            map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # test_loss = checkpoint['test_loss']
        model.eval()
        correct = 0.0
        total = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                noise = torch.empty_like(data).uniform_(-noise_level, noise_level)
                x_noisy = data + noise
                outputs = model(x_noisy)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        test_acc = 100 * correct / total
        test_collect.append(test_acc)
    test_collect = np.asarray(test_collect)
    test_acc_med = np.median(test_collect)
    print(f'{activation_type.split('=')[0]}, Test Accuracy: {test_acc_med:.2f}%')
    return test_acc_med
