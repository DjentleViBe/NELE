"""
Contains the neural network architecture for MNIST
"""
from torch import nn

class DeepAutoencoder(nn.Module):
    """
    Fully connected deep neural network
    """
    def __init__(self, activation):
        super().__init__()
        self.act = activation
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 1000),
            activation,
            nn.Linear(1000, 500),
            activation,
            nn.Linear(500, 250),
            activation,
            nn.Linear(250, 30)  # latent space
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(30, 250),
            activation,
            nn.Linear(250, 500),
            activation,
            nn.Linear(500, 1000),
            activation,
            nn.Linear(1000, 28*28),
            nn.Sigmoid()  # because MNIST pixels are in [0,1]
        )

    def forward(self, x):
        """
        Docstring for forward
        """
        x = x.view(x.size(0), -1)  # flatten input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(x.size(0), 1, 28, 28)  # reshape to image
        return decoded
