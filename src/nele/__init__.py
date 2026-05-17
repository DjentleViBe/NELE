"""
Nele is a Python package that provides a custom activation function for 
neural networks, designed to enhance performance and convergence. 
The package includes the implementation of the Nele activation function, 
which can be easily integrated into existing neural network 
architectures."""
from .activation import Nele

# This makes it available directly when importing the package
__all__ = ["Nele"]
