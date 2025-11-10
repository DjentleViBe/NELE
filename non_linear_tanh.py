import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from csv_operations import csv_write
from models.model_tanh import Net
# Generate nonlinear data: y = sin(x) + noise
torch.manual_seed(0)
x = torch.linspace(-3, 3, 200).unsqueeze(1)
y = torch.sin(x) + 0.2 * torch.randn(x.size())

model_tanh = Net()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model_tanh.parameters(), lr=0.01)

# Training loop
epochs = 2000
loss_collect = []
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model_tanh(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    loss_collect.append(loss.item())
    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluate model
model_tanh.eval()
predicted = model_tanh(x).detach()
loss_collect = torch.tensor(loss_collect)

# Write to CSV
csv_write('RESULTS/predictions_tanh.csv', x, predicted, 'x' , 'y_pred')
csv_write('RESULTS/loss_history_tanh.csv', torch.linspace(1, epochs, epochs), loss_collect, 'epoch', 'loss')
