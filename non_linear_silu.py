import torch
import torch.nn as nn
import torch.optim as optim
from csv_operations import csv_write
from models.model_silu import Net

# Generate nonlinear data: y = sin(x) + noise
torch.manual_seed(0)

def silu_net(x, y, pred_file, loss_file):
    model_silu = Net(x.shape[1])

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_silu.parameters(), lr=0.01)

    # Training loop
    epochs = 2000
    loss_collect = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model_silu(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_collect.append(loss.item())
        if (epoch+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluate model
    model_silu.eval()
    predicted = model_silu(x).detach()
    loss_collect = torch.tensor(loss_collect)

    # Write to CSV
    csv_write(pred_file, x, predicted, 'x' , 'y_pred')
    csv_write(loss_file, torch.linspace(1, epochs, epochs), loss_collect, 'epoch', 'loss')

    return loss.item()
