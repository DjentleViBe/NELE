import torch
import torch.nn as nn
import torch.optim as optim
from csv_operations import csv_write
from models.model_belu import Net

# Generate nonlinear data: y = sin(x) + noise
torch.manual_seed(0)

def belu_net(x, y, pred_file, loss_file, bezier_points, learn_rate, epochs=2000):
    model_belu = Net(x.shape[1], bezier_points)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_belu.parameters(), lr=learn_rate)
    x_vals = torch.linspace(-1, 1, bezier_points)
    y_vals = torch.where(x_vals >= 0, x_vals, 0.01 * x_vals)
    with torch.no_grad():
        model_belu.net[1].control_points[:,0] = x_vals  # x-coords
        model_belu.net[1].control_points[:,1] = y_vals  # y-coords# Training loop
    loss_collect = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model_belu(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_collect.append(loss.item())
        if (epoch+1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # Evaluate model
    model_belu.eval()
    predicted = model_belu(x).detach()
    loss_collect = torch.tensor(loss_collect)

    # Write to CSV
    csv_write(pred_file, x, predicted, 'x' , 'y_pred')
    csv_write(loss_file, torch.linspace(1, epochs, epochs), loss_collect, 'epoch', 'loss')
    sigma_est = torch.std(y - predicted)
    return loss.item(), sigma_est.item()
