import matplotlib.pyplot as plt
import torch
from csv_operations import csv_read

x = torch.linspace(-3, 3, 200).unsqueeze(1)
y = torch.sin(x) + 0.2 * torch.randn(x.size())
# Read the CSV file
x_vals_tanh, y_preds_tanh = csv_read('RESULTS/predictions_tanh.csv', 'x', 'y_pred')

# Plot
plt.figure(figsize=(8,5))
plt.scatter(x, y, label='Data', s=10)
plt.plot(x_vals_tanh, y_preds_tanh, 'r', label='Tanh')
# plt.scatter(x_vals, y_preds, s=10, alpha=0.5)  # optional: scatter for points
plt.xlabel('x')
plt.ylabel('y_pred')
plt.title('Curve fitting')
plt.legend()
plt.show()