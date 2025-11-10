import matplotlib.pyplot as plt
from csv_operations import csv_read

epochs, losses_tanh = csv_read('RESULTS/loss_history_tanh.csv', 'epoch', 'loss')

# Plot
plt.figure(figsize=(8,5))
plt.plot(epochs, losses_tanh, 'k', label='Tanh')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()