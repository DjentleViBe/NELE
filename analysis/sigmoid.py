import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, a=0.44715):
    # Vectorized ELU
    return 1 / (1 + np.exp(-x))

def sech(x):
    return 1 / np.cosh(x)

def kappa(x):
    e = np.exp(-x)
    y_1 = -e / (1 + e)**2
    y_2 = e*(1 - e)/(1 + e)**3
    kappa = np.abs(y_2) / (1 + y_1**2)**1.5
    return kappa

a = 0.44715
x_points = np.linspace(-5, 5, 200)
y_points = sigmoid(x_points, a)
kappa_points = kappa(x_points)
kappa_max = np.max(kappa_points)
x_max = x_points[np.argmax(kappa_points)]
y_max = sigmoid(x_max, a)
print(f'k_max={kappa_max}, x_max = {x_max}, y_max={y_max}')

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x_points, y_points, color = 'k', label='y')
ax2.plot(x_points, kappa_points, color = 'k', label=r'$\kappa$', linestyle = '--')
ax1.scatter(x_max, y_max, color ='k', label=r'$\kappa_{\text{max}}$')
ax1.legend()
plt.suptitle('Sigmoid', fontweight='bold')
plt.xlabel('x')
ax1.set_ylabel('y')
ax2.set_ylabel(r'$\kappa$')
plt.legend()
plt.tight_layout()
plt.savefig('./analysis/sigmoid.pdf')


