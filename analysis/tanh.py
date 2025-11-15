import numpy as np
import matplotlib.pyplot as plt

def tanh(x, alpha=1):
    # Vectorized ELU
    return np.tanh(x)

def sech(x):
    return 1 / np.cosh(x)

def kappa(x, alpha=1):
    # Vectorized curvature
    y = np.tanh(x)
    num = np.abs(-2 * y * (1 - y**2))
    den = (1 + (1 - y**2)**2)**(1.5)
    y_1 = sech(x)**2
    y_2 = -2*sech(x)**2*np.tanh(x)
    kappa = np.abs(y_2) / (1 + y_1**2)**1.5
    return kappa

x_points = np.linspace(-5, 5, 200)
y_points = []
kappa_points = []

alpha = 1
x_points = np.linspace(-5, 5, 200)
y_points = tanh(x_points)
kappa_points = kappa(x_points)

x_kmax = 0.9196
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
print(kappa(x_kmax, alpha))
ax1.plot(x_points, y_points, color = 'k', label='y')
ax2.plot(x_points, kappa_points, color = 'k', label=r'$\kappa$', linestyle = '--')
ax1.scatter(x_kmax, tanh(x_kmax, alpha), color ='k', label=r'$\kappa_{\text{max}}$')
ax1.legend()
plt.xlabel('x')
ax1.set_ylabel('y')
ax2.set_ylabel(r'$\kappa$')
plt.suptitle('tanh', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('./analysis/tanh.pdf')
