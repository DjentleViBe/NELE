import numpy as np
import matplotlib.pyplot as plt

def ELU(x, alpha=1):
    # Vectorized ELU
    return np.where(x >= 0, x, alpha*(np.exp(x) - 1))

def kappa(x, alpha=1):
    # Vectorized curvature
    return np.where(
        x >= 0,
        0,
        np.abs(alpha*np.exp(x)) / (1 + alpha**2 * np.exp(2*x))**1.5
    )

x_points = np.linspace(-5, 5, 200)
y_points = []
kappa_points = []

alpha = 1
x_points = np.linspace(-5, 5, 200)
y_points = ELU(x_points)
kappa_points = kappa(x_points)

x_kmax = 0.5*np.log(1 / (2*alpha**2))
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x_points, y_points, color = 'k', label='y')
ax2.plot(x_points, kappa_points, color = 'k', label=r'$\kappa$', linestyle = '--')
ax1.scatter(x_kmax, ELU(x_kmax, alpha), color ='k', label=r'$\kappa_{\text{max}}$')
ax1.legend()
plt.suptitle('ELU', fontweight='bold')
plt.xlabel('x')
ax1.set_ylabel('y')
ax2.set_ylabel(r'$\kappa$')
plt.legend()
plt.tight_layout()
plt.savefig('./analysis/ELU.pdf')
