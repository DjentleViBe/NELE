import numpy as np
import matplotlib.pyplot as plt

def softplus(x, alpha=1):
    # Vectorized softplus
    return np.log(1 + np.exp(x))

def kappa(x, alpha=1):
    # Vectorized curvature
    e = np.exp(x)
    y_1 = e / (1 + e)
    y_2 = e /(1 + e)**2
    kappa = np.abs(y_2) / (1 + y_1**2)**1.5
    return kappa

def softplus_process(x_points):
    y_points = []
    kappa_points = []

    alpha = 1
    y_points = softplus(x_points)
    kappa_points = kappa(x_points)

    x_kmax = 0.5*np.log(1 / (2*alpha**2))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    print(kappa(x_kmax))
    y_max = softplus(x_kmax, alpha)
    ax1.plot(x_points, y_points, color = 'k', label='y')
    ax2.plot(x_points, kappa_points, color = 'k', label=r'$\kappa$', linestyle = '--')
    ax1.scatter(x_kmax, y_max, color ='k', label=r'$\kappa_{\text{max}}$')
    ax1.legend()
    plt.suptitle('softplus', fontweight='bold')
    plt.xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_ylabel(r'$\kappa$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./analysis/softplus.pdf')
    return y_points, kappa_points, x_kmax, y_max
