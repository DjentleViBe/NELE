import numpy as np
import matplotlib.pyplot as plt

def SILU(x, alpha=1):
    # Vectorized ELU
    return x / (1 + np.exp(-x))

def kappa(x):
    e = np.exp(-x)
    e2 = np.exp(-2 * x)
    A = 1 + e
    y_1 = (1 + e + x*e)/(A**2)
    y_2 = (e*(2-x) + e2*(2+x)) / A**3
    kappa = np.abs(y_2) / (1 + y_1**2)**1.5
    return kappa

def silu_process(x_points):
    alpha = 1
    y_points = SILU(x_points)
    kappa_points = kappa(x_points)
    kappa_max = np.max(kappa_points)
    x_max = x_points[np.argmax(kappa_points)]
    y_max = SILU(x_max)
    print(f'k_max={kappa_max}, x_max = {x_max}, y_max={y_max}')

    x_kmax = 0.5*np.log(1 / (2*alpha**2))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x_points, y_points, color = 'k', label='y')
    ax2.plot(x_points, kappa_points, color = 'k', label=r'$\kappa$', linestyle = '--')
    ax1.scatter(x_max, y_max, color ='k', label=r'$\kappa_{\text{max}}$')
    ax1.legend()
    plt.suptitle('SiLU', fontweight='bold')
    plt.xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_ylabel(r'$\kappa$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./analysis/SILU.pdf')
    return y_points, kappa_points, x_kmax, y_max


