import numpy as np
import matplotlib.pyplot as plt

def LELU(x, alpha=0.5):
    # Vectorized LELU
    return np.where(
        x > 0,
        x,
        np.exp((1 - alpha)*x) -1 + alpha * x
    )

def kappa(x, alpha=1):
    # Vectorized curvature
    a = 1 - alpha
    y_1 = a*np.exp(a * x) - alpha
    y_2 = a**2 * np.exp(a*x)
    kappa = np.abs(y_2) / (1 + y_1**2)**1.5
    return np.where(
        x > 0,
        0,
        kappa
    )

def lelu_process(x_points):
    y_points = []
    kappa_points = []

    alpha = 0.3

    beta = 1 - 0.5**0.5
    x_kmax = -np.log(2) * np.sqrt(2)/2
    alpha = beta

    x_points = np.linspace(-5, 5, 200)
    y_points = LELU(x_points, alpha)
    kappa_points = kappa(x_points, alpha)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x_points, y_points, color = 'k', label='y')
    ax2.plot(x_points, kappa_points, color = 'k', label=r'$\kappa$', linestyle = '--')

    print(beta)
    y_max = LELU(x_kmax, beta)
    ax1.scatter(x_kmax, y_max, color ='k', label=r'$\kappa_{\text{max}}$')
    ax1.legend()
    plt.suptitle('LELU', fontweight='bold')
    plt.xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_ylabel(r'$\kappa$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./analysis/LELU.pdf')
    return y_points, kappa_points, x_kmax, y_max
