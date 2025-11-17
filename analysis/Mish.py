import numpy as np
import matplotlib.pyplot as plt

def Mish(x, a=0.44715):
    # Vectorized ELU
    return x*(np.tanh(np.log(1 + np.exp(x))))

def sech(x):
    return 1 / np.cosh(x)

def kappa(x):
    e = 1 + np.exp(x)
    e_minus = 1 + np.exp(-x)
    u = np.log(e)
    u_1 = x * np.exp(x) / e
    y_1 = np.tanh(u) + u_1 * sech(u)**2
    y_2 = u_1 * sech(u)**2 - (2 * sech(u)**2 * np.tanh(u) * u_1) \
    + sech(u)**2 * np.exp(-x) / e_minus**2\
    + sech(u)**2 * np.exp(x)
    kappa = np.abs(y_2) / (1 + y_1**2)**1.5
    return kappa

def mish_process(x_points):
    y_points = Mish(x_points)
    kappa_points = kappa(x_points)
    kappa_max = np.max(kappa_points)
    x_max = x_points[np.argmax(kappa_points)]
    y_max = Mish(x_max)
    print(f'k_max={kappa_max}, x_max = {x_max}, y_max={y_max}')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x_points, y_points, color = 'k', label='y')
    ax2.plot(x_points, kappa_points, color = 'k', label=r'$\kappa$', linestyle = '--')
    ax1.scatter(x_max, y_max, color ='k', label=r'$\kappa_{\text{max}}$')
    ax1.legend()
    plt.suptitle('Mish', fontweight='bold')
    plt.xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_ylabel(r'$\kappa$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./analysis/Mish.pdf')
    return y_points, kappa_points, x_max, y_max


