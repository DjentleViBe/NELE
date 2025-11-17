import numpy as np
import matplotlib.pyplot as plt

def GELU(x, a=0.44715):
    # Vectorized ELU
    return 0.5*x*(1 + np.tanh(np.sqrt(2/np.pi)*(x + a*x**3)))

def sech(x):
    return 1 / np.cosh(x)

def kappa(x):
    k = np.sqrt(2/np.pi)
    a = 0.441715
    u = k*(x + a*x**3)
    u_1 = k*(1+3*a*x**2)
    u_2 = k*6*a*x
    y_1 = 0.5 * x * sech(u)**2 * u_1 + 0.5 * (1 + np.tanh(u))
    y_2 = 0.5 * sech(u)**2 * u_1 + \
    sech(u)**2 * u_1 + \
    x * sech(u)**2 * u_2 + \
    x * u_1 * (-2 * sech(u)**2 * np.tanh(u) * u_1)
    kappa = np.abs(y_2) / (1 + y_1**2)**1.5
    return kappa

def gelu_process(x_points):
    a = 0.44715
    y_points = GELU(x_points, a)
    kappa_points = kappa(x_points)
    kappa_max = np.max(kappa_points)
    x_max = x_points[np.argmax(kappa_points)]
    y_max = GELU(x_max, a)
    print(f'k_max={kappa_max}, x_max = {x_max}, y_max={y_max}')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x_points, y_points, color = 'k', label='y')
    ax2.plot(x_points, kappa_points, color = 'k', label=r'$\kappa$', linestyle = '--')
    ax1.scatter(x_max, y_max, color ='k', label=r'$\kappa_{\text{max}}$')
    ax1.legend()
    plt.suptitle('GELU', fontweight='bold')
    plt.xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_ylabel(r'$\kappa$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./analysis/GELU.pdf')

    return y_points, kappa_points, x_max, y_max


