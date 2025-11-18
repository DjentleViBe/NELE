import numpy as np
import matplotlib.pyplot as plt
from tanh import tanh_process
from sigmoid import sigmoid_process
from softplus import softplus_process
from ELU import elu_process
from SILU import silu_process
from GELU import gelu_process
from lelu import lelu_process
from Mish import mish_process
from RELU import relu_process
from LRELU import lrelu_process
import matplotlib as mpl
x_points = np.linspace(-5, 5, 200)
colors = ["#0000FF", "#aec7e8", 
            "#ff0000", "#ffbb78",
            "#2ca02c", "#98df8a",
            "#8000FF", "#d57dc1",
            "#99342f", "#c7c7c7",
            '#000000']

activations =  ['Tanh', 'Sigmoid', 'Softplus', 'ELU', 'SiLU', 'GELU', 'ReLU', 'Leaky ReLU', 'LeLU', 'Mish']
y_points_collect = np.zeros((len(activations), 200))
kappa_points_collect = np.zeros((len(activations), 200))
x_max_collect = np.zeros((len(activations), 200))
y_max_collect = np.zeros((len(activations), 200))

y_points_collect[0], kappa_points_collect[0], x_max_collect[0], y_max_collect[0] = tanh_process(x_points)
y_points_collect[1], kappa_points_collect[1], x_max_collect[1], y_max_collect[1] = sigmoid_process(x_points)
y_points_collect[2], kappa_points_collect[2], x_max_collect[2], y_max_collect[2] = softplus_process(x_points)
y_points_collect[3], kappa_points_collect[3], x_max_collect[3], y_max_collect[3] = elu_process(x_points)
y_points_collect[4], kappa_points_collect[4], x_max_collect[4], y_max_collect[4] = silu_process(x_points)
y_points_collect[5], kappa_points_collect[5], x_max_collect[5], y_max_collect[5] = gelu_process(x_points)
y_points_collect[6], kappa_points_collect[6], x_max_collect[6], y_max_collect[6] = relu_process(x_points)
y_points_collect[7], kappa_points_collect[7], x_max_collect[7], y_max_collect[7] = lrelu_process(x_points)
y_points_collect[8], kappa_points_collect[8], x_max_collect[8], y_max_collect[8] = lelu_process(x_points)
y_points_collect[9], kappa_points_collect[9], x_max_collect[9], y_max_collect[9] = mish_process(x_points)
plt.close()
mpl.rcParams['pdf.use14corefonts'] = False
mpl.rcParams['pdf.fonttype'] = 42  # keeps colors in RGB
plt.style.use("tableau-colorblind10")
fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

for l in range(0, len(activations)):
    ax1.plot(x_points, y_points_collect[l], color = colors[l], linewidth = 0.7)

for l in range(0, len(activations)):
    ax1.scatter(x_max_collect[l], y_max_collect[l], label=activations[l],color = colors[l], s = 6, zorder=5, edgecolors='black', linewidth=0.03)

for l in range(0, len(activations)):
    ax3.plot(x_points, kappa_points_collect[l], color = colors[l], label=activations[l], linewidth = 0.7, linestyle = 'dashed')

ax3.set_xlabel('x')
ax1.set_ylabel('y')
ax3.set_ylabel(r'$\kappa$')
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.grid(True, linewidth=0.3)
ax3.grid(True, linewidth=0.3)
ax3.set_xlim(-5, 5)
ax3.locator_params(axis='x', nbins=10)
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig('./analysis/analysis.pdf')

