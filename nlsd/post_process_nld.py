# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
"""
Post process for NLSD study
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import config as cfg
from csv_operations import csv_read
from nlsd.smoothness import curvature_smoothness
from file_operations import reset_directory, create_directory

def plot_only(x, study_type):
    """
    Docstring for plot_only
    
    :param x: X values
    :param study_type: Analysis type
    """
    loss_collect = np.zeros(len(cfg.AF_NLSD))
    std_deviation_collect = np.zeros(len(cfg.AF_NLSD))
    for i, act in enumerate(cfg.AF_NLSD):
        create_directory('./PICS/NLSD/' + study_type + '/' + act)
        predicted_collect = []
        actual_collect = []
        losses_collect = []
        for j in range(1, 8):
            directory = 'RESULTS/NLSD/' + study_type + '/' + act + '=' + str(j) + '/'
            y, predicted, y_actual = csv_read(directory + 'predictions_' + act + \
                                              '=' + str(j) + '.csv', 'x', 'y_pred', 'y_actual')
            _, loss, _ = csv_read(directory + 'loss_history_' + act + '=' + str(j) \
                                  + '.csv', 'epoch', 'loss', '')
            losses_collect.append(loss)
            predicted_collect.append(predicted)
            actual_collect.append(y_actual)

        loss_c = np.median(losses_collect, axis =0)
        y_actual = np.median(actual_collect, axis=0)
        predicted = np.median(predicted_collect, axis = 0)
        y_actual = torch.tensor(y_actual)
        predicted = torch.tensor(predicted)
        sigma_est = torch.std(y_actual - predicted)

        loss_collect[i] = loss_c[-1]
        std_deviation_collect[i] = sigma_est.item()
    mpl.rcParams['pdf.use14corefonts'] = False
    mpl.rcParams['pdf.fonttype'] = 42  # keeps colors in RGB
    plt.style.use("tableau-colorblind10")
    plt.figure(figsize=(5,3))
    plt.bar(cfg.AF_NLSD_PLOT, std_deviation_collect, color=cfg.colors)
    plt.errorbar(cfg.AF_NLSD_PLOT, std_deviation_collect, yerr=loss_collect, \
                 fmt='none', ecolor="black", elinewidth=3, capsize=5)
    plt.errorbar(cfg.AF_NLSD_PLOT, std_deviation_collect, yerr=loss_collect, \
                 fmt='none', ecolor="white", elinewidth=0, capsize=3)
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./PICS/NLSD/' + study_type + \
                '/Loss_Bar_Chart_with_Error_Bars.pdf', transparent=False)
    plt.cla()
    plt.close()

    plt.figure(figsize=(5,3))
    plt.subplots_adjust(right = 0.65)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    for i, act in enumerate(cfg.AF_NLSD):
        losses_collect = []
        for j in range(1, 8):
            directory = 'RESULTS/NLSD/' + study_type + '/' + act + '=' + str(j) + '/'
            epochs, losses, _ = csv_read(directory + '/loss_history_' + act + '=' + str(j) \
                                         + '.csv', 'epoch', 'loss', '')
            losses_collect.append(losses)
        losses = np.median(losses_collect, axis=0)
        plt.plot(epochs, losses, cfg.colors[i], label=cfg.AF_NLSD_PLOT[i], linewidth = 0.8)
    # Plot
    plt.yscale('log')
    plt.legend(loc = 'lower right', bbox_to_anchor = (1.6, -0.04))
    plt.grid(True, linewidth=0.1)
    plt.tight_layout()
    plt.savefig('PICS/NLSD/' + study_type + '/training_loss.pdf', transparent=False)

    plt.cla()
    plt.close()

    plt.figure(figsize=(5,3))
    plt.subplots_adjust(right = 0.65)
    s3_collect = []
    for i, act in enumerate(cfg.AF_NLSD):
        y_preds_collect= []
        for j in range(1, 7):
            directory = 'RESULTS/NLSD/' + study_type + '/' + act + '=' + str(j) + '/'
            x_vals, y_preds, y = csv_read(directory + 'predictions_' + act + '=' + str(j) \
                                          +'.csv', 'x', 'y_pred','y_actual')
            y_preds_collect.append(y_preds)

        y_preds = np.median(y_preds_collect, axis = 0)
        plt.plot(x_vals, y_preds, cfg.colors[i], label=cfg.AF_NLSD_PLOT[i], linewidth=0.7)
        # s1 = smoothness_derivative_energy(x_vals, y_preds, 1)
        # s2 = smoothness_derivative_energy(x_vals, y_preds, 2)
        s3 = curvature_smoothness(x_vals, y_preds)
        # s4 = lipschitz_constant(x_vals, y_preds)
        # s5 = frequency_smoothness(y_preds)
        s3_collect.append(s3)
        # print(f'{act} & {round(s3, 2)} \\\\')
    # print("\n")
    # Plot
    plt.scatter(x, y, label='Data', color = 'k', s=10)
    # plt.scatter(x_vals, y_preds, s=10, alpha=0.5)  # optional: scatter for points
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc = 'lower right', bbox_to_anchor = (1.48, -0.13))
    plt.tight_layout()
    plt.savefig('PICS/NLSD/' + study_type + '/curve_fitting.pdf', transparent=False)
    return s3_collect

def process_nld(reset):
    """
    Docstring for process_nld
    
    :param reset: reset==1 deletes the folder contents
    """
    s3_print = np.zeros((len(cfg.FUNC_NLSD_PLOT), len(cfg.AF_NLSD)))
    if reset == 1:
        reset_directory('./PICS/NLSD/')
        reset_directory('./PICS/NLSD/')
    else:
        create_directory('./PICS/NLSD/')
        create_directory('./PICS/NLSD/')
    x = torch.linspace(-5, 5, 200).unsqueeze(1)
    ################### EXP NOISE ##########################
    create_directory('./PICS/NLSD/' + 'exp')
    s3_print[0] = plot_only(x, 'exp')

    ################### HYP NOISE ##########################
    create_directory('./PICS/NLSD/' + 'hyp')
    s3_print[1] =  plot_only(x, 'hyp')

    ################### QUAD NOISE ##########################
    create_directory('./PICS/NLSD/' + 'quad')
    s3_print[2] =  plot_only(x, 'quad')

    ################### SINE NOISE ##########################
    create_directory('./PICS/NLSD/' + 'sine')
    s3_print[3] = plot_only(x, 'sine')

    ################### TRIG NOISE ##########################
    create_directory('./PICS/NLSD/' + 'trig')
    s3_print[4] = plot_only(x, 'trig')

    ################### EXP-POLY NOISE ##########################
    x = torch.linspace(0, 10, 200).unsqueeze(1)
    create_directory('./PICS/NLSD/' + 'exppoly')
    s3_print[5] = plot_only(x, 'exppoly')

    for j in range(len(cfg.AF_NLSD)):
        print(f'\n\\texttt{{{cfg.AF_NLSD_PLOT[j]}}} & ', end='')
        for i in range(len(cfg.FUNC_NLSD_PLOT)):
            if i == len(cfg.FUNC_NLSD_PLOT) - 1:
                print(f'{round(s3_print[i][j], 2)}', end=' \\\\ ')
            else:
                print(f'{round(s3_print[i][j], 2)}', end=' & ')
    print('\n')
