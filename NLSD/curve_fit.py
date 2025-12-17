# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
"""
Functions for curve fit analysis
"""
import torch
from file_operations import reset_directory, create_directory
from NLSD.study import nlsd_data
import config as cfg

def curve_fit(device, reset, exec_study):
    """
    Curve fit study
    
    :param device: device name
    :param reset: reset==1 deletes the diretory contents
    :param exec: exec==1 uses cfg.STUDY for af_nlsd (7 runs in total)
    """
    print(f'Device : {device}')
    if reset == 1:
        reset_directory('./RESULTS/NLSD/')
        reset_directory('./PICS/NLSD/')
    else:
        create_directory('./RESULTS/NLSD/')
        create_directory('./PICS/NLSD/')
    loss_collect = []
    std_deviation_collect = []
    af_nlsd = cfg.AF_NLSD
    if exec_study == 1:
        af_nlsd = cfg.STUDY
    ################### SINE NOISE ##########################
    for af in af_nlsd:
        for afunc in cfg.FUNC_NLSD:
            create_directory('./PICS/NLSD/' + afunc + '/' + af)
            create_directory('./RESULTS/NLSD/' + afunc + '/' + af)
            print(f'Activation function : {af}, Function : {afunc}')
            if afunc == 'sine':
                x = torch.linspace(-5, 5, 200).unsqueeze(1)
                y = torch.sin(x) + (1 / cfg.noise_level) * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='sine')
            elif afunc == 'trig':
                ################### TRIG NOISE ##########################
                x = torch.linspace(-5, 5, 200).unsqueeze(1)
                y = torch.sin(x) + 0.5 * torch.sin(3*x) + \
                    (1 / cfg.noise_level) * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='trig')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'exp':
                ################### EXP NOISE ##########################
                x = torch.linspace(-5, 5, 200).unsqueeze(1)
                x_range = x.max() - x.min()
                y = torch.exp(-0.5*x) + (1 / cfg.noise_level) \
                    * 0.5 * x_range * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='exp')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'hyp':
                ################### HYP NOISE ##########################
                x = torch.linspace(-5, 5, 200).unsqueeze(1)
                y = torch.tanh(x) + (1 / cfg.noise_level) * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='hyp')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'quad':
                ################### QUAD NOISE ##########################
                x = torch.linspace(-5, 5, 200).unsqueeze(1)
                x_range = x.max() - x.min()
                y = x**2 + (1 / cfg.noise_level) * x_range * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='quad')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'exppoly':
                ################### EXP-POLY NOISE ##########################
                x = torch.linspace(0, 10, 200).unsqueeze(1)
                y = torch.tensor(x**3) / (torch.exp(x) - 1 + 1E-6) \
                    + (1 / cfg.noise_level) * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='exppoly')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)

def curve_fit_nele(reset):
    """
    Curve fit analysis for NELE
    
    :param reset: reset==1 deletes folder contents
    """
    x = torch.linspace(-5, 5, 200).unsqueeze(1)
    if reset == 1:
        reset_directory('./RESULTS/NLSD/')
        reset_directory('./PICS/NLSD/')
    else:
        create_directory('./RESULTS/NLSD/')
        create_directory('./PICS/NLSD/')
    loss_collect = []
    std_deviation_collect = []
    ################### SINE NOISE ##########################
    for af in cfg.AF_NLSD_NELE:
        for afunc in cfg.FUNC_NLSD_NELE:
            create_directory('./PICS/NLSD/' + afunc + '/' + af)
            create_directory('./RESULTS/NLSD/' + afunc + '/' + af)
            print(f'Activation function : {af}, Function : {afunc}')
            if afunc == 'sine':
                y = torch.sin(x) + (1 / cfg.noise_level) * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='sine')
            elif afunc == 'trig':
                ################### TRIG NOISE ##########################
                x = torch.linspace(-5, 5, 200).unsqueeze(1)
                y = torch.sin(x) + 0.5 * torch.sin(3*x) + \
                    (1 / cfg.noise_level) * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='trig')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'exp':
                ################### EXP NOISE ##########################
                y = torch.exp(-0.5*x) + (1 / cfg.noise_level) * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='exp')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'hyp':
                ################### HYP NOISE ##########################
                y = torch.tanh(x) + (1 / cfg.noise_level) * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='hyp')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'quad':
                ################### QUAD NOISE ##########################
                y = x**2 + (1 / cfg.noise_level) * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='quad')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'exppoly':
                ################### EXP-POLY NOISE ##########################
                x = torch.linspace(0, 10, 200).unsqueeze(1)
                y = torch.tensor(x**3) / (torch.exp(x) - 1 + 1E-6) \
                    + (1 / cfg.noise_level) * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='exppoly')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
