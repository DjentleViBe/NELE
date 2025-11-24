import torch
from file_operations import reset_directory, create_directory
from NLSD.study import nlsd_data
import config as cfg

def curve_fit(device, reset):
    x = torch.linspace(-5, 5, 200).unsqueeze(1)
    if reset == 1:
        reset_directory('./RESULTS/NLSD/')
        reset_directory('./PICS/NLSD/')
    else:
        create_directory('./RESULTS/NLSD/')
        create_directory('./PICS/NLSD/')
        create_directory('./PICS/NLSD/sine_noise')
        create_directory('./RESULTS/NLSD/sine_noise')
        create_directory('./PICS/NLSD/trig_noise')
        create_directory('./RESULTS/NLSD/trig_noise')
        create_directory('./PICS/NLSD/exp_noise')
        create_directory('./RESULTS/NLSD/exp_noise')
        create_directory('./PICS/NLSD/hyp_noise')
        create_directory('./RESULTS/NLSD/hyp_noise')
        create_directory('./PICS/NLSD/quad_noise')
        create_directory('./RESULTS/NLSD/quad_noise')
        create_directory('./PICS/NLSD/exppoly_noise')
        create_directory('./RESULTS/NLSD/exppoly_noise')
        create_directory('./PICS/NLSD/trig_noise')
        create_directory('./RESULTS/NLSD/trig_noise')
    loss_collect = []
    std_deviation_collect = []
    ################### SINE NOISE ##########################
    for af in cfg.AF_NLSD:
        for afunc in cfg.FUNC_NLSD:
            print(f'Activation function : {af}, Function : {afunc}')
            if afunc == 'sin':
                y = torch.sin(x) + 0.2 * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='sine_noise')
            elif afunc == 'trig':
                ################### TRIG NOISE ##########################
                x = torch.linspace(-5, 5, 200).unsqueeze(1)
                y = torch.sin(x) + 0.5 * torch.sin(3*x) + 0.2 * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='trig_noise')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'exp':
                ################### EXP NOISE ##########################
                y = torch.exp(-0.5*x) + 0.2 * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='exp_noise')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'hyp':
                ################### HYP NOISE ##########################
                y = torch.tanh(x) + 0.2 * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='hyp_noise')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'quad':
                ################### QUAD NOISE ##########################
                y = x**2 + 0.2 * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='quad_noise')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
            elif afunc == 'exppoly':
                ################### EXP-POLY NOISE ##########################
                
                x = torch.linspace(0, 10, 200).unsqueeze(1)
                y = torch.tensor(x**3) / (torch.exp(x) - 1 + 1E-6) + 0.2 * torch.randn(x.size())
                loss_val, std_val = nlsd_data(x, y, af=af, study_type='exppoly_noise')
                loss_collect.append(loss_val)
                std_deviation_collect.append(std_val)
         
    