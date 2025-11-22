import torch
from file_operations import reset_directory, create_directory
from NLSD.study import study_data

def curve_fit(reset):
    x = torch.linspace(-5, 5, 200).unsqueeze(1)
    if reset == 1:
        reset_directory('./RESULTS/')
        reset_directory('./PICS/')
    else:
        create_directory('./RESULTS/')
        create_directory('./PICS/')
        create_directory('./PICS/sine_noise')
        create_directory('./RESULTS/sine_noise')
        create_directory('./PICS/trig_noise')
        create_directory('./RESULTS/trig_noise')
        create_directory('./PICS/exp_noise')
        create_directory('./RESULTS/exp_noise')
        create_directory('./PICS/hyp_noise')
        create_directory('./RESULTS/hyp_noise')
        create_directory('./PICS/quad_noise')
        create_directory('./RESULTS/quad_noise')
        create_directory('./PICS/exppoly_noise')
        create_directory('./RESULTS/exppoly_noise')
        create_directory('./PICS/trig_noise')
        create_directory('./RESULTS/trig_noise')
    
    ################### SINE NOISE ##########################
    
    
    y = torch.sin(x) + 0.2 * torch.randn(x.size())
    study_data(x, y, 1000, 0.0005, study_type='sine_noise')

    ################### TRIG NOISE ##########################
    
    x = torch.linspace(-5, 5, 200).unsqueeze(1)
    y = torch.sin(x) + 0.5 * torch.sin(3*x) + 0.2 * torch.randn(x.size())
    study_data(x, y, 5000, 0.001, study_type='trig_noise')
    
    ################### EXP NOISE ##########################
    
    
    y = torch.exp(-0.5*x) + 0.2 * torch.randn(x.size())
    study_data(x, y, 10000, 0.0001, study_type='exp_noise')
    
    ################### HYP NOISE ##########################
    
    
    y = torch.tanh(x) + 0.2 * torch.randn(x.size())
    study_data(x, y, 1000, 0.001, study_type='hyp_noise')
    
    ################### QUAD NOISE ##########################
    

    y = torch.tensor(x**2) + 0.2 * torch.randn(x.size())
    study_data(x, y, 2000, 0.001, study_type='quad_noise')

    ################### EXP-POLY NOISE ##########################
    
    x = torch.linspace(0, 10, 200).unsqueeze(1)
    y = torch.tensor(x**3) / (torch.exp(x) - 1 + 1E-6) + 0.2 * torch.randn(x.size())
    study_data(x, y, 2000, 0.001, study_type='exppoly_noise')
    
    ################### HYP NOISE ##########################
    
    x = torch.linspace(-5, 5, 200).unsqueeze(1)
    y = torch.sin(x) + 0.5 * torch.sin(3*x) + 0.2 * torch.randn(x.size())
    study_data(x, y, 5000, 0.001, study_type='trig_noise')
    