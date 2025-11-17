import torch
from study import study_data
from file_operations import reset_directory

if __name__ == "__main__":
    '''
    x = torch.linspace(-5, 5, 200).unsqueeze(1)
    reset_directory('./RESULTS/')
    reset_directory('./PICS/')
    
    ################### SINE NOISE ##########################
    reset_directory('./PICS/sine_noise')
    reset_directory('./RESULTS/sine_noise')
    
    y = torch.sin(x) + 0.2 * torch.randn(x.size())
    study_data(x, y, 1000, 0.0005, study_type='sine_noise')

    ################### TRIG NOISE ##########################
    reset_directory('./PICS/trig_noise')
    reset_directory('./RESULTS/trig_noise')
    x = torch.linspace(-5, 5, 200).unsqueeze(1)
    y = torch.sin(x) + 0.5 * torch.sin(3*x) + 0.2 * torch.randn(x.size())
    study_data(x, y, 10000, 0.0001, study_type='trig_noise')
    
    ################### EXP NOISE ##########################
    reset_directory('./PICS/exp_noise')
    reset_directory('./RESULTS/exp_noise')
    
    y = torch.exp(-0.5*x) + 0.2 * torch.randn(x.size())
    study_data(x, y, 10000, 0.0001, study_type='exp_noise')
    
    ################### HYP NOISE ##########################
    reset_directory('./PICS/hyp_noise')
    reset_directory('./RESULTS/hyp_noise')
    
    y = torch.tanh(x) + 0.2 * torch.randn(x.size())
    study_data(x, y, 1000, 0.001, study_type='hyp_noise')
    
    ################### QUAD NOISE ##########################
    reset_directory('./PICS/quad_noise')
    reset_directory('./RESULTS/quad_noise')

    y = torch.tensor(x**2) + 0.2 * torch.randn(x.size())
    study_data(x, y, 2000, 0.001, study_type='quad_noise')

    ################### EXP-POLY NOISE ##########################
    reset_directory('./PICS/exppoly_noise')
    reset_directory('./RESULTS/exppoly_noise')
    x = torch.linspace(0, 10, 200).unsqueeze(1)
    y = torch.tensor(x**3) / (torch.exp(x) - 1 + 1E-6) + 0.2 * torch.randn(x.size())
    study_data(x, y, 2000, 0.001, study_type='exppoly_noise')
    '''
    ################### HYP NOISE ##########################
    reset_directory('./PICS/hyp_noise')
    reset_directory('./RESULTS/hyp_noise')
    x = torch.linspace(-5, 5, 200).unsqueeze(1)
    y = torch.tanh(x) + 0.2 * torch.randn(x.size())
    study_data(x, y, 3000, 0.0005, study_type='hyp_noise')

    
    