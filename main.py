import torch
from study import study_data
from file_operations import reset_directory

if __name__ == "__main__":
    reset_directory('./RESULTS/')
    reset_directory('./PICS/')
    reset_directory('./PICS/sine_noise')
    reset_directory('./RESULTS/sine_noise')
    x = torch.linspace(-3, 3, 200).unsqueeze(1)
    y = torch.sin(x) + 0.2 * torch.randn(x.size())
    study_data(x, y, study_type='sine_noise')
    
