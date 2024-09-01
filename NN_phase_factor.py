import torch.nn as nn 
#import torch.nn.functional as F
class phase_of_wave_function(nn.Module):
    def __init__(self, nmat, ndim, alpha_phase):
        super(phase_of_wave_function, self).__init__()

        nboson = ( nmat * nmat - 1) * ndim

        self.dense1 = nn.Linear(nboson, alpha_phase * nboson)
        self.lkReLU = nn.LeakyReLU(0.2)
        self.dense2 = nn.Linear(alpha_phase * nboson, 1)
        
    def forward(self, x):
        x = self.dense1(x)
        x = self.lkReLU(x)
        phase = self.dense2(x)
  
        return phase
