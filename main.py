###################################################################################
### PyTorch implimentation of Bosonic Matrix Model with BNAF by Masanori Hanada ###
### See https://github.com/hanxzh94/minimal_BMN_MQM for Xizhi Han's code. #########
### See also  https://github.com/nicola-decao/BNAF ################################
###################################################################################
import training
import BNAF
#import matrices
#import observables
#import torch
############################
## cheking random numbers ##
############################
#seed = 0
#print("set seed", seed)
#torch.manual_seed(seed)
#print(torch.randn(1))
###########################
## Simulation parameters ##
###########################
nmat = 2  # matrix size
ndim = 2  # number of matrices
nlayer = 1 # number of layers
if(nlayer == 1):
        alpha = [2] # number of nodes, for nlayer = 1
elif(nlayer == 2):        
        alpha = [2,2]  # number of nodes, for nlayer = 2
mass = 1.0 # mass; 0.5 * m^2 * tr X^2
tHooft = 0.0 # 't Hooft coupling lambda = g^2*N.
coeff_G = 1.0 # add coeff_G * TrG^2 to the Hamiltonian when info_gauge = True
num_epochs = 100 
num_batches = 10
batch_size = 512
lr = 0.01 # learning rate
info_load = True # True -> NN parameters are loaded from trained_parameters.pth, False -> nothing
info_save = True  # True -> NN parameters are saved in trained_parameters.pth, False -> nothing
info_gauge = True # True -> add TrG^2-term to the Hamiltonian 
#################################
## Create model with the masks ##
#################################
if(nlayer == 1):
        wave_function = BNAF.MaskedMLP_1_hidden_layer(nmat, ndim, alpha)
elif(nlayer == 2):
	wave_function = BNAF.MaskedMLP_2_hidden_layers(nmat, ndim, alpha)
##############
## Training ##
##############
losses = training.train(wave_function, batch_size, num_batches, num_epochs, ndim, nmat, tHooft, mass, lr, info_load, info_save, info_gauge, coeff_G)


