###################################################################################
### PyTorch implimentation of Bosonic Matrix Model with BNAF by Masanori Hanada ###
### See https://github.com/hanxzh94/minimal_BMN_MQM for Xizhi Han's code. #########
### See also  https://github.com/nicola-decao/BNAF ################################
###################################################################################
import training
import BNAF
import matrices
#import observables
import NN_phase_factor
import torch
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
nmat = 2  # matrix size N
ndim = 2  # number of matrices
nlayer = 1 # number of layers
if(nlayer == 1):
        alpha = [2] # number of nodes, for nlayer = 1
elif(nlayer == 2):        
        alpha = [2,2]  # number of nodes, for nlayer = 2
        
alpha_phase = 10 # width of NN for phase of wave function will be alpha_phase * (dim of input)
mass = 1.0 # mass; 0.5 * m^2 * tr X^2
tHooft = 0.0 # 't Hooft coupling lambda = g^2 * N.
coeff_G = 0.0 # used when info_gauge = True
coeff_yq = 10.0 # used when info_yq = True
num_epochs = 10
num_batches = 10
batch_size = 1024
lr = 0.01 # learning rate
info_load = True # True -> NN parameters are loaded from trained_parameters.pth, False -> nothing
info_save = True  # True -> NN parameters are saved in trained_parameters.pth, False -> nothing
############################################################################
## Caution! info_gauge = info_yq = True can make sense only if y = q = 0. ##
## Note also that info_yq = True makes sense only when batch_size is large.#
############################################################################
info_gauge = False # True -> coeff_G * TrG^2 to the Hamiltonian 
info_yq = True # True -> add c_{yq} * ( <\hat{x}> - y )^2 + ( <\hat{p}> - q )^2 to the loss function
#############################################################
## We need to choose appropriate y and q if info_yq = True ##
#############################################################
# The specific choice of y and q below means:
#   (N-1) D-branes at the origin, one probe D-brane at (y,0,..,0) in R^d with momentum (p,0,...,0)
#no string excited
nboson = ndim * (nmat * nmat - 1)
yvec = torch.zeros(1, nboson, requires_grad=False)
qvec = torch.zeros(1, nboson, requires_grad=False)

su_instance = matrices.SU(nmat, 1)
ymat = su_instance.vector_to_matrix(yvec, ndim, nmat, 1)
qmat = su_instance.vector_to_matrix(qvec, ndim, nmat, 1)

probe_location = 1.41421356 # distance between D-branes
probe_momentum = 1.41421356 # momentum of probe D-brane

ymat[0, 0, nmat-1, nmat-1] = probe_location
qmat[0, 0, nmat-1, nmat-1] = probe_momentum

for imat in range(nmat):# make ymat traceless
        ymat[0, 0,imat,imat] += - probe_location/nmat
        qmat[0, 0,imat,imat] += - probe_momentum/nmat
        
yvec = su_instance.matrix_to_vector(ymat, ndim, nmat, 1)
qvec = su_instance.matrix_to_vector(qmat, ndim, nmat, 1)
yvec = yvec.reshape(nboson)
qvec = qvec.reshape(nboson)
#################################
## Create model with the masks ##
#################################
if(nlayer == 1):
        wave_function = BNAF.MaskedMLP_1_hidden_layer(nmat, ndim, alpha)
        #wave_function_phase = BNAF.MaskedMLP_1_hidden_layer(nmat, ndim, alpha)
elif(nlayer == 2):
	wave_function = BNAF.MaskedMLP_2_hidden_layers(nmat, ndim, alpha)
        #wave_function_phase = BNAF.MaskedMLP_2_hidden_layers(nmat, ndim, alpha)
#############################################
## Create model for phase of wave function ##
#############################################
wave_function_phase = NN_phase_factor.phase_of_wave_function(nmat, ndim, alpha_phase)
##############
## Training ##
##############
losses = training.train(wave_function, wave_function_phase, batch_size, num_batches, num_epochs, ndim, nmat, tHooft, mass, lr, info_load, info_save, info_gauge, coeff_G, info_yq, coeff_yq, yvec, qvec)


