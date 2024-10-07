import matrices
import torch
####################################################################
### potential V(X) = Tr( 0.5 * m^2 * X_I^2 - 0.25 * [X_I,X_J]^2) ###
####################################################################
def calc_potential(xvec, tHooft, mass, ndim, nmat, batch_size, device):
    nboson = ndim * (nmat * nmat - 1)
    #################
    ### mass term ###
    #################
    potential = 0.5 * mass * mass * torch.sum(xvec**2, axis=1)
    #######################
    ### commutator term ###
    #######################
    su_instance = matrices.SU(nmat, batch_size, device)
    xmat = su_instance.vector_to_matrix(xvec, ndim, nmat, batch_size)

    quartic_term = torch.zeros( batch_size ).to(device)
    for idim in range(1,ndim):
        for jdim in range(0,idim):
            commutator =  torch.matmul(xmat[ : , idim], xmat[ : , jdim]) - torch.matmul(xmat[ : , jdim], xmat[ : , idim])
            quartic_term += 0.5 * tHooft / nmat * torch.sum(commutator * commutator.conj(), dim = (1,2) ).real
    potential += quartic_term
    
    return potential
############################################
### kinetic term 0.5 * [(d/dx)log_psi]^2 ###
############################################
def calc_kinetic(zvec, jacobian_matrix, log_psi_abs, ndim, nmat, batch_size, device):

    assert zvec.requires_grad
    
    nboson = ndim * (nmat * nmat - 1)

    sum_log_psi_abs = torch.sum(log_psi_abs)
    sum_log_psi_abs.backward(retain_graph=True)# needed to get zvec.grad = (d/dz)log_psi_abs
    ########################################################################
    ## zvec.grad is (d/dz)log_psi_abs.                   ###################
    ## We need to multiply dz/dx = (dx/dz)^{-1} = jacobian_sample_inverse ##
    ##  to get (d/dx)log_psi_abs.                        ###################
    ########################################################################
    kinetic_values = []
    for isample in range(batch_size):
        # get (dx/dz)^{-1}
        jacobian_sample_inverse = torch.linalg.inv(jacobian_matrix[isample, : , : ])
        # (d/dz)log_psi_abs -> (d/dx)log_psi_abs = (dx/dz)^{-1} * (d/dz)log_psi_abs
        deriv_log_psi_abs = torch.matmul(jacobian_sample_inverse, zvec.grad[isample, :])    
        kinetic_value = torch.sum(deriv_log_psi_abs * deriv_log_psi_abs) * 0.5
        kinetic_values.append(kinetic_value)
        
    # Concatenate kinetic values into a single differentiable tensor
    kinetic = torch.stack(kinetic_values).to(device)

    return kinetic
###########################################
### total energy = kinetic + potential  ###
###########################################
def calc_total_energy(xvec, zvec, jacobian_matrix, log_psi_abs, tHooft, mass, ndim, nmat, batch_size, device):

    pot = calc_potential(xvec, tHooft, mass, ndim, nmat, batch_size, device)
    kin = calc_kinetic(zvec, jacobian_matrix, log_psi_abs, ndim, nmat, batch_size, device)
    energy = pot + kin
    
    return energy
############################################
### kinetic term 0.5 * [(d/dx)log_psi]^2 ###
###    and gauge term <G^2>     ############
########################################
### Both requires (d/dx)log_psi_abs, ###
### so let's compute them together   ###
########################################
def calc_kinetic_and_gauge(xvec, zvec, jacobian_matrix, log_psi_abs, ndim, nmat, batch_size, device):

    assert zvec.requires_grad
    
    nboson = ndim * (nmat * nmat - 1)
    deriv_log_psi_abs_batch = []
    trG2 = []

    sum_log_psi_abs = torch.sum(log_psi_abs)
    sum_log_psi_abs.backward(retain_graph=True)# needed to get zvec.grad = (d/dz)log_psi_abs
    ########################################################################
    ## zvec.grad is (d/dz)log_psi_abs.                   ###################
    ## We need to multiply dz/dx = (dx/dz)^{-1} = jacobian_sample_inverse ##
    ##  to get (d/dx)log_psi_abs.                        ###################
    ########################################################################
    kinetic_values = []
    for isample in range(batch_size):
        # get (dx/dz)^{-1}
        jacobian_sample_inverse = torch.linalg.inv(jacobian_matrix[isample, : , : ])
        # (d/dz)log_psi_abs -> (d/dx)log_psi_abs = (dx/dz)^{-1} * (d/dz)log_psi_abs
        deriv_log_psi_abs = torch.matmul(jacobian_sample_inverse, zvec.grad[isample, :])    
        kinetic_value = torch.sum(deriv_log_psi_abs * deriv_log_psi_abs) * 0.5
        kinetic_values.append(kinetic_value)
        # keep (d/dx)log_psi_abs to compute gauge violation
        deriv_log_psi_abs_batch.append(deriv_log_psi_abs)
        
    # Concatenate kinetic values into a single differentiable tensor
    kinetic = torch.stack(kinetic_values)
    ############################
    ## trG2 (gauge violation) ##
    ############################
    deriv_log_psi_abs_batch = torch.stack(deriv_log_psi_abs_batch)
    su_instance = matrices.SU(nmat, batch_size, device)
    # let's use matrix form because we don't want to compute structure constant of SU(N) group
    xmat = su_instance.vector_to_matrix(xvec, ndim, nmat, batch_size)
    deriv_log_psi_abs_batch_mat = su_instance.vector_to_matrix(deriv_log_psi_abs_batch, ndim, nmat, batch_size)

    for isample in range(batch_size):
        gauge_value = 0
        for idim in range(0,ndim):
            gauge_temp = torch.matmul(xmat[isample, idim], deriv_log_psi_abs_batch_mat[isample, idim]) \
                - torch.matmul(deriv_log_psi_abs_batch_mat[isample, idim], xmat[isample, idim])
            gauge_value += torch.sum(gauge_temp * gauge_temp.conj()).real
        trG2.append(gauge_value)
    trG2 = torch.stack(trG2)

    return kinetic, trG2
###########################################
### total energy = kinetic + potential  ###
###########################################
def calc_total_energy_and_gauge(xvec, zvec, jacobian_matrix, log_psi_abs, tHooft, mass, ndim, nmat, batch_size, device):

    pot = calc_potential(xvec, tHooft, mass, ndim, nmat, batch_size, device)
    kin, trG2 = calc_kinetic_and_gauge(xvec, zvec, jacobian_matrix, log_psi_abs, ndim, nmat, batch_size, device)
    energy = pot + kin
    
    return energy, trG2

######################################################
### center of wave packet <\hat{x}> and <\hat{p}>  ###
######################################################
def center_of_wave_packet(phase, xvec):

    center_x = xvec 
    sum_phase = torch.sum(phase)
    center_p = torch.autograd.grad(outputs=sum_phase, inputs=xvec, create_graph=True, retain_graph=True)[0]
    
    return center_x, center_p
