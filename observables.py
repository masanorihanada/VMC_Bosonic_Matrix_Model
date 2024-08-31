import matrices
import torch
####################################################################
### potential V(X) = Tr( 0.5 * m^2 * X_I^2 - 0.25 * [X_I,X_J]^2) ###
####################################################################
def calc_potential(xvec, tHooft, mass, ndim, nmat, batch_size):
    nboson = ndim * (nmat * nmat - 1)
    #################
    ### mass term ###
    #################
    potential = 0.5 * mass * mass * torch.sum(xvec**2, axis=1)
    #######################
    ### commutator term ###
    #######################
    su_instance = matrices.SU(nmat, batch_size)
    xmat = su_instance.vector_to_matrix(xvec, ndim, nmat, batch_size)
    commutator = torch.zeros((batch_size,nmat,nmat), dtype=torch.complex64)
    quartic_values = []
    for isample in range(batch_size):
        quartic_value = 0
        for idim in range(1,ndim):
            for jdim in range(0,idim):
                commutator = torch.matmul(xmat[isample, idim], xmat[isample, jdim]) - torch.matmul(xmat[isample, jdim], xmat[isample, idim])
                quartic_value += 0.5 * tHooft / nmat * torch.sum(commutator * commutator.conj()).real
        quartic_values.append(quartic_value)
    potential += torch.stack(quartic_values)   
    return potential
############################################
### kinetic term 0.5 * [(d/dx)log_psi]^2 ###
############################################
def calc_kinetic(zvec, jacobian_matrix, log_psi_abs, ndim, nmat, batch_size):

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
    kinetic = torch.stack(kinetic_values)

    return kinetic

###########################################
### total energy = kinetic + potential  ###
###########################################
def calc_total_energy(xvec, zvec, jacobian_matrix, log_psi_abs, tHooft, mass, ndim, nmat, batch_size):

    pot = calc_potential(xvec, tHooft, mass, ndim, nmat, batch_size)
    kin = calc_kinetic(zvec, jacobian_matrix, log_psi_abs, ndim, nmat, batch_size)
    bosonic_energy = pot + kin
    #print(pot.requires_grad,kin.requires_grad,bosonic_energy.requires_grad)
    return bosonic_energy
