import torch
import observables 
import BNAF
import matrices

def train(wave_function, wave_function_phase, batch_size, num_batches, num_epochs, ndim, nmat, tHooft, mass, lr, info_load, info_save, info_gauge, coeff_G, info_yq, coeff_yq, yvec, qvec):

    nboson = ndim * (nmat * nmat - 1)
    #optimizer = torch.optim.AdamW(wave_function.parameters(), lr)
    #optimizer = torch.optim.AdamW(wave_function_phase.parameters(), lr)
    optimizer = torch.optim.SGD(wave_function.parameters(), lr)
    optimizer_phase = torch.optim.SGD(wave_function_phase.parameters(), lr)
    ######################################################
    ### Load the parameters of the model and optimizer ###
    ######################################################
    if(info_load == True):
        checkpoint = torch.load('./trained_parameters.pth', weights_only=True)
        wave_function.load_state_dict(checkpoint['wave_function_state_dict'])
        wave_function_phase.load_state_dict(checkpoint['wave_function_phase_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_phase.load_state_dict(checkpoint['optimizer_phase_state_dict'])
    ################
    ### Training ###
    ################
    for epoch in range(num_epochs):
        energy_epoch = 0
        loss_epoch = 0
        trG2_epoch = 0
        for ibatch in range(num_batches):
            #optimizer.zero_grad() # reset values of derivatives to zero
            #####################################################################
            ### Base distribution is Gaussian here   ############################
            ### If you want to use different distributions, change this part  ###
            #####################################################################
            # psi is the absolute value of wave function. phase is taken into account later. 
            # sample zvec from Gaussian
            zvec = torch.randn(batch_size, nboson, requires_grad=True)
            # apply the BNAF flow and get xvec            
            xvec = wave_function(zvec)
            # get Jacobian matrix and (log of) Jacobian
            log_jac, jacobian_matrix = BNAF.log_Jacobian(wave_function, zvec, batch_size, ndim, nmat)
            log_psi_abs = -0.5 * log_jac -0.25 * torch.sum(zvec * zvec, axis=1)
            ################################################################
            ### Calculate the energy. xvec is needed for potential.   ######
            ### zvec, jacobian, etc, are needed to get the kinetic term. ###
            ################################################################
            optimizer.zero_grad() # reset values of derivatives to zero (backproparagtion is used to calculate kinetic energy)
            optimizer_phase.zero_grad()
            if(info_gauge == True):
                energy, trG2 = observables.calc_total_energy_and_gauge(xvec, zvec, jacobian_matrix, log_psi_abs, tHooft, mass, ndim, nmat, batch_size)
                energy_batch = torch.sum(energy)/batch_size/(nmat*nmat)
                trG2_batch = torch.sum(trG2)/batch_size/(nmat*nmat)
                ### cost function = energy + c * TrG^2, up to normalization ###
                loss_batch = energy_batch + trG2_batch * coeff_G
                trG2_epoch += trG2_batch
            else:
                energy = observables.calc_total_energy(xvec, zvec, jacobian_matrix, log_psi_abs, tHooft, mass, ndim, nmat, batch_size)
                ### cost function = energy, up to normalization ###
                energy_batch = torch.sum(energy)/batch_size/(nmat*nmat)
                loss_batch = energy_batch + 0  # "+0" is used to avoid sharing the ID
            ##################################################################################
            ### Calculate <\hat{x}> and <\hat{p}> and add constraint term to loss function ###
            ##################################################################################
            if(info_yq == True):                     
                phase = wave_function_phase(xvec) # phase is needed when <\hat{p}> is nonzero
                center_x, center_p = observables.center_of_wave_packet(phase, xvec)
                ## additional contribution to kinetic energy from the phase factor ##
                kin_additional = 0.5 * torch.sum(center_p ** 2)/batch_size/(nmat*nmat)
                energy_batch += kin_additional
                loss_batch += kin_additional 
                
                center_x_batch = torch.sum(center_x, dim=0)/batch_size
                center_p_batch = torch.sum(center_p, dim=0)/batch_size

                # add constraint term to the loss
                loss_batch += torch.sum( ( center_x_batch - yvec )**2 + ( center_p_batch - qvec )**2 ) * coeff_yq
                # Add extra constraint to set the average phase to be zero.
                # This mounts to a constant shift of the phase and hence does not change physics.
                sum_phase = torch.sum(phase)
                loss_batch += sum_phase ** 2  / batch_size ** 2
            ###############################    
            ### Update model parameters ###
            ###############################
            loss_epoch += loss_batch
            energy_epoch += energy_batch
            optimizer.zero_grad() # reset values of derivatives to zero before Gradient Descent
            optimizer_phase.zero_grad()
            loss_batch.backward()
            optimizer.step()
            optimizer_phase.step()
        if(info_gauge == True):    
            print("epoch=",epoch+1,"E/N^2=",energy_epoch.item()/num_batches,"G/N^2=", trG2_epoch.item()/num_batches)
        else:
            if(info_yq == True):
                print("epoch=",epoch+1,"loss=",loss_epoch.item()/num_batches, "E/N^2=",energy_epoch.item()/num_batches)
                su_instance = matrices.SU(nmat, 1)
                center_x_mat = su_instance.vector_to_matrix(center_x_batch, ndim, nmat, 1)
                center_p_mat = su_instance.vector_to_matrix(center_p_batch, ndim, nmat, 1)

                print("center of xmat =", center_x_mat)
                print("center of pmat =", center_p_mat)
            if(info_yq == False):
                print("epoch=",epoch+1,"E/N^2=",energy_epoch.item()/num_batches)
    ######################################################
    ######################################################
    ### Save the parameters of the model and optimizer ###
    ######################################################
    if(info_save == True):                                          
        torch.save({
            'wave_function_state_dict': wave_function.state_dict(),
            'wave_function_phase_state_dict': wave_function_phase.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_phase_state_dict': optimizer_phase.state_dict(),
        }, './trained_parameters.pth')




