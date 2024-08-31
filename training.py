import torch
import observables 
import BNAF

def train(wave_function, batch_size, num_batches, num_epochs, ndim, nmat, tHooft, mass, lr, info_load, info_save, info_gauge, coeff_G):


    nboson = ndim * (nmat * nmat - 1)
    #optimizer = torch.optim.AdamW(wave_function.parameters(), lr)
    optimizer = torch.optim.SGD(wave_function.parameters(), lr)
    ######################################################
    ### Load the parameters of the model and optimizer ###
    ######################################################
    if(info_load == True):
        checkpoint = torch.load('./trained_parameters.pth', weights_only=True)
        wave_function.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ################
    ### Training ###
    ################
    for epoch in range(num_epochs):
        loss_epoch = 0
        trG2_epoch = 0
        for ibatch in range(num_batches):
            #optimizer.zero_grad() # reset values of derivatives to zero
            #####################################################################
            ### Base distribution is Gaussian here   ############################
            ### If you want to use different distributions, change this part  ###
            #####################################################################
            # psi is wave function. phase is not taken into account yet. (need to be fixed to study excited states)
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
            if(info_gauge == True):
                energy, trG2 = observables.calc_total_energy_and_gauge(xvec, zvec, jacobian_matrix, log_psi_abs, tHooft, mass, ndim, nmat, batch_size)
                trG2_batch = torch.sum(trG2)/batch_size/(nmat*nmat)
                ### cost function = energy + c * TrG^2, up to normalization ###
                loss_batch = torch.sum(energy)/batch_size/(nmat*nmat) + trG2_batch * coeff_G
                trG2_epoch += trG2_batch
            else:
                energy = observables.calc_total_energy(xvec, zvec, jacobian_matrix, log_psi_abs, tHooft, mass, ndim, nmat, batch_size)
                ### cost function = energy, up to normalization ###
                loss_batch = torch.sum(energy)/batch_size/(nmat*nmat)
            
           
            loss_epoch += loss_batch
            optimizer.zero_grad() # reset values of derivatives to zero before Gradient Descent
            loss_batch.backward()
            optimizer.step()
        if(info_gauge == True):    
            print("epoch=",epoch+1,"E/N^2=",loss_epoch.item()/num_batches,"G/N^2=", trG2_epoch.item()/num_batches)
        else:
            print("epoch=",epoch+1,"E/N^2=",loss_epoch.item()/num_batches)
    ######################################################
    ######################################################
    ### Save the parameters of the model and optimizer ###
    ######################################################
    if(info_save == True):                                          
        torch.save({
           'model_state_dict': wave_function.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
        }, './trained_parameters.pth')




