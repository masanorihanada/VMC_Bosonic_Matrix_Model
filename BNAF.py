#############################################################################
### PyTorch implimentation of Bosonic Matrix Model with BNAF ################
### See https://github.com/hanxzh94/minimal_BMN_MQM for Xizhi Han's code. ###
### See also  https://github.com/nicola-decao/BNAF ##########################
#############################################################################
#### get_weight_mask, MaskedLinear, and MaskedMLP are based on Xizhi's code.# 
#############################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
########################################################################################
###   Return the block autoregressive diagonal and off-diagonal masks of the weight. ###
########################################################################################
#
#  Args:
#      dim_in (int): input dimension of the weight
#      dim_out (int): output dimension of the weight
#      num_blocks (int): the number of autoregressive blocks
#
#  Return:
#      mask_d (tensor of shape (dim_out, dim_in)): "d" means diagonal, i.e., a binary mask where the weights within each block are ones
#      mask_o (tensor of shape (dim_out, dim_in)): "o" means off-diagonal, i.e., a binary mask where the weights from block i to block j and i < j are ones
#
# diagonal and off-diagonal should be distinguished because we impose the positivity constraint to the diagonal part
def get_weight_mask(dim_in, dim_out, num_blocks):
    # alpha will be used as num_blocks
    # "step" is the number of nodes in each layer in each block
    assert dim_in % num_blocks == 0 and dim_out % num_blocks == 0, f"invalid {dim_in}, {dim_out}, {num_blocks}"
    step_in, step_out =  dim_in // num_blocks, dim_out // num_blocks 
    #################################################
    ## mask for the block diagonals of the weights ##
    #################################################
    mask_d = torch.zeros((dim_out, dim_in))
    for iblock in range(num_blocks):
        mask_d[iblock * step_out : (iblock + 1) * step_out, iblock * step_in : (iblock + 1) * step_in] += 1
    #####################################################
    ## mask for the block autoregressive off-diagonals ##
    #####################################################
    mask_o = torch.ones((dim_out, dim_in))
    for iblock in range(num_blocks):
        mask_o[iblock * step_out : (iblock + 1) * step_out, iblock * step_in : ] -= 1
    return mask_d.float(), mask_o.float()
########################################################    
########################################################
## Masked MLP, optionally with SinhArcsinh activation ##
########################################################
########################################################

##################
## Masked Layer ##
##################
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask_d, mask_o, info_SinhArcsinh):
      
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask_d = mask_d
        self.mask_o = mask_o
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * torch.sqrt(torch.tensor(2.0)/(out_features+in_features)))
        self.bias = nn.Parameter(torch.randn(out_features)* torch.sqrt(torch.tensor(2.0)/(out_features+in_features)))
        self.info_SinhArcsinh = info_SinhArcsinh # True -> SinhArcsinh activation, False -> no activation
        
        if(self.info_SinhArcsinh == True):
            self.tail = nn.Parameter(torch.randn(out_features) * 0.1)
            self.skewness = nn.Parameter(torch.randn(out_features) * 0.1)
        else:
            self.tail = nn.Parameter(torch.zeros(out_features))
            self.skewness = nn.Parameter(torch.zeros(out_features))
       
        
        assert mask_d.shape == (out_features, in_features), "Mask shape must match weight shape."
        assert mask_o.shape == (out_features, in_features), "Mask shape must match weight shape."


    def forward(self, x):
        # Apply mask to weights
        masked_weight = torch.exp(self.weight) * self.mask_d + self.weight * self.mask_o
        # <- Xizhi's choice. Rather stable. 
        #masked_weight = self.weight * self.mask_d + self.weight * self.mask_o
        # <- This is not good.
        if(self.info_SinhArcsinh == True):# SinhArcsinh activation
            return torch.sinh(torch.exp(self.tail) * (self.skewness + torch.asinh(F.linear(x, masked_weight, self.bias))))
        else:# no activation at the output layer
            return F.linear(x, masked_weight, self.bias)
#############################
## MLP with 1 hidden layer ##
#############################
class MaskedMLP_1_hidden_layer(nn.Module):
    def __init__(self, nmat, ndim, alpha):
        super(MaskedMLP_1_hidden_layer, self).__init__()
        assert len(alpha) == 1
    
        nboson = ndim * (nmat * nmat - 1)
        num_blocks = nboson
        
        dim_in = nboson
        dim_out = nboson * alpha[0]
        mask_d, mask_o = get_weight_mask(dim_in, dim_out, num_blocks)
        info_SinhArcsinh = True
        self.hidden_layer_1 = MaskedLinear(dim_in, dim_out, mask_d, mask_o, info_SinhArcsinh)
        
        dim_in = nboson * alpha[0]
        dim_out = nboson
        mask_d, mask_o = get_weight_mask(dim_in, dim_out, num_blocks)
        info_SinhArcsinh = False
        self.output_layer = MaskedLinear(dim_in, dim_out, mask_d, mask_o, info_SinhArcsinh)
        
    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.output_layer(x)
        return x
##############################
## MLP with 2 hidden layers ##
##############################
class MaskedMLP_2_hidden_layers(nn.Module):
    def __init__(self, nmat, ndim, alpha):
        super(MaskedMLP_2_hidden_layers, self).__init__()
        assert len(alpha) == 2
    
        nboson = ndim * (nmat * nmat - 1)
        num_blocks = nboson
    
        dim_in = nboson
        dim_out = nboson * alpha[0]
        mask_d, mask_o = get_weight_mask(dim_in, dim_out, num_blocks)
        info_SinhArcsinh = True
        self.hidden_layer_1 = MaskedLinear(dim_in, dim_out, mask_d, mask_o, info_SinhArcsinh)

        dim_in = nboson * alpha[0]
        dim_out = nboson * alpha[1]
        mask_d, mask_o = get_weight_mask(dim_in, dim_out, num_blocks)
        info_SinhArcsinh = True
        self.hidden_layer_2 = MaskedLinear(dim_in, dim_out, mask_d, mask_o, info_SinhArcsinh)
        
        dim_in = nboson * alpha[1]
        dim_out = nboson
        mask_d, mask_o = get_weight_mask(dim_in, dim_out, num_blocks)
        info_SinhArcsinh = False
        self.output_layer = MaskedLinear(dim_in, dim_out, mask_d, mask_o, info_SinhArcsinh)
        
    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)

        return x
###########################################################################
### Compute Jacobian matrix dx/dz and (log of) Jacobian abs[det(dx/dz)] ###
### Note that dx/dz is lower-triangle in BNAF       #######################
###########################################################################
def log_Jacobian(model, zvec, batch_size, ndim, nmat):
    # Input validation
    assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
    assert isinstance(zvec, torch.Tensor), "zvec must be a torch.Tensor"
    assert zvec.shape[0] == batch_size, "zvec first dimension must match batch_size"
    
    nboson = ndim * (nmat * nmat - 1)
    device = zvec.device

    epsilon = 0.00000000000000000001
    
    log_jac = []
    jacobian_matrix = []
    for isample in range(batch_size):
        #################################################################################
        ### Caution: torch.autograd.functional.jacobian destroys computational graph. ###
        ### torch.autograd.functional.jvp does not have this problem.       #############
        #################################################################################
        # torch.autograd.functional.jvp gives Jacobian * vector.
        # by using all one-hot vectors, Jacobian can be obtained.
        # If we didn't need to take a derivative of Jacobian, torch.autograd.functional.jacobian could be used.  
        #zvec_sample = zvec[isample, :].clone().requires_grad_(True)
        jacobian_rows = []
        for ivec in range(nboson):
            # Create a one-hot vector
            v = torch.zeros(nboson, device=device)
            v[ivec] = 1.0
            # Compute the Jacobian-vector product
            grad = torch.autograd.functional.jvp(model, zvec[isample, :], v, create_graph=True, strict=True)
            # grad[0] is model(zvec), which is xvec in this case (note that model = wave_function is assumed)
            # grad[1] is (dx/dz)*v, which is the ivec-th row of Jacobian matrix 
            jacobian_rows.append(grad[1])  #.clone().requires_grad_(True))
        # Stack all rows to form the full Jacobian matrix
        jacobian_sample = torch.stack(jacobian_rows)
        # Although diagonal entries, that contribute to trace, does not become log(0),
        # off-diagonal entries do. Without epsilon, somehow these off-diagonal entries causes issues, it seems. 
        log_jac_sample = 0.5 * torch.trace(torch.log(jacobian_sample * jacobian_sample + epsilon))
        log_jac.append(log_jac_sample)
        jacobian_matrix.append(jacobian_sample)
    jacobian_matrix = torch.stack(jacobian_matrix)
    jacobian_matrix = jacobian_matrix.permute(0, 2, 1) # match the usual convention
    log_jac = torch.stack(log_jac)
    return log_jac, jacobian_matrix
    




