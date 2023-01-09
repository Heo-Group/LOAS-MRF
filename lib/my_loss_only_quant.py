import torch

def my_loss_only_quant(output,target,device):

    ## ruling out the T2w parameters from the target tissue parameters
    target = target[:,0:4]

    ## Tissue parameters are normalied to calculate the loss 
    ## Since they span different order of range, normalization is neccesary for balanced the loss calculation.
    min_4=torch.tensor([5,0.02,1e-6,0.2],device=device,requires_grad=False)
    max_4=torch.tensor([100,0.17,1e-4,3.0],device=device,requires_grad=False)

    output_norm=torch.div(output-min_4,max_4-min_4)
    target_norm=torch.div(target-min_4,max_4-min_4)

    diff_norm = (output_norm-target_norm)**2

    ## RMSE of each tissue parametes are monitored     
    K_diff=torch.sqrt(torch.mean(diff_norm[:,0]))
    M_diff=torch.sqrt(torch.mean(diff_norm[:,1]))
    T2m_diff=torch.sqrt(torch.mean(diff_norm[:,2]))
    T1w_diff=torch.sqrt(torch.mean(diff_norm[:,3]))

    ## The loss is given as the mean square error between estimated parameters (Output) and ground-truths (Input)
    error_total = torch.mean(diff_norm)

    return error_total,K_diff,M_diff,T2m_diff,T1w_diff