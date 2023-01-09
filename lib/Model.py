
import torch
import torch.nn as nn
import numpy as np

### The structure of FCNN which takes MTC-MRF signal as an input and 
### quantifies four tissue parameters. 

class nnModel(nn.Module):
    def __init__(self,ds_num,device):
        super(nnModel, self).__init__()
        linear1 = nn.Linear(ds_num,256)
        linear2 = nn.Linear(256,256)
        linear3 = nn.Linear(256,4)
        relu = nn.ReLU()
        sig  = nn.Sigmoid()
        self.device=device
        self.fc_module = nn.Sequential(
            linear1,
            relu,
            linear2,
            relu,
            linear3,
            sig
        )

    def forward(self,input):
        out = self.fc_module(input)

        ## De-normalize the tissue parameters (normalization has doen with sigmoid function)
        min_4=torch.tensor([5,0.02,1e-6,0.2],device=self.device,requires_grad=False)
        max_4=torch.tensor([100,0.17,1e-4,3.0],device=self.device,requires_grad=False)
        out_denorm = torch.mul(out,max_4-min_4) + min_4

        return out_denorm
