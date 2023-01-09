import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import time
import os
import argparse
from torch.utils.data import TensorDataset, DataLoader

from lib.Signal_Generation_noise import Signal_Gen_noise
from lib.Model import nnModel
from lib.my_loss_only_quant import my_loss_only_quant


## Specification of structure and learning-related hyperparameters
## can be easily manipulated with parser.  
parser = argparse.ArgumentParser(description='Setting')
parser.add_argument('--result', type=str, default='result',help='The number of folder for the save of model and loss')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--batch', type=int, default=256)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--ds', type=int, default=40,help='The number of dynamic scans')
args = parser.parse_args()


if __name__ == '__main__':
    with open(args.result+'/arguments.txt','w') as f:
        print(args,file = f)
    if not os.path.exists(args.result):
        os.mkdir(args.result)   
        
    ## GPU allocation
    GPU_NUM = args.gpu # GPU number
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU
    print ('Current cuda device ', torch.cuda.current_device()) # check
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    ## Dataset load ######################################################################
    dir_data = "data/"

    X_mat = sio.loadmat(dir_data + "TrainLabel_MTC.mat")
    train_X = X_mat['TrainLabel']

    X_mat = sio.loadmat(dir_data + "TestLabel_MTC.mat")
    test_X = X_mat['TestLabel']

    X_train=torch.FloatTensor(train_X)

    X_test=torch.FloatTensor(test_X)
    ## RF parameters Gen ################################################################
    ## Randomly sample the RF scan parameter from [0 1]
    ## and min max values of RF parameter are defined.
    ## Each range represents B1 power, freq offset, saturation time, delay time

    RF_parameters=torch.rand(args.ds,4,device=device,requires_grad=True)

    min_4=torch.tensor([0.9,8,0.4,3.5],device=device,requires_grad=False)
    max_4=torch.tensor([1.9,50,2,4.5],device=device,requires_grad=False)
    #####################################################################################
    trainset = TensorDataset(X_train)
    trainloader=DataLoader(trainset,batch_size=args.batch,shuffle=True)

    testset = TensorDataset(X_test)
    testloader=DataLoader(testset,batch_size=args.batch,shuffle=False)

    ## Model Define & Optimizer define ############################################
    fcnn = nnModel(args.ds,device)
    fcnn = fcnn.to(device)
    optimizer = torch.optim.Adam(fcnn.parameters(), lr=args.lr)
    optimizer_RF = torch.optim.Adam([RF_parameters], lr=args.lr)
    ################################################################################

    losses = []
    loss_test = []
    loss_quantification = []
    loss_K = []
    loss_M = []
    loss_T2m = []
    loss_T1w = []

    for epoch in range(args.epochs):
        start_time=time.time()
        batch_loss = 0 
        batch_loss_K = 0 
        batch_loss_M = 0 
        batch_loss_T2m = 0 
        batch_loss_T1w = 0 
        batch_loss_test=0

        for i,data in enumerate(trainloader):
            #### RF parameters de-normalization ############################
            ## de-normalized RF parameters with pre-defined range of each parameter.
            RF_parameters_norm = torch.sigmoid(RF_parameters)
            RF_parameters_denorm = torch.mul(RF_parameters_norm,max_4-min_4) + min_4
            ############################################
                    
            [X_batch]=data
            X_batch=X_batch.to(device)
            ## MTC-MRF signal is generated with scan parameters and tissue parameters
            ## via the analytical solution of two-pool Bloch equation
            ## Then fed to the fcnn to quantifies the tissue parameters. 
            MTC_MRF = Signal_Gen_noise(X_batch,RF_parameters_denorm,args.ds,device)
            x_pred = fcnn(MTC_MRF)

            loss,RMSE_K,RMSE_M,RMSE_T2m,RMSE_T1w = my_loss_only_quant(x_pred,X_batch,device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_RF.step()
            batch_loss += loss.item()
            
            if i+1 == (len(X_train)//args.batch):

                print('Epoch:', '%04d' % (epoch + 1),'Time. taken =', '{:.3f}'.format(time.time()-start_time))
                print('Epoch:', '%04d' % (epoch + 1),'Train Loss (X10^5) =', '{:.3f}'.format(1e5*batch_loss / (i+1)))

        ## Accuracy for test ####
        RF_parameters_norm = torch.sigmoid(RF_parameters)
        RF_parameters_denorm = torch.mul(RF_parameters_norm,max_4-min_4) + min_4

        
        with torch.no_grad(): # for test 
            test_loss = 0.0
            for j, data in enumerate(testloader):
                [X_batch]=data
                X_batch=X_batch.to(device)
                MTC_MRF = Signal_Gen_noise(X_batch,RF_parameters_denorm,args.ds,device)
                x_pred = fcnn(MTC_MRF)

                loss,RMSE_K,RMSE_M,RMSE_T2m,RMSE_T1w = my_loss_only_quant(x_pred,X_batch,device)

                batch_loss_test += loss.item()
                batch_loss_K += RMSE_K.item()
                batch_loss_M += RMSE_M.item()
                batch_loss_T2m += RMSE_T2m.item()
                batch_loss_T1w += RMSE_T1w.item()
            print('Epoch:', '%04d' % (epoch + 1),'Test Loss (X10^5)=', '{:.3f}'.format(1e5*batch_loss_test / (j+1)))
            print('=========================================================================================')


        ## In oder to minitor the train/test loss.
        ## In addition to the RMSE loss from testset for each tissue parameters.
        losses.append(batch_loss/ (i+1))
        loss_test.append(batch_loss_test/ (j+1))
        loss_K.append(batch_loss_K/ (j+1))
        loss_M.append(batch_loss_M/ (j+1))
        loss_T2m.append(batch_loss_T2m/ (j+1))
        loss_T1w.append(batch_loss_T1w/ (j+1))
        
        ## save model and every loss 
        ## save updated scan paramters for each epoch 
        ## RF_parameterN stands for the Nth updated scan parameter set.

        PATH=args.result+'/NN_model.pth'
        torch.save(fcnn.state_dict(), PATH)
        RF_parameters_save=RF_parameters_denorm.cpu()
        RF_parameters_save=RF_parameters_save.detach().numpy()

        sio.savemat(args.result+'/RF_parameters'+'{0:d}'.format(epoch+1)+'.mat',{'RF_parameters': RF_parameters_save.tolist()})
        sio.savemat(args.result+'/Trainloss'+'.mat',{'Trainloss': losses})
        sio.savemat(args.result+'/Testloss'+'.mat',{'Testloss': loss_test})
        sio.savemat(args.result+'/RMSE_K'+'.mat',{'RMSE_K': loss_K})
        sio.savemat(args.result+'/RMSE_M'+'.mat',{'RMSE_M': loss_M})
        sio.savemat(args.result+'/RMSE_T2m'+'.mat',{'RMSE_T2m': loss_T2m})
        sio.savemat(args.result+'/RMSE_T1w'+'.mat',{'RMSE_T1w': loss_T1w})
            




