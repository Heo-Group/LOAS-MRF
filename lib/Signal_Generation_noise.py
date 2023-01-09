import torch
import numpy as np
import math

### Signal_Gen_noise function generates MTC-MRF signals (input for FCNN)
### with respect to the tissue parameters and acquisition scan parameters 

def Signal_Gen_noise(tissue_param,RF_param,ds_num,device):
      
  pred_val=tissue_param
  B1_t=RF_param[:,0]
  PPM=RF_param[:,1]
  Tsat=RF_param[:,2]
  RdeadT=RF_param[:,3]
  T2_real = torch.div(pred_val[:,3],pred_val[:,4])  
  RMmRa = pred_val[:,0]*pred_val[:,1]*pred_val[:,3]  #R*C(Mn)*T1w 

  freqq=torch.tensor(2*math.pi,dtype = torch.float32,device=device,requires_grad=False) 
  gyr=torch.tensor(42.56,dtype = torch.float32,device=device,requires_grad=False)
  T1m=torch.tensor(1.0,dtype = torch.float32,device=device,requires_grad=False) 
  w0=torch.tensor(128.0,dtype = torch.float32,device=device,requires_grad=False) 
  
  
  length=pred_val.size()[0]
  Mns = torch.zeros([ds_num,length],device=device)

### Generate the MTC-MRF signal according to the given scan parameters ##########################
### Analytical solution of two-pool Bloch equation with scan parameters and tissue parameters ###
  for ind_i in range(ds_num):
      
    B1=B1_t[ind_i]
    w1=B1*freqq*gyr  
    ppm1=PPM[ind_i]
    freq1=ppm1*w0  
    Tsat1 = Tsat[ind_i]
    RDT = RdeadT[ind_i] 
            
    f_off = 2*math.pi*freq1 #offset freq of 3.5ppm
    R=pred_val[:,0]
    C=pred_val[:,1]
    T2m=pred_val[:,2]
    T1w=pred_val[:,3]
      
    Rrfb=(w1**2)*math.pi*(T2m/math.pi)*(1/(1+(f_off*T2m)**2))
    upper1= (RMmRa/T1m) + Rrfb + 1/T1m + R
    lower1= (RMmRa*(T1m+Rrfb)) + (1+((w1/f_off)**2)*(T1w/T2_real))*(T1m + Rrfb + R)
    Mss = torch.div(upper1,lower1)
    
    Beta1 = 1/T1w + ((w1**2)*T2_real)/(1 + (f_off*T2_real)**2) + R*C
    Beta2 = 1/T1m + ((w1**2)*T2m)/(1 + (f_off*T2m)**2) + R
    Beta3 = 0.5 * (torch.sqrt((Beta2 - Beta1)**2 + 4*C*(R**2)) - Beta1 - Beta2)
    
    Mns_tmp1 = Mss + ((1 - torch.exp(-RDT/T1w)) - Mss)*torch.exp(Beta3*Tsat1)
    Mns_tmp = torch.reshape(Mns_tmp1,(1,length))

    Mns[ind_i,:]=Mns_tmp

    
  Result = Mns
 
#### Add white gaussian noise to the MTC-MRF signals ###############
#### SNR(db) is given as follows, and it can be modified ###########
  target_snr_db = 46.021
  x_watts=Result**2
  sig_avg_watts=torch.mean(x_watts,-1)
  sig_avg_db=10*(torch.log(sig_avg_watts)/torch.log(torch.tensor(10,dtype = torch.float32)))

  noise_avg_db = sig_avg_db - target_snr_db
  noise_avg_watts = 10 ** (noise_avg_db / 10.)

  mean_noise = 0
  std_noise = torch.sqrt(noise_avg_watts)
  std_noise.unsqueeze_(-1)
  std_noise = std_noise.expand(ds_num,length)
  noise_volts = torch.normal(mean_noise,std_noise)

  y_volts= Result+noise_volts

#### Additionally add T2 signal to the MTC-MRF signals #########################################
#### When the dynamic scan number is N, the total signal length would be N+1 with T2 signal  ###
  T2_input=torch.reshape(T2_real,(1,length))
  T2_input=T2_input.clone()
  y_out=torch.cat((y_volts,T2_input),0)
  
  y_out = Result
  y_out=torch.transpose(y_out,0,1) 

  return y_out