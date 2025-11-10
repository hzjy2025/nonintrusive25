import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
import torch
import torch.nn.functional as F
import random
from random import shuffle
from Separate_LidDriven_nonlinear_bc_3NN_split import CustomedEqs, CustomedNet,p_normalize_mat
from Normalization import Normalization
from NN_bc_3NN_split import train, DEVICE, train_options_default
import numpy as np
from collections import OrderedDict
import torch.nn as nn
from Activations_plus import Swish
import torch.nn.init as init
torch.manual_seed(12)  
np.random.seed(1234)
ACTIVATE     = Swish
continue_tr = True

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

resi_lamda = 0
EPOCH   = int(3E4)
train_options_default['lamda'] = lambda epoch: 0.96**(epoch//1000)
NumSolsdir = 'NumSols_bc'

epoch_pre1 = int(3E5)
epoch_pre2 = int(3E5)
epoch_pre3 =int(3E5)

l_lid_Vec        = [20]
H1_lid_Vec         = [5]
# [5,10,15,20]
rvalue_lid_Vec        = [20]

SampleNum_Vec     = [100]
NetSize_Vec       = [100]
NResi_Vec         = [2000]
VarsRange_dict    = {'SampleNum': (SampleNum_Vec, SampleNum_Vec[0], ), \
                     'NetSize'  : (NetSize_Vec,   NetSize_Vec[0], ), \
                     'l'  : (l_lid_Vec,   l_lid_Vec[0], ), \
                     'H1'  : (H1_lid_Vec,   H1_lid_Vec[0], ), \
                     'rvalue'  : (rvalue_lid_Vec,   rvalue_lid_Vec[0], ), \
                     'NResi': (NResi_Vec,   NResi_Vec[0], )
                     }

Vars_dict         = {key:value[1] for key, value in VarsRange_dict.items()}

Nval = 100
grad_loss_lambda_old = 0.001
grad_loss_lambda = 0.001
label_grad_case = True
if label_grad_case:
    resultsdir = 'results_2dlid_3NN_split_bc_new_snap1000_timing'
else:
    resultsdir = 'results_2dlid_3NN_split_bc_new_snap1000_timing_nograd'

tune = False
resi_lamda = 0
if label_grad_case:
    resultsdir_save = 'results_2dlid_3NN_split_bc_new_snap1000_timing'
else:
    resultsdir_save = 'results_2dlid_3NN_split_bc_new_snap1000_timing_nograd'


class Reduce_Net(nn.Module):
    def __init__(self, layers2=None,OldNetfile2=None):
        super(Reduce_Net, self).__init__()
        self.lossfun = nn.MSELoss(reduction='mean')

        NetDict2 = OrderedDict()


        if not(layers2 or OldNetfile2):
            raise Exception('At least one of the parameters "Layers2" and "OldNerfile2" should be given ')
        if OldNetfile2:
            oldnet = torch.load(OldNetfile2, map_location=lambda storage, loc: storage)
            layers2 = oldnet['layers']
        self.layers2 = layers2
        for i in range(len(layers2)-1):
            key    = "Layer2%d_Linear"%i
            Value  = nn.Linear(layers2[i],layers2[i+1])
            init.xavier_uniform_(Value.weight)
            init.zeros_(Value.bias)
            NetDict2[key] = Value

            if i != len(layers2)-2:
                key    = "Layer2%d_avtivate"%i
                Value  = ACTIVATE()
                NetDict2[key] = Value
        self.unet2 = nn.Sequential(NetDict2)

    def forward(self,x):
        return self.unet2(x)

    def loadnet(self, OldNetfile):
        state_dict = torch.load(OldNetfile, map_location=lambda storage, loc: storage)['state_dict']
        self.load_state_dict(state_dict)
    def savenet(self, netfile):
        torch.save({'layers': self.layers2, 'state_dict': self.state_dict()}, netfile )

def Dict2Str(dicti):
    describes = ''
    for name in keys:
        val = dicti[name]
        if val == 80:
            val =10;
        describes += '_'+name+''+str(val)
    describes = 'LidDriven' + describes
    return describes

def GetLabelData(roeqs):
    labeled_inputs = roeqs.parameters
    labeled_outputs_p = roeqs.label_p.T
    labeled_outputs_u = roeqs.label_u.T
    labeled_outputs_v = roeqs.label_v.T
    return (labeled_inputs, labeled_outputs_p, labeled_outputs_u, labeled_outputs_v, 'Label',0.9,)

def GetResiData(roeqs,Np):
    Nin = roeqs.design_space.shape[1]
    Resi_inputs  = Normalization.Anti_Mapminmax(np.random.rand(Np,Nin)*2-1,  roeqs.design_space)
    dummy = np.zeros((Np,roeqs.M))
    return (Resi_inputs, dummy, 'Resi',0.9,)

def GetValData(roeqs,Nval):

    Val_inputs = roeqs.ValidationParameters
    arr = np.arange(Val_inputs.shape[0])
    #np.random.shuffle(arr)
    list_para = arr[:Nval]
    Val_exact =  roeqs.ValidationSamples
    p_exact = Val_exact[0::3,list_para]
    p_exact_norm = p_normalize_mat(p_exact,roeqs.N_1x,roeqs.N_1y)
    u_exact = Val_exact[1::3,list_para]
    v_exact = Val_exact[2::3,list_para]
    return Val_inputs[list_para,:], p_exact_norm, u_exact, v_exact

def GetHybridData(roeqs,Np):
    LabelData = GetLabelData(roeqs)
    ResiData  = GetResiData(roeqs,Np)
    return (LabelData[0], LabelData[1], LabelData[2], LabelData[3],ResiData[0],ResiData[1],'Hybrid',ResiData[3],)

class continue_testcases(object):
    def __init__(self, l, rvalue, H1num, ControlVarName = 'SampleNum',tune=False):
        self.ControlVarName = ControlVarName
        if  ControlVarName not in VarsRange_dict.keys():
            raise Exception('invalid var name')
        self.name = ControlVarName
        self.rvalue = rvalue
        self.H1num = H1num
        self.l = l
        self.Vals = VarsRange_dict[ControlVarName][0][0]
        self.tune = tune

    def gen_fun(self):
        localdict = Vars_dict.copy()

        localdict[self.ControlVarName] =  self.Vals

        localdict['l'] = self.l

        localdict['rvalue'] = self.rvalue

        localdict['H1']= self.H1num

        return localdict, Dict2Str(localdict)

    def Calculate(self, label_grad = False, grad_loss_lambda = 1e-4):
        losshistory = {}
        case = self.gen_fun()
        losshistory[case[1]+'train']=self.CaseSim(case, label_grad = label_grad, grad_loss_lambda = grad_loss_lambda)
        from scipy.io import savemat
        savemat('Test'+self.name+'_losshistory.mat', losshistory)
        return losshistory


    def CaseSim(self, case, label_grad = False, num_parabc = 10, grad_loss_lambda = 1e-4):

        matfilePOD        = NumSolsdir + '/LidDrivenPOD_rand_bc_snapshot_final1000.mat'
        matfileValidation = NumSolsdir + '/LidDriven_bc_Validation.mat'
        netfile1 = resultsdir  + '/'+            case[1]         +f"p{str(grad_loss_lambda_old).replace('.', '_').replace('-', '_')}.tar"
        netfile2 = resultsdir  + '/'+            case[1]         +f"u{str(grad_loss_lambda_old).replace('.', '_').replace('-', '_')}.tar"
        netfile3 = resultsdir  + '/'+            case[1]         +f"v{str(grad_loss_lambda_old).replace('.', '_').replace('-', '_')}.tar"

        netfile_save1 = resultsdir_save  + '/'+            case[1]             +f"p{str(grad_loss_lambda).replace('.', '_').replace('-', '_')}.tar"
        netfile_save2 = resultsdir_save  + '/'+            case[1]             +f"u{str(grad_loss_lambda).replace('.', '_').replace('-', '_')}.tar"
        netfile_save3 = resultsdir_save  + '/'+            case[1]             +f"v{str(grad_loss_lambda).replace('.', '_').replace('-', '_')}.tar"
        

        roeqs = CustomedEqs(matfilePOD,case[0]['SampleNum'],matfileValidation,case[0]['l'])
        
        layers1 = [num_parabc+2, *[ case[0]['NetSize'] ]*4, case[0]['rvalue']*case[0]['H1']+2*case[0]['H1']+1]

        layers2 = [case[0]['l'],case[0]['rvalue'],case[0]['rvalue']]

        Val_inputs, p_exact_norm, u_exact, v_exact = GetValData(roeqs,Nval)

        Val_inputs = torch.tensor(Val_inputs).float().to(DEVICE)

        options = train_options_default.copy()
        options['EPOCH'] = EPOCH
        
        data = GetHybridData(roeqs,case[0]['NResi'])
        options['weight_decay']=0
        options['NBATCH'] = 10

        reduce_net = Reduce_Net(layers2 = layers2).to(DEVICE)
        Net1 =CustomedNet(num_parabc,rvalue=case[0]['rvalue'], varaible = 'p', H1=case[0]['H1'],layers1=layers1, reduce_net=reduce_net,
                         roeqs=roeqs).to(DEVICE)

        
        reduce_net = Reduce_Net(layers2 = layers2).to(DEVICE)
        Net2 =CustomedNet(num_parabc,rvalue=case[0]['rvalue'], varaible = 'u', H1=case[0]['H1'],layers1=layers1, reduce_net=reduce_net,
                         roeqs=roeqs).to(DEVICE)

        

        reduce_net = Reduce_Net(layers2 = layers2).to(DEVICE)
        Net3 =CustomedNet(num_parabc,rvalue=case[0]['rvalue'], varaible = 'v', H1=case[0]['H1'],layers1=layers1, reduce_net=reduce_net,
                         roeqs=roeqs).to(DEVICE)       

        if self.tune:

            options['LR'] = 1e-06
            optimizer1 = torch.optim.Adam(Net1.parameters(), lr=options['LR'], weight_decay=options['weight_decay'])

            options['lamda'] = lambda epoch: 0.95**(epoch//1000)
            scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=options['lamda'])

            options['LR'] = 1e-06
            optimizer2 = torch.optim.Adam(Net2.parameters(), lr=options['LR'], weight_decay=options['weight_decay'])

            options['lamda'] = lambda epoch: 0.95**(epoch//1000)
            scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda=options['lamda'])

            options['LR'] = 1e-06
            optimizer3 = torch.optim.Adam(Net3.parameters(), lr=options['LR'], weight_decay=options['weight_decay'])

            options['lamda'] = lambda epoch: 0.95**(epoch//1000)
            scheduler3 = torch.optim.lr_scheduler.LambdaLR(optimizer3, lr_lambda=options['lamda'])

        print('Continue training')

        import time

        start_time = time.time()

        if self.tune:
        
            trainhistory=train(Net1,Net2,Net3,data, netfile_save1, netfile_save2, netfile_save3, Val_inputs, p_exact_norm, u_exact, v_exact, options=options, resi_lamda = resi_lamda, label_grad = label_grad,  optimizer_give=True, optimizer_given1=optimizer1, optimizer_given2=optimizer2, optimizer_given3=optimizer3,
              scheduler_give=True, scheduler_given1=scheduler1, scheduler_given2=scheduler2, scheduler_given3=scheduler3, epoch_last1 =epoch_pre1, epoch_last2 =epoch_pre2, epoch_last3 =epoch_pre3, epoch_grad = 20000, continue_tr = True, grad_loss_lambda = grad_loss_lambda,netfile1_old=netfile1,netfile2_old=netfile2,netfile3_old=netfile3)

        else:
            trainhistory=train(Net1,Net2,Net3,data, netfile_save1, netfile_save2, netfile_save3, Val_inputs, p_exact_norm, u_exact, v_exact, options=options, resi_lamda = resi_lamda, label_grad = label_grad, epoch_last1 =epoch_pre1, epoch_last2 =epoch_pre2, epoch_last3 =epoch_pre3, epoch_grad = 20000, continue_tr = True, grad_loss_lambda = grad_loss_lambda,netfile1_old=netfile1,netfile2_old=netfile2,netfile3_old=netfile3)

        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")

        return trainhistory

if __name__ == '__main__':
    for i in range(len(l_lid_Vec)):
        for j in range(len(rvalue_lid_Vec)):
            for k in range(len(H1_lid_Vec)):
                
                l_lid_new = l_lid_Vec[i]
                rvalue_new = rvalue_lid_Vec[j]
                H1_new = H1_lid_Vec[k]
                keys = list( Vars_dict.keys() )
                if continue_tr:
                    continue_testcases(l_lid_new,rvalue_new,H1_new,'SampleNum',tune=tune).Calculate(label_grad = label_grad_case, grad_loss_lambda = grad_loss_lambda)