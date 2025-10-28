import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
from Burger_nonlinear_output_no_norm import CustomedEqs, CustomedNet
from Normalization import Normalization
from NNburger_resi import train, DEVICE, train_options_default, evaluate
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.init as init
from Activations_plus import Swish

ACTIVATE     = Swish
torch.manual_seed(12)  # reproducible
np.random.seed(1234)

#
EPOCH   = int(10E4)
train_options_default['lamda'] = lambda epoch: 0.9**(epoch//2000)


Nval = 100

Fre_h = 9
SampleNum_Vec     = [40]

rlist             = [6,8]
#rlist             = [2,4,6,8,10]
#2,4,6,8,10  6,8,10,12,14,16,18,20
penalty_grad_case = False
label_grad_case = True
kernel_grad = False

### No label
strong_resi = False

if not(strong_resi):
    resi_lamda = 0
else:
    resi_lamda = 1.

if penalty_grad_case:
    resultsdir = f'resultsburger_high_fre_lbl_resi{resi_lamda}_penalize_gradient'

if label_grad_case:
    resultsdir = f'resultsburger_high_fre_lbl_resi{resi_lamda}_ladel_data_and_gradient'

if penalty_grad_case and label_grad_case:
    resultsdir = f'resultsburger_high_fre_lbl_resi{resi_lamda}_penalize_ladel_data_gradient'

if label_grad_case and kernel_grad:
    resultsdir = f'resultsburger_high_fre_lbl_resi{resi_lamda}_kernel_ladel_data_and_gradient'

if not(penalty_grad_case or label_grad_case or kernel_grad):
    resultsdir = f'resultsburger_high_fre_lbl_resi{resi_lamda}'

if strong_resi:
    resultsdir = 'strong_resi_off_tr_results'





class Reduce_Net(nn.Module):
    def __init__(self, layers2=None,OldNetfile2=None):
        super(Reduce_Net, self).__init__()
        self.lossfun = nn.MSELoss(reduction='mean')

        NetDict2 = OrderedDict()
        #NetDict['start'] = nn.Linear(1,1)


        if not(layers2 or OldNetfile2):
            raise Exception('At least one of the parameters "Layers2" and "OldNerfile2" should be given ')
        if OldNetfile2:
            oldnet = torch.load(OldNetfile2, map_location=lambda storage, loc: storage)
            layers2 = oldnet['layers']
        self.layers2 = layers2
        for i in range(len(layers2)-1):
            key    = "Layer2%d_Linear"%i
            Value  = nn.Linear(layers2[i],layers2[i+1])
            init.xavier_normal_(Value.weight)
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
        #self.load_state_dict(torch.load(OldNetfile)['state_dict'],  map_location=lambda storage, loc: storage)
        state_dict = torch.load(OldNetfile, map_location=lambda storage, loc: storage)['state_dict']
        self.load_state_dict(state_dict)
    def savenet(self, netfile):
        torch.save({'layers': self.layers2, 'state_dict': self.state_dict()}, netfile )

M_Vec             = [20]
# First NN structure [L,H1,1]
# fix Hidden layer to 1 neuron [L,H1,1], num of online weights = H1*L+H1+H1+1 = H1*L+2H1+1
onNN_H1_fix           = [1]
H1 = 1
# Second NN structure [L,L,1]
# loop over different online NN, [L,L,1], num of online weights = L^2+L+L+1 = L^2 + 2L + 1
onNN_H1_L           = M_Vec

###Nettype_Vec       = ['Label', 'Resi','Hybrid'][ist2:ien2]
###!!! SampleNum_Vec : how many snapshots
###!!! M_Vec : how many POD modes
###!!! Num_label_par1 and Num_label_par2 : how many label data to train offline  6,8,10,12,14,16,18,20


NetSize_Vec       = [100]
NResi_Vec         = [1000, 1500, 2000, 2500]
VarsRange_dict    = {'SampleNum': (SampleNum_Vec, 40, ), \
                     'NetSize'  : (NetSize_Vec,   100, ), \
                     'rvalue'  : (rlist,   10, ), \
                     'NResi': (NResi_Vec,  1000, )
                     }
#                     'NResi'    : (NResi_Vec,   5000, ), \
#                     }

Vars_dict         = {key:value[1] for key, value in VarsRange_dict.items()}
# fix svd rb basis functions to 10
Vars_dict['M']=20
Vars_dict['H1']=1
Vars_dict['Fre_h']=Fre_h
keys = list( Vars_dict.keys() )

def Dict2Str(dicti):
    describes = ''
    for name in keys:
        val = dicti[name]
        describes += '_'+name+''+str(val)
    describes = 'Burges1D' + describes
    return describes

class gen_testcases_fixH1(object):
    def __init__(self, ControlVarName = 'SampleNum',lamda=1.):
        self.ControlVarName = ControlVarName
        if  ControlVarName not in VarsRange_dict.keys():
            raise Exception('invalid var name')
        self.name = ControlVarName
        self.Vals = VarsRange_dict[ControlVarName][0]
        self.lamda_label = lamda
    def __iter__(self):
        return self.gen_fun()

    def gen_fun(self):
        localdict = Vars_dict.copy()
        for val in VarsRange_dict[self.ControlVarName][0]:
            localdict[self.ControlVarName] = val
# fix M = Vars_dict['M']=10
            for rvalue in rlist:
                localdict['rvalue'] = rvalue
                for H1num in onNN_H1_fix:
                    localdict['H1']= H1num
                    yield localdict, Dict2Str(localdict)

    def Calculate(self, penalty_grad = False, label_grad = False, kernel_grad = False, strong_resi = False):
        losshistory = {}
        for case in self:
            print(case[1])
            losshistory[case[1]+'train'], losshistory[case[1]+'test']=self.CaseSim(case, penalty_grad = penalty_grad, label_grad = label_grad, kernel_grad = kernel_grad, strong_resi = strong_resi)
        from scipy.io import savemat
        savemat('Test'+self.name+'_losshistory.mat', losshistory)
        return losshistory

# After Calculate(self), all trained network saved in netfile, then load netfile to predict solution for mu in
# validation set (final error check set)


    def CaseSim(self, case, penalty_grad = False, label_grad = False, kernel_grad = False, strong_resi = False):

        # online network mu to online NN weights
        netfile1 = resultsdir  + '/'+            case[1]         +'labeldata_50'    +'.tar'

        if penalty_grad:

            netfile1 = resultsdir_grad  + '/'+            case[1]         +'labeldata_50'    +'.tar'

        if os.path.isfile(netfile1):
            pass

            #continue

            ##################################
            ################################## 1st place to tune the frequency h
        roeqs = CustomedEqs(case[0]['M'], h=Fre_h, Num_par1 = case[0]['SampleNum'], Num_par2 = case[0]['SampleNum']
                            , Num_label_par1 = 50, Num_label_par2 = 50)
        data = GetHybridData(roeqs,case[0]['NResi'])
        labeled_outputs_min = data[3]
        labeled_outputs_max = data[4]

        # from 2 mu to r*H1+2H1+1 outputs
        layers1 = [2, *[ case[0]['NetSize'] ]*4, case[0]['rvalue']*case[0]['H1']+2*case[0]['H1']+1]
        layers2 = [case[0]['M'],case[0]['rvalue'],case[0]['rvalue']]
        reduce_net = Reduce_Net(layers2 = layers2)

        Net =CustomedNet(case[0]['rvalue'],case[0]['H1'], #labeled_outputs_min,labeled_outputs_max,
                         layers1=layers1,reduce_net=reduce_net, roeqs=roeqs).to(DEVICE)

        #Net.initial_w()
        np.random.seed(1234)
        Val_inputs, Val_source, Val_exact = GetValData(roeqs,Nval)
        Val_inputs = torch.tensor(Val_inputs).float().to(DEVICE)
        Val_source = torch.tensor(Val_source).float().to(DEVICE)
        #Val_exact = torch.tensor(Val_exact).float().to(DEVICE)
        options = train_options_default.copy()
        options['EPOCH'] = EPOCH

        print(data[2])
        options['weight_decay']=0
        if strong_resi:
            options['NBATCH'] = 10
            trainhistory, testhistory, loss_history_resi=train_strong(Net,data, netfile1, Val_inputs, Val_source, Val_exact, lamda=self.lamda_label, options=options, resi_lamda = resi_lamda, penalty_grad = penalty_grad, label_grad = label_grad, kernel_grad = kernel_grad)

        else:
            if penalty_grad:
                options['NBATCH'] = 10
            else:
                options['NBATCH'] = 1
            trainhistory, testhistory, loss_history_resi=train(Net,data, netfile1, Val_inputs, Val_source, Val_exact, lamda=self.lamda_label, options=options, resi_lamda = resi_lamda, penalty_grad = penalty_grad, label_grad = label_grad, kernel_grad = kernel_grad)


        resi_save = resultsdir  + '/'+            case[1]             +'resi.npz'
        np.savez(resi_save, loss_history_resi)
        return trainhistory, testhistory



def GetLabelData(roeqs):
    labeled_inputs = roeqs.parameters_label
    labeled_outputs_normalize = roeqs.Samples_label_normalize
    labeled_outputs_min = roeqs.Samples_min_val
    labeled_outputs_max = roeqs.Samples_max_val
    labeled_outputs= roeqs.Samples_label.T
    return (labeled_inputs, labeled_outputs, labeled_outputs_normalize, labeled_outputs_min, labeled_outputs_max, 'Label',0.9,)

def GetResiData(roeqs,Np):
    Nin = roeqs.design_space.shape[1]
    Resi_inputs  = Normalization.Anti_Mapminmax(np.random.rand(Np,Nin)*2-1,  roeqs.design_space)
    alpha1 = Resi_inputs[:,0:1]
    alpha2 = Resi_inputs[:,1:2]
    Resi_source = roeqs.getsource(alpha1, alpha2)
    return (Resi_inputs, Resi_source, 'Resi',0.9,)

def GetValData(roeqs,Nval):
    Nin = roeqs.design_space.shape[1]
    Val_inputs  = Normalization.Anti_Mapminmax(np.random.rand(Nval,Nin)*2-1,  roeqs.design_space)
    alpha1 = Val_inputs[:,0:1]
    alpha2 = Val_inputs[:,1:2]
    Val_source = roeqs.getsource(alpha1, alpha2)
    Val_exact = roeqs.phix(roeqs.xgrid.T,alpha1,alpha2)
    return Val_inputs, Val_source, Val_exact

def GetHybridData(roeqs,Np):
    LabelData = GetLabelData(roeqs)
    ResiData  = GetResiData(roeqs,Np)
    return (LabelData[0], LabelData[1],LabelData[2], LabelData[3], LabelData[4],ResiData[0],ResiData[1],'Hybrid',ResiData[3],)

class continue_train_fixH1(object):
    def __init__(self, rvalue, H1, ControlVarName, ControlVar,lamda=1.,tune=False):
        self.ControlVarName = ControlVarName
        if  ControlVarName not in VarsRange_dict.keys():
            raise Exception('invalid var name')
        self.name = ControlVarName
        self.ControlVar = ControlVar
        self.rvalue = rvalue
        self.H1num = H1
        self.lamda_label = lamda
        self.tune = tune

        ###self.Vals = VarsRange_dict[ControlVarName][0]
    def __iter__(self):
        return self.gen_fun()

    def gen_fun(self):
        localdict = Vars_dict.copy()

        localdict[self.ControlVarName] = self.ControlVar
        localdict['rvalue'] = self.rvalue
        localdict['H1']= self.H1num
        yield localdict, Dict2Str(localdict)

    def Calculate(self, penalty_grad = False, label_grad = False, kernel_grad = False, strong_resi = False):
        losshistory = {}
        for case in self:
            print(case[1])
            losshistory[case[1]+'train'], losshistory[case[1]+'test']=self.CaseSim(case, penalty_grad = penalty_grad, label_grad = label_grad, kernel_grad = kernel_grad, strong_resi = strong_resi)
            break
        from scipy.io import savemat
        savemat('Test'+self.name+'_losshistory.mat', losshistory)
        return losshistory

# After Calculate(self), all trained network saved in netfile, then load netfile to predict solution for mu in
# validation set (final error check set)

    def CaseSim(self, case, penalty_grad = False, label_grad = False, kernel_grad = False, strong_resi = False):
        # online network mu to online NN weights
        netfile1 = resultsdir  + '/'+            case[1]         +'labeldata_50'    +'.tar'
        if penalty_grad:

            netfile1 = resultsdir_grad  + '/'+            case[1]         +'labeldata_50'    +'.tar'

        if os.path.isfile(netfile1):
            print('Continue training')
            pass

            #continue

            #continue

            ##################################
            ################################## 1st place to tune the frequency h
        roeqs = CustomedEqs(case[0]['M'], h=Fre_h, Num_par1 = case[0]['SampleNum'], Num_par2 = case[0]['SampleNum']
                            , Num_label_par1 = 50, Num_label_par2 = 50)
        data = GetHybridData(roeqs,case[0]['NResi'])
        labeled_outputs_min = data[3]
        labeled_outputs_max = data[4]

        # store Reduce_Net also
        layers1 = [2, *[ case[0]['NetSize'] ]*4, case[0]['rvalue']*case[0]['H1']+2*case[0]['H1']+1]
        layers2 = [case[0]['M'],case[0]['rvalue'],case[0]['rvalue']]
        reduce_net = Reduce_Net(layers2 = layers2)

        Net =CustomedNet(case[0]['rvalue'],case[0]['H1']     #,labeled_outputs_min,labeled_outputs_max
                         ,layers1=layers1,reduce_net=reduce_net, roeqs=roeqs).to(DEVICE)
        np.random.seed(1234)
        Val_inputs, Val_source, Val_exact = GetValData(roeqs,Nval)
        Val_inputs = torch.tensor(Val_inputs).float().to(DEVICE)
        Val_source = torch.tensor(Val_source).float().to(DEVICE)
        #Val_exact = torch.tensor(Val_exact).float().to(DEVICE)
        options = train_options_default.copy()
        checkpoint = torch.load(netfile1,map_location=torch.device('cpu'))
        epoch_pre = checkpoint['epoch']
        Net.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(Net.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=options['lamda'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(checkpoint['optimizer_state_dict'])

        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(checkpoint['scheduler_state_dict'])


        print(epoch_pre)
        options['weight_decay']=0
        

        ######If find lr is so slow, use this part, otherwise do not run
        #######!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.tune:
            #options['LR'] = 0.001*0.95**(epoch_pre//2000)
            options['LR'] = 2.941878337474921e-05
            optimizer = torch.optim.Adam(Net.parameters(), lr=options['LR'], weight_decay=options['weight_decay'])

            options['lamda'] = lambda epoch: 0.95**(epoch//5000)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=options['lamda'])


        if strong_resi:
            options['NBATCH'] = 10
            trainhistory, testhistory, loss_history_resi=train_strong(Net,data, netfile1, Val_inputs, Val_source, Val_exact,  optimizer_give=True, optimizer_given=optimizer,
          scheduler_give=True, scheduler_given=scheduler, epoch_last = epoch_pre, lamda=self.lamda_label, options=options, resi_lamda = resi_lamda, penalty_grad = penalty_grad, label_grad = label_grad, kernel_grad = kernel_grad)

        else:
            if penalty_grad:
                options['NBATCH'] = 10
            else:
                options['NBATCH'] = 1
            trainhistory, testhistory, loss_history_resi=train(Net,data, netfile1, Val_inputs, Val_source, Val_exact,  optimizer_give=True, optimizer_given=optimizer,
          scheduler_give=True, scheduler_given=scheduler, epoch_last = epoch_pre, lamda=self.lamda_label, options=options, resi_lamda = resi_lamda, penalty_grad = penalty_grad, label_grad = label_grad, kernel_grad = kernel_grad)


        return trainhistory, testhistory


#######use xavier uniform, penalty_grad = True
if __name__ == '__main__':

    continue_train = True

    if continue_train:

        for r in rlist:
            for sam in SampleNum_Vec:
                continue_train_fixH1(rvalue=r, H1=H1, ControlVarName='SampleNum', ControlVar=sam,lamda=1,tune=True).Calculate(penalty_grad = penalty_grad_case, label_grad=label_grad_case)


    else:
        if strong_resi:
    
            gen_testcases_fixH1('SampleNum',lamda=0).Calculate(penalty_grad = penalty_grad_case, label_grad=label_grad_case)
            
        else:
            
            gen_testcases_fixH1('SampleNum',lamda=1).Calculate(penalty_grad = penalty_grad_case, label_grad=label_grad_case)



