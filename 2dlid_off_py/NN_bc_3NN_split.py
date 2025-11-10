import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
from Clenshaw_Curtis import clenshaw_curtis_quadrature_2d_discrete, clenshaw_curtis_quadrature_2d_torch
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.utils.data as Data
from collections import OrderedDict
from Activations_plus import Swish
import torch.nn.init as init

ACTIVATE     = Swish
torch.manual_seed(12)  
np.random.seed(1234)
DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_options_default ={'EPOCH':50000,\
                        'LR':0.001, \
                        'lamda': lambda epoch: 0.95**(epoch//2000),\
                        'epoch_print': 5000,\
                        'epoch_save':5000,
                        'epoch_error_check':5000
                        }



class POD_Net(nn.Module):
    def __init__(self, layers1=None,OldNetfile1=None):
        super(POD_Net, self).__init__()
        self.lossfun = nn.MSELoss(reduction='mean')
        NetDict1 = OrderedDict()
        if not(layers1 or OldNetfile1):
            raise Exception('At least one of the parameters "Layers" and "OldNerfile" should be given ')
        if OldNetfile1:
            oldnet = torch.load(OldNetfile1, map_location=lambda storage, loc: storage)
            layers1 = oldnet['layers']
        self.layers1 = layers1
        for i in range(len(layers1)-1):
            key    = "Layer%d_Linear"%i
            Value  = nn.Linear(layers1[i],layers1[i+1])
            init.xavier_uniform_(Value.weight)
            init.zeros_(Value.bias)
            NetDict1[key] = Value

            if i != len(layers1)-2:
                key    = "Layer%d_avtivate"%i
                Value  = ACTIVATE()
                NetDict1[key] = Value
        self.unet = nn.Sequential(NetDict1)

    def grad(self,a,b):
        if b.grad is not None:
            b.grad.zero_()
        da_db = ag.grad(a, b, None,
                   create_graph=True, retain_graph=True)[0]
        return da_db
    @staticmethod
    def u_net(self,x):
        pass

    @staticmethod
    def forward(self,x):
        pass
        
    @staticmethod
    def loss_NN(self, xlabel, ylabel):
            pass
    @staticmethod
    def loss_PINN(self, x, f):
        pass
    @staticmethod
    def loss_Strong_Residue(self,x):
        pass
    @staticmethod
    def loss_label(self, x, p_star_norm, u_star, v_star):
        pass

    def loadnet(self, OldNetfile1):

        state_dict = torch.load(OldNetfile1, map_location=lambda storage, loc: storage)['state_dict']
        self.load_state_dict(state_dict)
    def savenet(self, netfile):
        torch.save({'layers': self.layers1, 'state_dict': self.state_dict()}, netfile )


def train(Net1,Net2,Net3,data, netfile1, netfile2, netfile3, Val_inputs, p_exact_norm, u_exact, v_exact, optimizer_give = False, optimizer_given1 = None, optimizer_given2 = None, optimizer_given3 = None,
          scheduler_give = False, scheduler_given1 = None, scheduler_given2 = None, scheduler_given3 = None, epoch_last1 =0, epoch_last2 =0, epoch_last3 =0,  options=train_options_default, netfilesave_provide=False, netfilesave=None, resi_lamda = 0., label_grad = False, epoch_grad=50000, grad_loss_lambda=1e-6,continue_tr = False,netfile1_old = None,netfile2_old = None,netfile3_old = None):

    """
    Inputs:
    netfile1, netfile2, netfile3 is the file to save the trained network
    netfile1_old are old netfile, load and continue to train from it
    
    """
    
    if len(data) == 8:
        labeled_inputs, labeled_outputs_p, labeled_outputs_u, labeled_outputs_v, inputs, outputs,datatype,trainratio= data

    else:
        raise Exception('Expect inout <data> a tuple with 8 elements, but got %d'%len(data))

    num_label = labeled_inputs.shape[0]

    labeled_outputs_p_int =labeled_outputs_p[:,Net1.roeqs.Nh_list_in_order]

    if netfilesave_provide:
        netfile1 = netfilesave

    label_p_x = Net1.roeqs.Compute_dp_dxc(labeled_outputs_p.reshape((labeled_outputs_p.shape[0],Net1.roeqs.Np_1d+1,Net1.roeqs.Np_1d+1)))
    label_p_x = torch.tensor(label_p_x[:,1:-1,1:-1].reshape((num_label,-1))).float().to(DEVICE)
    label_p_y = Net1.roeqs.Compute_dp_dyc_label(labeled_outputs_p.reshape((labeled_outputs_p.shape[0],Net1.roeqs.Np_1d+1,Net1.roeqs.Np_1d+1)))
    label_p_y = torch.tensor(label_p_y[:,1:-1,1:-1].reshape((num_label,-1))).float().to(DEVICE)
    label_u_x = Net2.roeqs.Compute_d_dxc(labeled_outputs_u.reshape((labeled_outputs_p.shape[0],Net2.roeqs.Np_1d+1,Net2.roeqs.Np_1d+1)))
    label_u_x = torch.tensor(label_u_x[:,1:-1,1:-1].reshape((num_label,-1))).float().to(DEVICE)
    label_u_y = Net2.roeqs.Compute_d_dyc_label(labeled_outputs_u.reshape((labeled_outputs_p.shape[0],Net2.roeqs.Np_1d+1,Net2.roeqs.Np_1d+1)))
    label_u_y = torch.tensor(label_u_y[:,1:-1,1:-1].reshape((num_label,-1))).float().to(DEVICE)
    label_v_x = Net3.roeqs.Compute_d_dxc(labeled_outputs_v.reshape((labeled_outputs_p.shape[0],Net3.roeqs.Np_1d+1,Net3.roeqs.Np_1d+1)))
    label_v_x = torch.tensor(label_v_x[:,1:-1,1:-1].reshape((num_label,-1))).float().to(DEVICE)
    label_v_y = Net3.roeqs.Compute_d_dyc_label(labeled_outputs_v.reshape((labeled_outputs_p.shape[0],Net3.roeqs.Np_1d+1,Net3.roeqs.Np_1d+1)))
    label_v_y = torch.tensor(label_v_y[:,1:-1,1:-1].reshape((num_label,-1))).float().to(DEVICE)

    num_parabc = inputs.shape[1] - 2

    off_resi_jac_p,off_resi_hes_p = Net1.compute_off_all(torch.tensor(inputs).float().to(DEVICE))
    off_resi_jac_u,off_resi_hes_u = Net2.compute_off_all(torch.tensor(inputs).float().to(DEVICE))
    off_resi_jac_v,off_resi_hes_v = Net3.compute_off_all(torch.tensor(inputs).float().to(DEVICE))

    off_label_jac_p,off_label_hes_p = Net1.compute_off_all_label(torch.tensor(labeled_inputs).float().to(DEVICE))
    off_label_jac_u,off_label_hes_u = Net2.compute_off_all_label(torch.tensor(labeled_inputs).float().to(DEVICE))
    off_label_jac_v,off_label_hes_v = Net3.compute_off_all_label(torch.tensor(labeled_inputs).float().to(DEVICE))

    #  [x_num_int,L,2]
    #  [x_num_int,L,2,2]
    p_off_jac_ksi = torch.tensor( Net1.off_jac_ksi[Net1.roeqs.Nh_list_in_order,:,:] ).float().to(DEVICE)
    p_off_hes_ksi = torch.tensor( Net1.off_hes_ksi[Net1.roeqs.Nh_list_in_order,:,:,:] ).float().to(DEVICE)
    u_off_jac_ksi = torch.tensor( Net2.off_jac_ksi[Net2.roeqs.Nh_list_in_order,:,:] ).float().to(DEVICE)
    u_off_hes_ksi = torch.tensor( Net2.off_hes_ksi[Net2.roeqs.Nh_list_in_order,:,:,:] ).float().to(DEVICE)
    v_off_jac_ksi = torch.tensor( Net3.off_jac_ksi[Net3.roeqs.Nh_list_in_order,:,:] ).float().to(DEVICE)
    v_off_hes_ksi = torch.tensor( Net3.off_hes_ksi[Net3.roeqs.Nh_list_in_order,:,:,:] ).float().to(DEVICE)
    
    #  [num_label,x_num_int,L,2]
    #  [num_label,x_num_int,L,2,2]
    off_ksi_label_jac_p = p_off_jac_ksi.unsqueeze(0).expand(num_label, -1, -1, -1).clone()
    off_ksi_label_hes_p = p_off_hes_ksi.unsqueeze(0).expand(num_label, -1, -1, -1, -1).clone()
    off_ksi_label_jac_u = u_off_jac_ksi.unsqueeze(0).expand(num_label, -1, -1, -1).clone()
    off_ksi_label_hes_u = u_off_hes_ksi.unsqueeze(0).expand(num_label, -1, -1, -1, -1).clone()
    off_ksi_label_jac_v = v_off_jac_ksi.unsqueeze(0).expand(num_label, -1, -1, -1).clone()
    off_ksi_label_hes_v = v_off_hes_ksi.unsqueeze(0).expand(num_label, -1, -1, -1, -1).clone()


    if np.mean(inputs[:,1]) < 10:
        theta_resi = inputs[:,1:2]
    else:
        theta_resi = 2*np.pi*inputs[:,1:2]/360


    labeled_inputs  = torch.tensor(labeled_inputs).float().to(DEVICE)
    labeled_outputs_p_int = torch.tensor(labeled_outputs_p_int).float().to(DEVICE)
    labeled_outputs_u = torch.tensor(labeled_outputs_u).float().to(DEVICE)
    labeled_outputs_v = torch.tensor(labeled_outputs_v).float().to(DEVICE)


    dataset   = Data.TensorDataset(labeled_inputs,\
                                       labeled_outputs_p_int,\
                                       labeled_outputs_u,\
                                       labeled_outputs_v,\
                                       label_p_x,\
                                       label_p_y,\
                                       label_u_x,\
                                       label_u_y,\
                                       label_v_x,\
                                       label_v_y,\
                                       off_ksi_label_jac_p,\
                                       off_ksi_label_hes_p,\
                                       off_ksi_label_jac_u,\
                                       off_ksi_label_hes_u,\
                                       off_ksi_label_jac_v,\
                                       off_ksi_label_hes_v,\
                                       )
    trainsize =  int(labeled_inputs.shape[0] * trainratio)
    testsize  =  labeled_inputs.shape[0] - trainsize
    trainset, testset = Data.random_split(dataset, [trainsize, testsize])
    trainloader = Data.DataLoader(trainset, batch_size= trainsize//options['NBATCH'], shuffle = True)
    testloader  = Data.DataLoader(testset , batch_size= testsize                    , shuffle = True) 

    NBatch = len(trainloader)


    def lossHybrid_nonli1(lamba4=1.):

        loss = 0.

        if lamba4 != 0: 
            
            loss21 = Net1.loss_label(labeled_inputs_batch, labeled_outputs_p_int_batch)
            
            loss  += lamba4*(loss21)

        if label_grad and epoch >= epoch_grad:

            loss41 = Net1.loss_label_grad(labeled_inputs_batch,off_ksi_label_jac_p_batch,off_ksi_label_hes_p_batch,label_p_x_batch,label_p_y_batch)
            if epoch%epoch_grad == 0:

                print("|epoch=%5d | solution loss=(%11.7e), gradient loss=(%11.7e)"%(epoch,loss21,loss41))
            
            loss = loss + grad_loss_lambda*(loss41)

        elif label_grad and continue_tr:
            loss41 = Net1.loss_label_grad(labeled_inputs_batch,off_ksi_label_jac_p_batch,off_ksi_label_hes_p_batch,label_p_x_batch,label_p_y_batch)
            if epoch%epoch_grad == 0:

                print("|epoch=%5d | solution loss=(%11.7e), gradient loss=(%11.7e)"%(epoch,loss21,loss41))
            
            loss = loss + grad_loss_lambda*(loss41)

            
        return loss

    def lossHybrid_nonli2(lamba4=1.):

        loss = 0.

        if lamba4 != 0: 
            
            loss22 = Net2.loss_label(labeled_inputs_batch, labeled_outputs_u_batch)

            loss  += lamba4*(loss22)

        if label_grad and epoch >= epoch_grad:

            
            loss42 = Net2.loss_label_grad(labeled_inputs_batch,off_ksi_label_jac_u_batch,off_ksi_label_hes_u_batch,label_u_x_batch,label_u_y_batch)

            if epoch%epoch_grad == 0:

                print("|epoch=%5d | solution loss=(%11.7e), gradient loss=(%11.7e)"%(epoch,loss22,loss42))

            loss = loss + grad_loss_lambda*loss42

        elif label_grad and continue_tr:

            loss42 = Net2.loss_label_grad(labeled_inputs_batch,off_ksi_label_jac_u_batch,off_ksi_label_hes_u_batch,label_u_x_batch,label_u_y_batch)

            if epoch%epoch_grad == 0:

                print("|epoch=%5d | solution loss=(%11.7e), gradient loss=(%11.7e)"%(epoch,loss22,loss42))

            loss = loss + grad_loss_lambda*loss42

            
        return loss
    def lossHybrid_nonli3(lamba4=1.):

        loss = 0.

        if lamba4 != 0: 
                        
            loss23 = Net3.loss_label(labeled_inputs_batch, labeled_outputs_v_batch)
            loss  += lamba4*(loss23)

        if label_grad and epoch >= epoch_grad:
         
            loss43 = Net3.loss_label_grad(labeled_inputs_batch,off_ksi_label_jac_v_batch,off_ksi_label_hes_v_batch,label_v_x_batch,label_v_y_batch)
            if epoch%epoch_grad == 0:

                print("|epoch=%5d | solution loss=(%11.7e), gradient loss=(%11.7e)"%(epoch,loss23,loss43))
            loss = loss + grad_loss_lambda*(loss43)

        elif label_grad and continue_tr:
            loss43 = Net3.loss_label_grad(labeled_inputs_batch,off_ksi_label_jac_v_batch,off_ksi_label_hes_v_batch,label_v_x_batch,label_v_y_batch)
            if epoch%epoch_grad == 0:

                print("|epoch=%5d | solution loss=(%11.7e), gradient loss=(%11.7e)"%(epoch,loss23,loss43))
            loss = loss + grad_loss_lambda*(loss43)
           
        return loss


    
    def closure1():
        optimizer1.zero_grad()
        loss = lossHybrid_nonli1()
        loss.backward()
        return loss

    def closure2():
        optimizer2.zero_grad()
        loss = lossHybrid_nonli2()
        loss.backward()
        return loss

    def closure3():
        optimizer3.zero_grad()
        loss = lossHybrid_nonli3()
        loss.backward()
        return loss


    loss_history_train = np.zeros((options['EPOCH']*3, 1))

    if continue_tr:

        checkpoint1 = torch.load(netfile1_old,map_location=torch.device('cpu'))
        epoch_pre1 = checkpoint1['epoch']
        Net1.load_state_dict(checkpoint1['Net1_state_dict'])
        optimizer1 = torch.optim.Adam(Net1.parameters())
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=options['lamda'])
        optimizer1.load_state_dict(checkpoint1['optimizer_state_dict'])
        scheduler1.load_state_dict(checkpoint1['scheduler_state_dict'])

    else:
        optimizer1 = torch.optim.Adam(Net1.parameters(), lr=options['LR'], weight_decay=options['weight_decay'])
        optimizer2 = torch.optim.Adam(Net2.parameters(), lr=options['LR'], weight_decay=options['weight_decay'])
        optimizer3 = torch.optim.Adam(Net3.parameters(), lr=options['LR'], weight_decay=options['weight_decay'])
    
        scheduler1 = torch.optim.lr_scheduler.LambdaLR(optimizer1, lr_lambda=options['lamda'])
        scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda=options['lamda'])
        scheduler3 = torch.optim.lr_scheduler.LambdaLR(optimizer3, lr_lambda=options['lamda'])


    if optimizer_give and scheduler_give:
        optimizer1 = optimizer_given1
        scheduler1 = scheduler_given1


    relative_exact_p = np.sqrt(clenshaw_curtis_quadrature_2d_discrete((p_exact_norm.T.reshape(p_exact_norm.shape[1],Net1.roeqs.N_1x-2,Net1.roeqs.N_1y-2))**2, Net1.roeqs.N_1x, Net1.roeqs.N_1y, discard_endpoints=True))

    relative_exact_u = np.sqrt(clenshaw_curtis_quadrature_2d_discrete((u_exact.T.reshape(p_exact_norm.shape[1],Net2.roeqs.N_1x-2,Net2.roeqs.N_1y-2))**2, Net2.roeqs.N_1x, Net2.roeqs.N_1y, discard_endpoints=True))

    relative_exact_v = np.sqrt(clenshaw_curtis_quadrature_2d_discrete((v_exact.T.reshape(p_exact_norm.shape[1],Net3.roeqs.N_1x-2,Net3.roeqs.N_1y-2))**2, Net3.roeqs.N_1x, Net3.roeqs.N_1y, discard_endpoints=True))

    pdiff_norm = Net1.error_compute(Val_inputs,p_exact_norm, u_exact, v_exact, relative_exact_p, relative_exact_u, relative_exact_v)
    Errorpuv = np.mean(pdiff_norm/relative_exact_p)

    print("|starting | p_error=(%11.7e)"%(Errorpuv))
    loss_history_test = np.zeros((options['EPOCH'], 1))
    for epoch in range(options['EPOCH']):
        Net1.train()
       
        loss_train = 0
        loss_test =0
        for nbatch, (labeled_inputs_batch,labeled_outputs_p_int_batch, labeled_outputs_u_batch, labeled_outputs_v_batch, label_p_x_batch, label_p_y_batch,label_u_x_batch,label_u_y_batch,label_v_x_batch,label_v_y_batch,off_ksi_label_jac_p_batch, off_ksi_label_hes_p_batch, off_ksi_label_jac_u_batch, off_ksi_label_hes_u_batch, off_ksi_label_jac_v_batch, off_ksi_label_hes_v_batch) in enumerate(trainloader):      
            running_loss=optimizer1.step(closure1)
            loss_train += running_loss 
            with torch.no_grad():
                for nbatch2, (labeled_inputs_test,labeled_outputs_p_int_test, labeled_outputs_u_test, labeled_outputs_v_test, label_p_x_test, label_p_y_test,label_u_x_test,label_u_y_test,label_v_x_test,label_v_y_test,off_ksi_label_jac_p_test, off_ksi_label_hes_p_test, off_ksi_label_jac_u_test, off_ksi_label_hes_u_test, off_ksi_label_jac_v_test, off_ksi_label_hes_v_test) in enumerate(testloader):  
                    loss_test   += Net1.loss_label(labeled_inputs_test, labeled_outputs_p_int_test)
            if epoch%options['epoch_print'] == 0 or epoch == options['EPOCH']-1:
                print("|epoch=%5d,nbatch=%5d | loss=(%11.7e,  %11.7e)"%(epoch,nbatch,running_loss,loss_test)) 
        loss_history_train[epoch,0] = loss_train/NBatch


        scheduler1.step()
        if epoch % options['epoch_save'] == 0 or epoch == options['EPOCH']-1:
            torch.save({
                'epoch':epoch+epoch_last1,
                'Net1_state_dict': Net1.state_dict(),
                'optimizer_state_dict': optimizer1.state_dict(),
                'scheduler_state_dict': scheduler1.state_dict(),
            }, netfile1)

        
        if epoch % options['epoch_error_check'] == 0 or epoch == options['EPOCH']-1:
            Net1.eval()  
            
            with torch.no_grad():
                pdiff_norm = Net1.error_compute(Val_inputs,p_exact_norm, u_exact, v_exact, relative_exact_p, relative_exact_u, relative_exact_v)
           
                Errorpuv = np.mean(pdiff_norm/relative_exact_p)

                print("|epoch=%5d | p_error=(%11.7e)"%(epoch,Errorpuv))

        loss_history_test[epoch,0] = loss_test/NBatch

        if epoch>=20000:
            if np.all( loss_history_test[epoch-2000:epoch+1,0]>loss_history_test[epoch-2001:epoch,0]):
                break



    
    if continue_tr:
        checkpoint2 = torch.load(netfile2_old,map_location=torch.device('cpu'))
        epoch_pre2 = checkpoint2['epoch']
        Net2.load_state_dict(checkpoint2['Net2_state_dict'])
        optimizer2 = torch.optim.Adam(Net2.parameters())
        scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer2, lr_lambda=options['lamda'])
        optimizer2.load_state_dict(checkpoint2['optimizer_state_dict'])
        scheduler2.load_state_dict(checkpoint2['scheduler_state_dict'])


    if optimizer_give and scheduler_give:
        optimizer2 = optimizer_given2
        scheduler2 = scheduler_given2
        
    udiff_norm = Net2.error_compute(Val_inputs,p_exact_norm, u_exact, v_exact, relative_exact_p, relative_exact_u, relative_exact_v)
            
    Errorpuv = np.mean(udiff_norm/relative_exact_u)

    print("|starting | u_error=(%11.7e)"%(Errorpuv))
    loss_history_test = np.zeros((options['EPOCH'], 1))
    for epoch in range(options['EPOCH']):
        Net2.train()
       
        loss_train = 0
        loss_test =0
        for nbatch, (labeled_inputs_batch,labeled_outputs_p_int_batch, labeled_outputs_u_batch, labeled_outputs_v_batch, label_p_x_batch, label_p_y_batch,label_u_x_batch,label_u_y_batch,label_v_x_batch,label_v_y_batch,off_ksi_label_jac_p_batch, off_ksi_label_hes_p_batch, off_ksi_label_jac_u_batch, off_ksi_label_hes_u_batch, off_ksi_label_jac_v_batch, off_ksi_label_hes_v_batch) in enumerate(trainloader):      
            running_loss=optimizer2.step(closure2)
            loss_train += running_loss 
            with torch.no_grad():
                for nbatch2, (labeled_inputs_test,labeled_outputs_p_int_test, labeled_outputs_u_test, labeled_outputs_v_test, label_p_x_test, label_p_y_test,label_u_x_test,label_u_y_test,label_v_x_test,label_v_y_test,off_ksi_label_jac_p_test, off_ksi_label_hes_p_test, off_ksi_label_jac_u_test, off_ksi_label_hes_u_test, off_ksi_label_jac_v_test, off_ksi_label_hes_v_test) in enumerate(testloader):  
                    loss_test   += Net2.loss_label(labeled_inputs_test, labeled_outputs_u_test)
            if epoch%options['epoch_print'] == 0 or epoch == options['EPOCH']-1:
                print("|epoch=%5d,nbatch=%5d | loss=(%11.7e,  %11.7e)"%(epoch,nbatch,running_loss,loss_test)) 
        loss_history_train[options['EPOCH']+epoch,0] = loss_train/NBatch


        scheduler2.step()
        if epoch % options['epoch_save'] == 0 or epoch == options['EPOCH']-1:
            torch.save({
                'epoch':epoch+epoch_last2,
                'Net2_state_dict': Net2.state_dict(),
                'optimizer_state_dict': optimizer2.state_dict(),
                'scheduler_state_dict': scheduler2.state_dict(),
            }, netfile2)

        
        if epoch % options['epoch_error_check'] == 0 or epoch == options['EPOCH']-1:
            Net2.eval()  
            
            with torch.no_grad():
            
                udiff_norm = Net2.error_compute(Val_inputs,p_exact_norm, u_exact, v_exact, relative_exact_p, relative_exact_u, relative_exact_v)
            
                Errorpuv = np.mean(udiff_norm/relative_exact_u)

                print("|epoch=%5d | u_error=(%11.7e)"%(epoch,Errorpuv))

        loss_history_test[epoch,0] = loss_test/NBatch

        if epoch>=20000:
            if np.all( loss_history_test[epoch-2000:epoch+1,0]>loss_history_test[epoch-2001:epoch,0]):
                break


    if continue_tr:
        checkpoint3 = torch.load(netfile3_old,map_location=torch.device('cpu'))
        epoch_pre3 = checkpoint3['epoch']
        Net3.load_state_dict(checkpoint3['Net3_state_dict'])
        optimizer3 = torch.optim.Adam(Net3.parameters())
        scheduler3 = torch.optim.lr_scheduler.LambdaLR(optimizer3, lr_lambda=options['lamda'])
        optimizer3.load_state_dict(checkpoint3['optimizer_state_dict'])
        scheduler3.load_state_dict(checkpoint3['scheduler_state_dict'])

    if optimizer_give and scheduler_give:
        optimizer3 = optimizer_given3
        scheduler3 = scheduler_given3

    vdiff_norm = Net3.error_compute(Val_inputs,p_exact_norm, u_exact, v_exact, relative_exact_p, relative_exact_u, relative_exact_v)
    Errorpuv = np.mean(vdiff_norm/relative_exact_v)

    print("|starting | v_error=(%11.7e)"%(Errorpuv))

    loss_history_test = np.zeros((options['EPOCH'], 1))

    for epoch in range(options['EPOCH']):
        Net3.train()
       
        loss_train = 0
        loss_test =0
        for nbatch, (labeled_inputs_batch,labeled_outputs_p_int_batch, labeled_outputs_u_batch, labeled_outputs_v_batch, label_p_x_batch, label_p_y_batch,label_u_x_batch,label_u_y_batch,label_v_x_batch,label_v_y_batch,off_ksi_label_jac_p_batch, off_ksi_label_hes_p_batch, off_ksi_label_jac_u_batch, off_ksi_label_hes_u_batch, off_ksi_label_jac_v_batch, off_ksi_label_hes_v_batch) in enumerate(trainloader):      
            running_loss=optimizer3.step(closure3)
            loss_train += running_loss 
            with torch.no_grad():
                for nbatch2, (labeled_inputs_test,labeled_outputs_p_int_test, labeled_outputs_u_test, labeled_outputs_v_test, label_p_x_test, label_p_y_test,label_u_x_test,label_u_y_test,label_v_x_test,label_v_y_test,off_ksi_label_jac_p_test, off_ksi_label_hes_p_test, off_ksi_label_jac_u_test, off_ksi_label_hes_u_test, off_ksi_label_jac_v_test, off_ksi_label_hes_v_test) in enumerate(testloader):  
                    loss_test   += Net3.loss_label(labeled_inputs_test, labeled_outputs_v_test)
            if epoch%options['epoch_print'] == 0 or epoch == options['EPOCH']-1:
                print("|epoch=%5d,nbatch=%5d | loss=(%11.7e,  %11.7e)"%(epoch,nbatch,running_loss,loss_test)) 
        loss_history_train[2*options['EPOCH']+epoch,0] = loss_train/NBatch

        scheduler3.step()
        if epoch % options['epoch_save'] == 0 or epoch == options['EPOCH']-1:
            torch.save({
                'epoch':epoch+epoch_last3,
                'Net3_state_dict': Net3.state_dict(),
                'optimizer_state_dict': optimizer3.state_dict(),
                'scheduler_state_dict': scheduler3.state_dict(),
            }, netfile3)

        
        if epoch % options['epoch_error_check'] == 0 or epoch == options['EPOCH']-1:
            Net3.eval()  
            
            with torch.no_grad():
                vdiff_norm = Net3.error_compute(Val_inputs,p_exact_norm, u_exact, v_exact, relative_exact_p, relative_exact_u, relative_exact_v)


                Errorpuv = np.mean(vdiff_norm/relative_exact_v)

                print("|epoch=%5d | v_error=(%11.7e)"%(epoch,Errorpuv))

        loss_history_test[epoch,0] = loss_test/NBatch

        if epoch>=20000:
            if np.all( loss_history_test[epoch-2000:epoch+1,0]>loss_history_test[epoch-2001:epoch,0]):
                break



    return loss_history_train

