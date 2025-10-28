import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.autograd as ag
import torch.utils.data as Data
from collections import OrderedDict
from Activations_plus import Swish

ACTIVATE     = Swish
torch.manual_seed(12)  # reproducible
np.random.seed(1234)
DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_options_default ={'EPOCH':50000,\
                        'LR':0.001, \
                        'lamda': lambda epoch: 0.95**(epoch//2000),\
                        'epoch_print': 200,\
                        'epoch_save':200,
                        'epoch_error_check':200
                        }

def gaussian_kernel(xgrid, sigma=1.0):
    # Gaussian kernel formula
    r = xgrid[:, np.newaxis] - xgrid[np.newaxis, :]
    gaussian = np.exp(-r**2 / (2 * sigma**2))
    return gaussian
    
class para_Net(nn.Module):
    def __init__(self, layers1=None,OldNetfile1=None):
        super(para_Net, self).__init__()
        self.lossfun = nn.MSELoss(reduction='mean')
        NetDict1 = OrderedDict()

        #NetDict['start'] = nn.Linear(1,1)
        if not(layers1 or OldNetfile1):
            raise Exception('At least one of the parameters "Layers1" and "OldNerfile1" should be given ')
        if OldNetfile1:
            oldnet = torch.load(OldNetfile1, map_location=lambda storage, loc: storage)
            layers1 = oldnet['layers']
        self.layers1 = layers1
        for i in range(len(layers1)-1):
            key    = "Layer1%d_Linear"%i
            Value  = nn.Linear(layers1[i],layers1[i+1])
            init.xavier_normal_(Value.weight)
            init.zeros_(Value.bias)
            NetDict1[key] = Value

            if i != len(layers1)-2:
                key    = "Layer1%d_avtivate"%i
                Value  = ACTIVATE()
                NetDict1[key] = Value
        self.unet1 = nn.Sequential(NetDict1)



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
        #return self.u_net(x).detach().cpu().numpy()
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

    def loadnet(self, OldNetfile):
        #self.load_state_dict(torch.load(OldNetfile)['state_dict'],  map_location=lambda storage, loc: storage)
        state_dict = torch.load(OldNetfile, map_location=lambda storage, loc: storage)['state_dict']
        self.load_state_dict(state_dict)
    def savenet(self, netfile):
        torch.save({'layers': self.layers1, 'state_dict': self.state_dict()}, netfile )



def train(Net,data, netfile1, Val_inputs, Val_source, Val_exact, optimizer_give = False, optimizer_given = None,
          scheduler_give = False, scheduler_given = None, epoch_last =0,lamda=1., options=train_options_default, netfilesave_provide=None, netfilesave=None, resi_lamda = 0., penalty_grad = False, label_grad = False, kernel_grad = False):

    lamba4 = lamda
    if len(data) == 9:
        labeled_inputs, labeled_outputs, labeled_normalize, labeled_outputs_min, labeled_outputs_max, inputs, source,datatype,trainratio= data
    elif len(data) == 6:
        labeled_inputs, labeled_outputs, inputs, source,datatype,trainratio= data
    else:
        raise Exception('Expect inout <data> a tuple with 8 elements, but got %d'%len(data))

    weight = np.ones((inputs.shape[0], 1))
    lb = Net.roeqs.design_space[0:1,:]
    ub = Net.roeqs.design_space[1:2,:]
    dis = np.zeros((inputs.shape[0], 1))
    for i in range(inputs.shape[0]):
        diff = ( inputs[i:i+1,:]-labeled_inputs )/(ub-lb)
        dis[i,0] = np.linalg.norm(diff, axis=1).min()
    weight = dis/dis.max()
    dataset   = Data.TensorDataset(torch.tensor(inputs).float().to(DEVICE),
                                   torch.tensor(source).float().to(DEVICE),
                                   torch.tensor(weight).float().to(DEVICE),
                                   )
    trainsize =  int(inputs.shape[0] * trainratio)
    testsize  =  inputs.shape[0] - trainsize
    trainset, testset = Data.random_split(dataset, [trainsize, testsize])
    trainloader = Data.DataLoader(trainset, batch_size= trainsize//options['NBATCH'], shuffle = True)
    testloader  = Data.DataLoader(testset , batch_size= testsize                    , shuffle = True)


    dataset_label   = Data.TensorDataset(torch.tensor(labeled_inputs).float().to(DEVICE),
                                   torch.tensor(labeled_outputs).float().to(DEVICE),
                                   )
    trainsize_label =  int(labeled_inputs.shape[0] * trainratio)
    testsize_label  =  labeled_inputs.shape[0] - trainsize_label
    trainset_label, testset_label = Data.random_split(dataset_label, [trainsize_label, testsize_label])
    trainloader_label = Data.DataLoader(trainset_label, batch_size= trainsize_label//options['NBATCH'], shuffle = True)
    testloader_label  = Data.DataLoader(testset_label , batch_size= testsize_label                    , shuffle = True)

    def lossHybrid_nonli(lamba4=1.):

        loss = 0.

        if lamba4 != 0:        
            loss2 = Net.loss_label(labeled_inputs, labeled_outputs)
            loss  += lamba4*loss2

        if label_grad:

            loss4 = Net.loss_label_grad(labeled_inputs)
            if epoch%options['epoch_print'] == 0:
                print("|epoch=%5d,nbatch=%5d | label loss= %11.7e,  grad loss= %11.7e"%(epoch,nbatch,loss,loss4))
            loss = loss + 0.001*loss4

        return loss


    def closure():
        optimizer.zero_grad()
        loss = lossHybrid_nonli(lamba4=lamba4)
        loss.backward()
        return loss

    loss_history_train = np.zeros((options['EPOCH'], 1))
    NBatch = len(trainloader)
    loss_history_test  = np.zeros((options['EPOCH'], 1))
    loss_history_resi  = np.zeros((options['EPOCH'], 1))

    if optimizer_give:
        optimizer = optimizer_given
    else:
        optimizer = torch.optim.Adam(Net.parameters(), lr=options['LR'], weight_decay=options['weight_decay'])
    if scheduler_give:
        scheduler = scheduler_given
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=options['lamda'])
    for epoch in range(options['EPOCH']):
        loss_train = 0
        loss_test = 0
        loss_resi = 0
        for nbatch, (labeled_inputs, labeled_outputs) in enumerate(trainloader_label):
            running_loss=optimizer.step(closure)
            loss_train += running_loss
            for nbatch, (inputs,source,weight) in enumerate(trainloader):
                loss_resi += Net.loss_Strong_Residue(inputs,source,weight,lamba1=1.,lamba2=1.,lamba3=1.)
            with torch.no_grad():
                for nbatch2, (labeled_inputs, labeled_outputs) in enumerate(testloader_label):
                    loss_test   += lossHybrid_nonli(lamba4=lamba4)
            if epoch%options['epoch_print'] == 0:
                print("|epoch=%5d,nbatch=%5d | loss=(%11.7e,  %11.7e,  %11.7e)"%(epoch,nbatch,running_loss,loss_test,loss_resi))
        loss_history_train[epoch,0] = loss_train/NBatch
        loss_history_test[ epoch,0] = loss_test/NBatch
        loss_history_resi[ epoch,0] = loss_resi/NBatch


        scheduler.step()
        if epoch % options['epoch_save'] == 0 or epoch == options['EPOCH']-1:

            if netfilesave_provide:

                torch.save({
                'epoch':epoch+epoch_last,
                'model_state_dict': Net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, netfilesave)

            else:

                torch.save({
                    'epoch':epoch+epoch_last,
                    'model_state_dict': Net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, netfile1)

        if epoch % options['epoch_error_check'] == 0:
            
                
            error = np.mean(Net.error_val(Val_inputs, Val_exact))

            print("|epoch=%5d | error=(%11.7e)"%(epoch, error))


    return loss_history_train, loss_history_test, loss_history_resi



def evaluate(Net,netfile,Val_inputs, Val_source, Val_exact):

    Net.loadnet(netfile)


    loss_resi = Net.loss_Strong_Residue(Val_inputs,Val_source,weight=1,lamba1=1.,lamba2=1.,lamba3=1.)

    error = Net.error_val(Val_inputs, Val_exact)

    return loss_resi, error