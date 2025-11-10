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

ACTIVATE     = Swish()
torch.manual_seed(12)  # reproducible
np.random.seed(1234)
DEVICE     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_options_default ={'EPOCH':50000,\
                        'LR':0.001, \
                        'lamda': lambda epoch: 0.95**(epoch//5000),\
                        'epoch_print': 100,\
                        'epoch_save':100,
                        'epoch_error_check':100
                        }


class para_Net(nn.Module):
    def __init__(self, layers1=None,OldNetfile1=None, init_type='zero', init_weights=None):
        super(para_Net, self).__init__()
        self.lossfun = nn.MSELoss(reduction='mean')
        NetDict1 = OrderedDict()

        if not(layers1 or OldNetfile1):
            raise Exception('At least one of the parameters "Layers1" and "OldNerfile1" should be given ')
        if OldNetfile1:
            oldnet = torch.load(OldNetfile1, map_location=lambda storage, loc: storage)
            layers1 = oldnet['layers']
        self.layers1 = layers1
        self.hidden = nn.Linear(self.layers1[0], self.layers1[1])
        self.output = nn.Linear(self.layers1[1], self.layers1[2])

        if init_weights is None:
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(self.hidden.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(self.hidden.bias)
                nn.init.kaiming_normal_(self.output.weight)
                nn.init.zeros_(self.output.bias)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(self.hidden.weight)
                nn.init.zeros_(self.hidden.bias)
                nn.init.xavier_normal_(self.output.weight)
                nn.init.zeros_(self.output.bias)
            elif init_type == 'uniform':
                nn.init.uniform_(self.hidden.weight, -0.1, 0.1)  
                nn.init.zeros_(self.hidden.bias)
                nn.init.uniform_(self.output.weight, -0.1, 0.1)
                nn.init.zeros_(self.output.bias)
            elif init_type == 'normal':
                nn.init.normal_(self.hidden.weight, 0, 0.02)  
                nn.init.zeros_(self.hidden.bias)
                nn.init.normal_(self.output.weight, 0, 0.02)
                nn.init.zeros_(self.output.bias)

            elif init_type == 'zero':
                nn.init.zeros_(self.hidden.weight) 
                nn.init.zeros_(self.hidden.bias)
                nn.init.zeros_(self.output.weight)
                nn.init.zeros_(self.output.bias)
            else:
                raise ValueError(f"Unsupported initialization type: {init_type}")
        else:

            with torch.no_grad():
                self.hidden.weight.copy_(init_weights['hidden_weight'])
                self.hidden.bias.copy_(init_weights['hidden_bias'])
                self.output.weight.copy_(init_weights['output_weight'])
                self.output.bias.copy_(init_weights['output_bias'])



    def unet1(self,x):
        x = torch.tanh(self.hidden(x))
        x = self.output(x)
        return x



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
        state_dict = torch.load(OldNetfile, map_location=lambda storage, loc: storage)['state_dict']
        self.load_state_dict(state_dict)
    def savenet(self, netfile):
        torch.save({'layers': self.layers1, 'state_dict': self.state_dict()}, netfile )

