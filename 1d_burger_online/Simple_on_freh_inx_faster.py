import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../1dburger_pyfile")))
import scipy.io
from Burger_nonlinear_output_no_norm import CustomedEqs, CustomedNet
from Simple_On_Deltau_Burger_nonlinear import On_CustomedNet
from Normalization import Normalization
from Clenshaw_Curtis import clenshaw_curtis_weights, clenshaw_curtis_quadrature_np
from Simple_On_NNburger import DEVICE, train_options_default
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.init as init
from Activations_plus import Swish
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval, chebder
import copy
from numpy.polynomial.legendre import leggauss
torch.manual_seed(12)  
np.random.seed(1234)

ACTIVATE     = Swish

train_options_default['lamda'] = lambda epoch: 0.95**(epoch//5000)

regu_lambda=1e1

# change the following setting for test offline results in different cases
Fre_h = 9
SampleNum_Vec_ontrain     = [40]
rlist_ontrain             = [12]
inx_vec = list(range(50,100))
init_random = False
u_off_regu = True
plot_online = False
Nval = 100
adam_epoch = 400000   
save_loss_history = False
# set one of them to be true and the other false
continue_off_strong = True
continue_off_weak = False
# when continue_off_weak is True, the number of test functions
test_fun = 20


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

def GetValData(roeqs,Nval):
    Nin = roeqs.design_space.shape[1]
    Val_inputs  = Normalization.Anti_Mapminmax(np.random.rand(Nval,Nin)*2-1,  roeqs.design_space)
    alpha1 = Val_inputs[:,0:1]
    alpha2 = Val_inputs[:,1:2]
    Val_source = roeqs.getsource(alpha1, alpha2)
    Val_exact = roeqs.phix(roeqs.xgrid.T,alpha1,alpha2)
    return Val_inputs, Val_source, Val_exact

M_Vec             = [20]
onNN_H1_fix           = [5]
NetSize_Vec       = [100]
NResi_Vec         = [1000]

VarsRange_dict    = {'SampleNum': (SampleNum_Vec_ontrain, 40, ), \
                     'NetSize'  : (NetSize_Vec,   100, ), \
                     'rvalue'  : (rlist_ontrain ,   10, ), \
                     'NResi': (NResi_Vec,  1000, )
                     }

Vars_dict         = {key:value[1] for key, value in VarsRange_dict.items()}
Vars_dict['M']=20
Vars_dict['H1']=5
Vars_dict['Fre_h']=Fre_h
keys = list( Vars_dict.keys() )
roeqs = CustomedEqs(M_Vec[-1], h=Fre_h, Num_par1 = SampleNum_Vec_ontrain[0], Num_par2 =SampleNum_Vec_ontrain[0])
Val_inputs, Val_source, Val_exact = GetValData(roeqs,Nval)

resultsdir = '../1dburger_pyfile/resultsburger_high_fre_lbl_resi0_ladel_data_and_gradient'
savedir = f'../1d_burger_online/resultsburger_new_resi0_Freh{Fre_h}_label_grad'

def Dict2Str(dicti):
    describes = ''
    for name in keys:
        val = dicti[name]
        describes += '_'+name+''+str(val).replace('.', '_')
    describes = 'Burges1D' + describes
    return describes

def chebyshev_basis(x, degree):
    """
    x: [N_x, 1]
    Returns: [N_x, degree+1]
    """
    x = x.view(-1, 1)
    T0 = torch.ones_like(x)
    T1 = x
    T = [T0, T1]

    for _ in range(2, degree + 1):
        Tn = 2 * x * T[-1] - T[-2]
        T.append(Tn)
    
    return torch.cat(T, dim=1)


def chebyshev_basis_symbolic(x, degree):
    # x: [N, 1]
    theta = torch.arccos(x)  # [N, 1]
    n = torch.arange(0, degree + 1, device=x.device).reshape(1, -1)  # [1, degree+1]
    T = torch.cos(n * theta)  # [N, degree+1]
    return T

def evaluate_all_chebyshev(T, coeffs):
    """
    coeffs: [20, degree+1]
    Returns: [N_x, 20]
    """
  # [N_x, deg+1]
    return T @ coeffs.T  # [N_x, 20]



def chebyshev_project_and_diff2(u, x_cheb):
    """
    Return interpolant, first order derivative, and second order derivative functions.
    

    u      : function values at Chebyshev points (from 1 to -1)
    x_cheb : Chebyshev points (from 1 to -1)
    
    u_interp : interpolant of u
    u_prime  : first order derivative
    u_double_prime : second order derivative
    """
    coeffs = chebfit(x_cheb, u, deg=len(u) - 1)
    coeffs_deriv = chebder(coeffs)
    coeffs_second_deriv = chebder(coeffs_deriv)

    def u_interp(x):
        return chebval(x, coeffs)

    def u_prime(x):
        return chebval(x, coeffs_deriv)

    def u_double_prime(x):
        return chebval(x, coeffs_second_deriv)

    return u_interp, u_prime, u_double_prime

def compute_regu_lambda(epoch):
    if epoch <= 100:
        # From (0, 1e7) to (100, 1e6)
        x1, y1 = 0, 1e7
        x2, y2 = 100, 1e6
    elif epoch <= 10000:

        x1, y1 = 100, 1e6
        x2, y2 = 10000, 1e5
    elif epoch <= 50000:

        x1, y1 = 10000, 1e5
        x2, y2 = 50000, 1e4
    elif epoch <= 100000:

        x1, y1 = 50000, 1e4
        x2, y2 = 100000, 1e3
    elif epoch <= 150000:

        x1, y1 = 100000, 1e3
        x2, y2 = 150000, 0
    else:
        # keep it at 0
        return 0
    # y = kx + b
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    return k * epoch + b




def getsource_verify(x,alpha1, alpha2,h=9):

    f1    = (1+alpha1*x)*(x**2-1)
    f2    = np.sin(-h*alpha2*x/3)
    f1_x  = 3*alpha1*x**2 +2*x -alpha1
    f2_x  = -h*alpha2/3*np.cos(-h*alpha2*x/3)
    f1_xx = 6*alpha1*x + 2
    f2_xx = -h**2*alpha2**2/9*np.sin(-h*alpha2*x/3)
    phi_x  = f1*f2_x + f1_x*f2
    phi_xx = 2*f1_x*f2_x + f1*f2_xx +f1_xx*f2
    source = f1*f2*phi_x - phi_xx
    return source[0,:]
    
class gen_testcases_fixH1(object):
    def __init__(self,Val_inputs, Val_source, Val_exact, rvalue, H1, freh, ControlVarName = 'SampleNum',ControlVar=80):
        if  ControlVarName not in VarsRange_dict.keys():
            raise Exception('invalid var name')
        self.ControlVarName = ControlVarName
        self.ControlVar = ControlVar
        self.rvalue = rvalue
        self.H1num = H1
        self.freh = freh
        self.Val_inputs = Val_inputs
        self.Val_source = Val_source
        self.Val_exact = Val_exact
        
        self.lossstop = 0
        self.stoppatience = 5000
        self.min_loss = 10**19


    def gen_fun(self):
        localdict = Vars_dict.copy()
        localdict[self.ControlVarName] = self.ControlVar
        localdict['rvalue'] = self.rvalue
        localdict['H1']= self.H1num
        localdict['Fre_h']=self.freh
        return localdict, Dict2Str(localdict)


    def Calculate(self,inx_check, adam_epoch = 500000, continue_off_weak = False, continue_off_strong = True, init_random = False, u_off_regu = True, regu_lambda=10,plot_online=False):
        init_type='kaiming'
        losshistory = {}
        errorhistory = {}
        epochhistory = {}    

        case = self.gen_fun()
        print(case[1])
        losshistory[case[1]+'train'], errorhistory[case[1]+'train'], epochhistory[case[1]+'train']=self.CaseSim_nonlinear_train_nn(case,self.Val_inputs, self.Val_source, self.Val_exact,inx=inx_check, adam_epoch = adam_epoch, continue_off_weak = continue_off_weak, continue_off_strong = continue_off_strong, init_random = init_random, u_off_regu = u_off_regu, regu_lambda=regu_lambda,plot_online=plot_online)

            
        return losshistory[case[1]+'train'], errorhistory[case[1]+'train'], epochhistory[case[1]+'train']

    def CaseSim_nonlinear_train_nn(self, case,Val_inputs, Val_source, Val_exact, inx=55, continue_off_weak = False, continue_off_strong = True, adam_epoch = 500000, tol_weak = 1e-20, init_random = False, u_off_regu = True, regu_lambda=10,plot_online=False,train_whole=False,train_point=5, omega=1e3):

        netfile1 = resultsdir  + '/' +          case[1]        +'labeldata_50'    +'.tar'    

        if os.path.isfile(netfile1):
            pass

        roeqs = CustomedEqs(case[0]['M'], h=Fre_h, Num_par1 = case[0]['SampleNum'], Num_par2 = case[0]['SampleNum'])
        # from 2 mu to r*H1+2H1+1 outputs
        layers1 = [2, *[ case[0]['NetSize'] ]*4, case[0]['rvalue']*case[0]['H1']+2*case[0]['H1']+1]
        
        layers2 = [case[0]['M'],case[0]['rvalue'],case[0]['rvalue']]
        
        reduce_net = Reduce_Net(layers2 = layers2)

        Net =CustomedNet(case[0]['rvalue'],case[0]['H1']     
                         ,layers1=layers1,reduce_net=reduce_net, roeqs=roeqs).to(DEVICE)

        checkpoint = torch.load(netfile1,map_location=torch.device('cpu'))

        Net.load_state_dict(checkpoint['model_state_dict'])

        podout = Net.offout

        podout_jac = Net.off_jac_ksi

        podout_hessian = Net.off_hes_ksi

        Phi = Net.To_phir(podout)

        phi_jac_mat, phi_hes_mat = Net.Phi_deri(Net.offout)

        Phir_x, Phir_xx = Net.Phi_x(phi_jac_mat, phi_hes_mat,Net.off_jac_ksi, Net.off_hes_ksi)

        Val_inputs = Val_inputs[inx:inx+1,:]
        alpha1 = Val_inputs[:,0:1]
        alpha2 = Val_inputs[:,1:2]

        Val_source = roeqs.getsource_end(alpha1, alpha2)
        Val_exact = Val_exact[inx:inx+1,:]

        Val_inputs = torch.tensor(Val_inputs).float().to(DEVICE)
        Val_source = torch.tensor(Val_source).float().to(DEVICE)
        Val_source = Val_source.reshape(-1)
        Val_exact = Val_exact.reshape(-1)

        W1_mu,b1_mu,W2_mu,b2_mu = Net.muNN_out_to_weight(Net.u_net(Val_inputs))
        
        print(f'the test example parameter is {Val_inputs}')

        allout = Net.on_fun(Phi,W1_mu, b1_mu,W2_mu, b2_mu)

        uhat_jac_mat,uhat_hes_mat = Net.online_deri(Phi,W1_mu, b1_mu,W2_mu, b2_mu)

        u_off = Net.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout,Phir_x, Phir_xx, 'u')

        u_off_x = Net.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout,Phir_x, Phir_xx, 'u_x')

        u_off_x_x = Net.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout,Phir_x, Phir_xx, 'u_x_x')
        
        options = train_options_default.copy()
        
        trainhistory = []
        
        errors = []

        if init_random:

            def xavier_init(fan_in, fan_out, dtype=torch.float32):
                limit = torch.sqrt(torch.tensor(6.0) / (fan_in + fan_out))
                return torch.empty(fan_out, fan_in, device=DEVICE, dtype=dtype).uniform_(-limit, limit)

        
        if continue_off_strong:

            Phi_on = Phi

            Phi_on = torch.tensor(Phi_on).float().to(DEVICE)

            layers3 = [case[0]['rvalue'],case[0]['H1'],1]
            init_weights = {
            'hidden_weight' : W1_mu[0,:,:].T,
            'hidden_bias': b1_mu[0,0,:],
             'output_weight': W2_mu[0,:,:].T,
               'output_bias' :  b2_mu[0,0,:]
            }

            if init_random:

                input_dim, hidden_dim = W1_mu.shape[1], W1_mu.shape[2]
                output_dim = W2_mu.shape[2]
                
                init_weights = {
                    'hidden_weight': xavier_init(input_dim, hidden_dim),             
                    'hidden_bias': torch.zeros(hidden_dim).to(DEVICE),                            
                    'output_weight': xavier_init(hidden_dim, output_dim),            
                    'output_bias': torch.zeros(output_dim).to(DEVICE)                             
                }


            
            On_Net = On_CustomedNet(case[0]['rvalue'], Phi, Phir_x, Phir_xx, u_off.T, u_off_x.T, u_off_x_x.T,
                                 roeqs=roeqs,layers1=layers3, init_weights = init_weights).to(DEVICE)

           

            optimizer_Adam = torch.optim.Adam(On_Net.parameters(),lr=0.01)
            scheduleradam = torch.optim.lr_scheduler.StepLR(optimizer_Adam, step_size=3000, gamma=0.9)
            

            error_his = np.sqrt(clenshaw_curtis_quadrature_np((allout[0,:,0].detach().cpu().numpy() - Val_exact)**2, Val_exact.shape[0], discard_endpoints=False)/clenshaw_curtis_quadrature_np((Val_exact)**2, Val_exact.shape[0], discard_endpoints=False))
            print(f'starting error : {error_his}')
            errors.append(error_his)
            uhat, uhat_jac_mat, uhat_hes_mat = On_Net.compute_deri_2_jaches2(Phi_on)

            uhat = torch.unsqueeze(uhat, 0)
            uhat_jac_mat = torch.unsqueeze(uhat_jac_mat, 0)
            uhat_hes_mat = torch.unsqueeze(uhat_hes_mat, 0)

            loss = On_Net.loss_for_thisNN_Strong_no_on(uhat_jac_mat, uhat_hes_mat, uhat, Phir_x, Phir_xx, Val_source)
            print(f'starting loss : {loss}')
            residual_fun = On_Net.loss_for_thisNN_Strong_no_on_plot(uhat_jac_mat, uhat_hes_mat, uhat, Phir_x, Phir_xx, Val_source)

            size_tr = 500
            x_tr, w_tr = leggauss(size_tr)

            sorted_indices_tr = np.argsort(x_tr)

            n = roeqs.xgrid.shape[0]
            idx = np.arange(n-1, -1, -1)
            xgrid_sorted = roeqs.xgrid[idx, 0]
            source_tr = torch.tensor(getsource_verify(x_tr,alpha1, alpha2)).float().to(DEVICE)

            podout_tr = np.zeros((size_tr,podout.shape[1]))
            podout_tr_dx = np.zeros((size_tr,podout.shape[1]))
            podout_tr_dxx = np.zeros((size_tr,podout.shape[1]))
            coeffs_list_tr = []

            for i in range(podout.shape[1]):
                uhat_sorted = podout[idx,i].detach().cpu().numpy()
                coeffs_list_tr.append(chebfit(xgrid_sorted, uhat_sorted, deg=n - 1))

            coeffs_np_tr = np.stack(coeffs_list_tr)
            coeffs = torch.tensor(coeffs_np_tr).float().to(DEVICE) # shape [M, Nh]

            
            # Our model from x to solution u
            x_tr = torch.tensor(x_tr).float().view(-1, 1).to(DEVICE)
            x_tr.requires_grad_(True)
            T_cheby_tr = chebyshev_basis_symbolic(x_tr, coeffs.shape[1] - 1)
            podout_tr = evaluate_all_chebyshev(T_cheby_tr, coeffs)

            Phi_tr = Net.To_phir(podout_tr)

            u_tr = On_Net.u_net(Phi_tr) # [N, 1]


            dy_dx_tr = torch.autograd.grad(
                    u_tr,
                    x_tr,
                    grad_outputs=torch.ones_like(u_tr),
                    create_graph=True
                )[0] # [N, 1]

            d2y_dx2_tr = torch.autograd.grad(
                        dy_dx_tr, x_tr,
                        grad_outputs=torch.ones_like(dy_dx_tr),
                        create_graph=True,
                        retain_graph=True
                    )[0]  # [N, 1]

            
            
            # Verify 
            size_verify = 100
            # choose the x points to test the residual loss
            #x_verify = roeqs.xgrid[1:-1,0]
            #x_verify = np.random.uniform(-1, 1, size=size_verify)
            x_verify, w_verify = leggauss(size_verify)

            w_verify = torch.tensor(w_verify).float().to(DEVICE)
            ### Get test functions on new nodes
              
            source_verify = torch.tensor(getsource_verify(x_verify,alpha1, alpha2)).float().to(DEVICE)
         
            ### Get the podout function values on other nodes, then compute the phi and phi_x
            podout_verify = np.zeros((size_verify,podout.shape[1]))
            podout_verify_dx = np.zeros((size_verify,podout.shape[1]))
            podout_verify_dxx = np.zeros((size_verify,podout.shape[1]))
            
            ### Old way to compute derivatives
            for i in range(podout.shape[1]):
                uhat_sorted = podout[idx,i]
                u_interp, u_prime, u_double_prime = chebyshev_project_and_diff2(uhat_sorted.detach().cpu().numpy() , xgrid_sorted)
                u_eval = u_interp(x_verify)
                du_eval = u_prime(x_verify)
                ddu_eval = u_double_prime(x_verify)
                podout_verify[:,i] = u_eval
                podout_verify_dx[:,i] = du_eval
                podout_verify_dxx[:,i] = ddu_eval

            podout_verify_old = torch.tensor(podout_verify).float().to(DEVICE)
            podout_verify_dx_old = torch.tensor(podout_verify_dx).float().to(DEVICE)
            podout_verify_dxx_old = torch.tensor(podout_verify_dxx).float().to(DEVICE)

            Phi_verify_old = Net.To_phir(podout_verify_old)
        # Phi [Nh,r]
            phi_jac_mat_verify_old, phi_hes_mat_verify_old = Net.Phi_deri(podout_verify_old)
        # [Nh,r]
            Phir_x_verify_old, Phir_xx_verify_old = Net.Phi_x(phi_jac_mat_verify_old, phi_hes_mat_verify_old,podout_verify_dx_old, podout_verify_dxx_old)

            uhat_verify_old, uhat_jac_mat_verify_old, uhat_hes_mat_verify_old = On_Net.compute_deri_2_jaches2(Phi_verify_old)

            uhat_verify_old = torch.unsqueeze(uhat_verify_old, 0)
            uhat_jac_mat_verify_old = torch.unsqueeze(uhat_jac_mat_verify_old, 0)
            uhat_hes_mat_verify_old = torch.unsqueeze(uhat_hes_mat_verify_old, 0)

            u_on = On_Net.compute_deri_2(uhat_jac_mat_verify_old, uhat_hes_mat_verify_old, uhat_verify_old, Phir_x_verify_old, Phir_xx_verify_old, 'u')
            u_on_x = On_Net.compute_deri_2(uhat_jac_mat_verify_old, uhat_hes_mat_verify_old, uhat_verify_old, Phir_x_verify_old, Phir_xx_verify_old, 'u_x')
            u_on_xx = On_Net.compute_deri_2(uhat_jac_mat_verify_old, uhat_hes_mat_verify_old, uhat_verify_old, Phir_x_verify_old, Phir_xx_verify_old, 'u_x_x')

                  
            ### New way to compute derivatives
            
            ### Our model! from x to solution u
            x_verify = torch.tensor(x_verify).float().view(-1, 1).to(DEVICE)
            x_verify.requires_grad_(True)

            
            T_cheby_verify_loop = chebyshev_basis(x_verify, coeffs.shape[1] - 1)

            T_cheby_verify_symbolic = chebyshev_basis_symbolic(x_verify, coeffs.shape[1] - 1)

            print('Two different ways to get T')
            #print(torch.max((T_cheby_verify_loop-T_cheby_verify_symbolic)**2)) very small

            T_cheby_verify = chebyshev_basis_symbolic(x_verify, coeffs.shape[1] - 1)
            
            podout_verify = evaluate_all_chebyshev(T_cheby_verify, coeffs)

            #print(torch.max((podout_verify-podout_verify_old)**2))

            Phi_verify = Net.To_phir(podout_verify)

            #print(torch.max((Phi_verify-Phi_verify_old)**2))

            u_verify = On_Net.u_net(Phi_verify) # [N, 1]

            dy_dx_verify = torch.autograd.grad(
                    u_verify,
                    x_verify,
                    grad_outputs=torch.ones_like(u_verify),
                    create_graph=True
                )[0] # [N, 1]

            d2y_dx2_verify = torch.autograd.grad(
                        dy_dx_verify, x_verify,
                        grad_outputs=torch.ones_like(dy_dx_verify),
                        create_graph=True,
                        retain_graph=True
                    )[0]  # [N, 1]    

            print(uhat_verify_old-u_verify)
            print(u_on_x.reshape(-1)-dy_dx_verify.reshape(-1))
            print(u_on_xx.reshape(-1)-d2y_dx2_verify.reshape(-1))

            if plot_online:

                uhat_tr_off, uhat_jac_mat_tr_off, uhat_hes_mat_tr_off = On_Net.compute_deri_2_jaches2(Phi_tr)
                   
                plt.figure(figsize=(10,4))
                plt.subplot(1, 2, 1)
                plt.plot(x_tr[sorted_indices_tr], uhat_tr_off[sorted_indices_tr,0].detach().cpu().numpy(), 'k-', label='Offline prediction function')
                plt.plot(roeqs.xgrid[:,0], Val_exact, 'b-', label='Exact')
                plt.legend()

            regu_lambda = 1.0

            uhat_tr_off = Net.on_fun(Phi_tr,W1_mu, b1_mu,W2_mu, b2_mu)

            u_off_tr = uhat_tr_off[0,:,0]

            best_model = None
            
            for epoch in range(adam_epoch):

                regu_lambda = compute_regu_lambda(epoch)

                T_cheby_tr = chebyshev_basis_symbolic(x_tr, coeffs.shape[1] - 1)
                    
                podout_tr = evaluate_all_chebyshev(T_cheby_tr, coeffs)

                Phi_tr = Net.To_phir(podout_tr)
    
                u_tr = On_Net.u_net(Phi_tr) # [N, 1]
    
                dy_dx_tr = torch.autograd.grad(
                        u_tr,
                        x_tr,
                        grad_outputs=torch.ones_like(u_tr),
                        create_graph=True
                    )[0] # [N, 1]
    
                d2y_dx2_tr = torch.autograd.grad(
                            dy_dx_tr, x_tr,
                            grad_outputs=torch.ones_like(dy_dx_tr),
                            create_graph=True,
                            retain_graph=True
                        )[0]  # [N, 1]

                if u_off_regu:

                    residual = u_tr[:,0]*dy_dx_tr[:,0]-d2y_dx2_tr[:,0]-source_tr

                    bc_value = u_tr[np.array([0, -1]),0] 
                    fbc = torch.mean((bc_value)**2)

                    loss_resi = torch.mean((residual)**2)
                    loss_bc = fbc*omega
                    loss_regu = torch.mean((u_tr[:,0]-u_off_tr)**2)
                    
                  
                    loss = loss_resi + loss_bc + regu_lambda*loss_regu
                else:
                    loss = On_Net.loss_for_thisNN_Strong_no_on(uhat_jac_mat_tr, uhat_hes_mat_tr, uhat_tr, Phir_x_tr, Phir_xx_tr, source_tr)
   

                trainhistory.append(loss.item())
                optimizer_Adam.zero_grad()
              
                loss.backward(retain_graph=True)
                optimizer_Adam.step()
                with torch.no_grad():
                    error_his = On_Net.error_for_thisNN_no_on(Phi_on, Val_exact[1:-1])
                    errors.append(error_his.item())

        
                scheduleradam.step()


                
                T_cheby_verify = chebyshev_basis_symbolic(x_verify, coeffs.shape[1] - 1)

                podout_verify = evaluate_all_chebyshev(T_cheby_verify, coeffs)

                Phi_verify = Net.To_phir(podout_verify)
        
                u_verify = On_Net.u_net(Phi_verify) # [N, 1]
        
                dy_dx_verify = torch.autograd.grad(
                            u_verify,
                            x_verify,
                            grad_outputs=torch.ones_like(u_verify),
                            create_graph=True
                        )[0] # [N, 1]
        
                d2y_dx2_verify = torch.autograd.grad(
                                dy_dx_verify, x_verify,
                                grad_outputs=torch.ones_like(dy_dx_verify),
                                create_graph=True,
                                retain_graph=True
                            )[0]  # [N, 1]    

                residual_verify = u_verify[:,0]*dy_dx_verify[:,0]-d2y_dx2_verify[:,0]-source_verify

                loss_resi_verify = torch.mean((residual_verify)**2)

                self.lossstop += 1
                if  loss_resi_verify.item() < self.min_loss-1e-7:
                    best_model = copy.deepcopy(On_Net.state_dict())
                    self.lossstop = 0
                    self.min_loss = loss_resi_verify.item()
                    best_epoch = epoch
               
                if self.lossstop > self.stoppatience and epoch > 50000 and loss_resi_verify.item() < 1e-1:

                    break
            
                if epoch%options['epoch_print'] == 0:
                    
                    print("|epoch=%5d | Residual loss=%11.7e| BC loss=%11.7e| Regu loss=%11.7e| Validation loss=%11.7e| error=%11.7e"%(epoch,loss_resi.item(),loss_bc.item(),regu_lambda*loss_regu.item(),loss_resi_verify.item(),errors[-1]))


            if plot_online:
                uhat, uhat_jac_mat, uhat_hes_mat = On_Net.compute_deri_2_jaches2(Phi_on)
                plt.subplot(1, 2, 2)
                plt.plot(roeqs.xgrid[:,0], uhat[:,0].detach().cpu().numpy(), 'r-', label='Online')
                plt.plot(roeqs.xgrid[:,0], Val_exact, 'b-', label='Exact')
    
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"chebyshev_derivative_comparison{inx}.png", dpi=300)

            On_Net.load_state_dict(best_model)
        else:

            raise ValueError("Please give a valid training method.")


        return  trainhistory[:best_epoch], np.array(errors[:best_epoch]), np.arange(best_epoch)

if __name__ == '__main__':

    if continue_off_weak:

        matfile = savedir  + '/' + f'Burger_plot_data_adam100k_test_fun{test_fun}_r_{rlist_ontrain[0]}_regu_lambda{regu_lambda}_size3000_lre3.mat'

    else:

        matfile = savedir  + '/' +f'Burger_plot_data_adam100k_onloss_strong_r_{rlist_ontrain[0]}_regu_lambda{regu_lambda}_size3000_lre3.mat'


    if init_random:
        if continue_off_weak:

            matfile = savedir  + '/' + f'Burger_plot_data_adam100k_test_fun{test_fun}_r_{rlist_ontrain[0]}_regu_lambda{regu_lambda}_size3000_lre3_random.mat'
    
        else:
    
            matfile = savedir  + '/' +f'Burger_plot_data_adam100k_onloss_strong_r_{rlist_ontrain[0]}_regu_lambda{regu_lambda}_size3000_lre3_random.mat'

    
    datas = {}
    datas['SampleNum_Vec_ontrain'] = SampleNum_Vec_ontrain
    datas['inx_vec'] = inx_vec

    

    h = onNN_H1_fix[0]
    
    

    if save_loss_history:

        indr_list_error = []
        indr_list_loss = []

        for inx in inx_vec:
        
            for r in rlist_ontrain:
                
                
                samnum_list_error = []
                samnum_list_loss = []
                
                for samnum in SampleNum_Vec_ontrain:
                    
                    indr = rlist_ontrain.index(r)
                    indsamnum = SampleNum_Vec_ontrain.index(samnum)
                    
                    obj = gen_testcases_fixH1(Val_inputs, Val_source, Val_exact, rvalue=r, H1=h, freh=Fre_h, ControlVarName = 'SampleNum',ControlVar=samnum)
                    
                    losshistory, errorhistory, epochhistory = obj.Calculate(inx_check=inx, adam_epoch = adam_epoch, continue_off_strong = continue_off_strong, continue_off_weak = continue_off_weak, init_random = init_random, u_off_regu = u_off_regu,plot_online=plot_online)
        
                    samnum_list_loss.append(losshistory)
                    samnum_list_error.append(errorhistory)
                    
               
            indr_list_loss.append(samnum_list_loss)
            indr_list_error.append(samnum_list_error)
        
    
        indr_array_loss = np.array(indr_list_loss, dtype=object)
        indr_array_error = np.array(indr_list_error, dtype=object)
        datas['indr_array_loss'] = indr_array_loss
        datas['indr_array_error'] = indr_array_error
    else:

        indr_list_error_begin = []
        indr_list_loss_begin = []
        indr_list_error_end  = []
        indr_list_loss_end  = []

        for inx in inx_vec:
        
            for r in rlist_ontrain:
                
                
                samnum_list_error_begin = []
                samnum_list_loss_begin = []
                samnum_list_error_end = []
                samnum_list_loss_end = []
                
                for samnum in SampleNum_Vec_ontrain:
                    
                    indr = rlist_ontrain.index(r)
                    indsamnum = SampleNum_Vec_ontrain.index(samnum)
                    
                    obj = gen_testcases_fixH1(Val_inputs, Val_source, Val_exact, rvalue=r, H1=h, freh=Fre_h, ControlVarName = 'SampleNum',ControlVar=samnum)
                    
                    losshistory, errorhistory, epochhistory = obj.Calculate(inx_check=inx, adam_epoch = adam_epoch, continue_off_strong = continue_off_strong, continue_off_weak = continue_off_weak, init_random = init_random, u_off_regu = u_off_regu,plot_online=plot_online)
        
                    samnum_list_loss_begin.append(losshistory[0])
                    samnum_list_error_begin.append(errorhistory[0])
                    samnum_list_loss_end.append(losshistory[-1])
                    samnum_list_error_end.append(errorhistory[-1])
                    
               
            indr_list_loss_begin.append(samnum_list_loss_begin)
            indr_list_error_begin.append(samnum_list_error_begin)
            indr_list_loss_end.append(samnum_list_loss_end)
            indr_list_error_end.append(samnum_list_error_end)
        
    
        indr_array_loss_end = np.array(indr_list_loss_end, dtype=object)
        indr_array_error_end = np.array(indr_list_error_end, dtype=object)
        indr_array_loss_begin = np.array(indr_list_loss_begin, dtype=object)
        indr_array_error_begin = np.array(indr_list_error_begin, dtype=object)
        datas['indr_array_loss_begin'] = indr_array_loss_begin
        datas['indr_array_error_begin'] = indr_array_error_begin
        datas['indr_array_loss_end'] = indr_array_loss_end
        datas['indr_array_error_end'] = indr_array_error_end
        
    scipy.io.savemat(matfile, datas)
    
