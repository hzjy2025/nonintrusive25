import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
from Chebyshev import Chebyshev1D
from Simple_On_NNburger import DEVICE, para_Net
from Normalization import Normalization
from Clenshaw_Curtis import clenshaw_curtis_weights, clenshaw_curtis_quadrature_np, clenshaw_curtis_quadrature
from scipy.io import loadmat
import numpy as np
from numpy import linalg as LA
import torch
import torch.autograd as ag
import torch.nn.functional as F
from torch.func import hessian
from torch.func import jacrev
from torch import nn
import random
from torch.func import vmap, vjp
from functorch import jacrev, jacfwd
from random import shuffle
import matplotlib.pyplot as plt
import scipy.io
import time
from scipy import linalg


# reproducible
torch.manual_seed(1234)
np.random.seed(1234)


class off_fun_eval_deri():
    def __init__(self, xL=-1, xR=1, N_1x=129):
        self.xL, self.xR= xL,xR
        self.xlen = xR-xL
        self.N_1x  = N_1x
        self.xChby = Chebyshev1D(xL=self.xL, xR=self.xR, M=self.N_1x-1)

        self.D1_x = torch.tensor(self.xChby.DxCoeff(1)).float().to(DEVICE)
        self.D2_x = torch.tensor(self.xChby.DxCoeff(2)).float().to(DEVICE)



    def svd_hign_sol(self,U,L):

        Usvd, s, Vh = linalg.svd(U)

        return Usvd[:,:L]

    def Compute_d_dxc(self, phi):
        
        return torch.matmul(self.D1_x,phi)

    def Compute_d_dxc2(self, phi):
        
        return self.Compute_d_dxc( self.Compute_d_dxc(phi) )

    def compute_off_no_theta(self,phi,kind):

        if kind == 'u_xc':

            return self.Compute_d_dxc(phi)

        elif kind == 'u_xc2':

            return self.Compute_d_dxc2(phi)

        else:

            print('Please give a valid type')


class off_all():

    def __init__(self, N_1x):

        self.N_1x = N_1x

    def RB_fun(self, sol_snap, snap_num, L,xL=-1, xR=1, N_1x=129):

        off_ob = off_fun_eval_deri(xL=-1, xR=1, N_1x=self.N_1x)

        Psi_RB = off_ob.svd_hign_sol(sol_snap[:,:snap_num],L)

        onlinein = Psi_RB

        return onlinein


    def compute_jaches_ksi(self,onlinein, L, xL=-1, xR=1, N_1x=129, Xdim=1):

        off_ob = off_fun_eval_deri(xL=-1, xR=1, N_1x=N_1x)

        off_jac = np.zeros((N_1x,L))

        off_hes = np.zeros((N_1x,L))

        phi = onlinein

        off_jac[:,:] = off_ob.compute_off_no_theta(phi,'u_xc')

        off_hes[:,:] = off_ob.compute_off_no_theta(phi,'u_xc2')

        return off_jac, off_hes



    def eigen_snap(self,sol_snap, snap_num, L, theta):

        accu = 0.9999999

        U_snap = sol_snap[:,:snap_num]

        Usvd_u, s_u, Vh_u = linalg.svd(U_snap)

        sq_sum_u = np.sum(s_u**2)

        rate_u = 0

        L_i = 0

        Llist = []

        ratelist_u = []

        for i in range(16):

            L_i = L_i + 1

            Llist.append(L_i)

            rate_u = 1 - np.sum(s_u[:L_i]**2)/sq_sum_u

            ratelist_u.append(rate_u)

        Larr = np.array(Llist)

        ratearr_u = np.array(ratelist_u)

        fig, ax = plt.subplots()

        ax.plot(Larr,ratearr_u,label = 'u')

        fig.suptitle('Projection error with respect to L, high fidelity solution')
        plt.xlabel('L')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

        return Usvd_u

class On_CustomedNet(para_Net):
    def __init__(self, rvalue, Phi, Phir_x, Phir_xx, u_off, u_off_x, u_off_x_x, roeqs=None, omega=1e5, scale_give=False,
                 scale=None,use_POD=False, use_POD_more=False,podout=None,podout_jac=None,podout_hessian=None,
                 podout_more=None,podout_jac_more=None,podout_hessian_more=None,layers1=[5,1,1],OldNetfile1=None, init_type='zero', init_weights=None, test_fun =5):
        super(On_CustomedNet, self).__init__(layers1=layers1,OldNetfile1=OldNetfile1, init_type=init_type, init_weights=init_weights)
        self.M = roeqs.M
        self.lb = torch.tensor(roeqs.design_space[0:1,:]).float().to(DEVICE)
        self.ub = torch.tensor(roeqs.design_space[1:2,:]).float().to(DEVICE)
        self.roeqs = roeqs
        self.rvalue = rvalue

        npsource = roeqs.getsource(roeqs.parameters[:,0:1], roeqs.parameters[:,1:2])
        self.source = torch.tensor( npsource ).float().to(DEVICE)
        ###self.labeledLoss = self.loss_Eqs(self.labeled_inputs,self.labeled_outputs, self.source)
        self.omega = omega

        self.use_POD=use_POD
        self.use_POD_more=use_POD_more
         #[M,Nh_in]
        Modes_all, sigma, _ = np.linalg.svd(self.roeqs.Samples);
        
        self.POD_modes = torch.tensor(Modes_all[1:-1,:test_fun].T).float().to(DEVICE)
        self.POD_modes_end = torch.tensor(Modes_all[:,:test_fun].T).float().to(DEVICE)
         #[Nh,M]
        self.POD_modes_dx_int = self.roeqs.dx@Modes_all[:,:test_fun]
        #[M,Nh_in]
        self.POD_modes_dx = torch.tensor(self.POD_modes_dx_int[1:-1,:].T).float().to(DEVICE)
        self.POD_modes_dx_end = torch.tensor(self.POD_modes_dx_int.T).float().to(DEVICE)

        if use_POD:

            self.podout=torch.tensor(podout).float().to(DEVICE)
            self.podout_jac=podout_jac
            self.podout_hessian=podout_hessian
            
        elif use_POD_more:

            self.podout_more=torch.tensor(podout_more).float().to(DEVICE)
            self.podout_jac_more=podout_jac_more
            self.podout_hessian_more=podout_hessian_more


        # Phi [Nh,r]
        self.Phi = torch.tensor(Phi, requires_grad=True).float().to(DEVICE)
        # Phi [Nh_in,r]
        self.Phir_x = Phir_x.to(DEVICE)
        # Phi [Nh_in,r]
        self.Phir_xx = Phir_xx.to(DEVICE)
        # [Nh]
        self.u_off = u_off.to(DEVICE)
        # [Nh_in]
        self.u_off_x = u_off_x.to(DEVICE)
        # [Nh_in]
        self.u_off_x_x = u_off_x_x.to(DEVICE)
         # [r,1]

        scaling_factor_x = 1.0
        self.scaling_factor = scaling_factor_x ** (1/2)

        if scale_give:

            self.scaling_factor = scale

      # Rescale the offline quantities
        self.u_off_rescaled = self.u_off / self.scaling_factor
        self.u_off_x_rescaled = self.u_off_x / self.scaling_factor
        self.u_off_x_x_rescaled = self.u_off_x_x / self.scaling_factor
# This is a one hidden layer, with Phi as inputs and u as outputs, optimize w1,b1,w2,b2
    def u_net(self,x):
       
        out = self.unet1(x)
        return out

    def forward(self,x):
        return self.u_net(x).detach().cpu().numpy()

    def compute_deri_2_jaches2(self, online_in):

        online_in = torch.tensor(online_in, requires_grad=True).float()

        compute_batch_output = vmap(self.u_net, in_dims=(0), randomness='same')

        batch_out = compute_batch_output(online_in)

        compute_batch_jacobian = vmap(jacrev(self.u_net, argnums=0), in_dims=(0), randomness='same')

        batch_jacobian0 = compute_batch_jacobian(online_in)

        compute_batch_hessian = vmap(torch.func.jacfwd(jacrev(self.u_net, argnums=0),randomness='same'), in_dims=(0), randomness='same')

        batch_hessian0 = compute_batch_hessian(online_in)

        # self.on_fun R^{r} >>> R^{1}
        # batch_out  [x_num,1]
        # batch_jacobian0.shape [x_num,1,r]
        # return [x_num,1], [x_num,r], [r,x_num,r]
        return batch_out,batch_jacobian0[:,0,:],torch.transpose(batch_hessian0[:,0,:,:], 0, 1)

    def on_fun(self, off_out_mesh, W1, b1, W2, b2):

        return (torch.tanh(off_out_mesh@W1+b1))@W2+b2

    def online_deri(self, online_in, W1, b1, W2, b2):
        # Given the weights and bias of the U, give derivative of u w.r.t. online_in

        l1 = torch.matmul(online_in,W1) + b1

        uhat_jac = torch.matmul((torch.transpose(W2, 1, 2)*(1-torch.tanh(l1)**2)),torch.transpose(W1, 1, 2))

        uhat_hes = torch.matmul(torch.matmul(torch.matmul(torch.matmul(W1,torch.diag_embed(W2[:,:,0])), torch.diag_embed(-2*torch.tanh(torch.transpose(l1, 0, 1)))) , torch.diag_embed(1-torch.tanh(torch.transpose(l1, 0, 1))**2)),torch.transpose(W1, 1, 2))

        return uhat_jac, torch.transpose(torch.transpose(uhat_hes, 0, 2), 0, 1)


    def loss_for_thisNN_Strong_no_on_plot(self, uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, source):

        u_on = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u')
        u_on_x = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u_x')
        u_on_xx = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u_x_x')
        u = u_on[0,:]
        u_x = u_on_x[0,:]
        u_x_x = u_on_xx[0,:] 
        
        return u*u_x-u_x_x-source


    def loss_for_thisNN_Strong_no_on_int(self, uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, source):

        u_on = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u')
        u_on_x = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u_x')
        u_on_xx = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u_x_x')
        u = u_on[0,:]
        u_x = u_on_x[0,:]
        u_x_x = u_on_xx[0,:] 
        f1 = u*u_x-u_x_x-source
        f1=f1.reshape(1,-1)
        loss = clenshaw_curtis_quadrature(f1[:,1:-1]**2, f1.shape[1], discard_endpoints=True) 
        
        return loss
        

    def loss_for_thisNN_Strong_no_on(self, uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, source):

        u_on = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u')
        u_on_x = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u_x')
        u_on_xx = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u_x_x')
        u = u_on[0,:]
        u_x = u_on_x[0,:]
        u_x_x = u_on_xx[0,:] 
        f1 = torch.mean((u*u_x-u_x_x-source)**2)

        bc_value = u_on[0,np.array([0, -1])] 
        fbc = torch.mean((bc_value)**2)
        loss = f1+fbc*self.omega
        
        return loss


    def loss_for_thisNN_Strong_no_on_regu(self, uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, source, u_off, regu_lambda=10):

        u_on = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u')
        u_on_x = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u_x')
        u_on_xx = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, 'u_x_x')
        u = u_on[0,:]
        u_x = u_on_x[0,:]
        u_x_x = u_on_xx[0,:] 
        residual = u*u_x-u_x_x-source
        f1 = torch.mean((residual)**2)
        bc_value = u_on[0,np.array([0, -1])] 
        fbc = torch.mean((bc_value)**2)
        
        return f1, fbc*self.omega, torch.mean((u_on[0,:]-u_off)**2)
        

    def error_for_thisNN_no_on(self, Phi_on, u_star):

        u_on = self.u_net(Phi_on)
        
        u = u_on[1:-1,0]

        err = np.sqrt(np.sum((u.detach().cpu().numpy() - u_star)**2)/np.sum(u_star**2))

        return err


    def vmap_2nd_deri(self,offd1,offd2,ond1,ond2):

        B = torch.sum(ond2*offd1,2)

        hat_d_d = torch.sum(torch.transpose(B, 0, 1) * offd1,1) + torch.sum(ond1*offd2,1)

        return hat_d_d

    def compute_deri_2(self, uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, kind):

        '''
         batch_jacobian shape [Nh,L], the partial derivative of reduced basis function w.r.t. input x
         First dim is number of different collocation points, second dim is each RB functions


        '''

        prime_x = batch_jacobian

        prime_xx = batch_hess

        if kind == 'u_x':

            u_x = torch.sum(uhat_jac_mat*prime_x,2)

            return u_x

        elif kind == 'u_x_x':

            uhat_x_x_vmap = vmap(self.vmap_2nd_deri, in_dims=(None, None, 0, 0), randomness='same')
            uhat_x_x = uhat_x_x_vmap(prime_x, prime_xx, uhat_jac_mat, uhat_hes_mat)
            return uhat_x_x

        elif kind == 'u':

            return allout[:,:,0]


        else:
            print('Please type a valid kind of derivative!')

  

