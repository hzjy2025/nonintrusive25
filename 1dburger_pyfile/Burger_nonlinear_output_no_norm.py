import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
from Chebyshev import Chebyshev1D
from NNburger_resi import para_Net, DEVICE
from Normalization import Normalization
from Generate_lid_data import tr_data, choose_para, get_inter_2d_inx, get_inter_2d_value
from Verify_Retrain import plot_cheb_fun

from scipy.io import loadmat
import numpy as np
from numpy import linalg as LA
import torch
import torch.autograd as ag
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
from torch.func import hessian
from torch.func import jacrev
from functools import partial
import random
from torch import nn
from torch.utils.data import DataLoader
from torch.func import vmap, vjp
import torch.nn.functional as F
from functools import partial
from functorch import jacrev, jacfwd
from random import shuffle
from scipy.io import loadmat
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from scipy import linalg
from itertools import chain
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Eqs parameters
a  = 1
Newton = {'iterMax':100, 'eps':1E-10}

# reproducible
torch.manual_seed(1234)
np.random.seed(1234)

def checkpoint(model, filepath):
  torch.save(model.state_dict(),filepath)
def resume(model, filepath):
  model.load_state_dict(torch.load(filepath))

class off_fun_eval_deri():
    def __init__(self, xL=-1, xR=1, N_1x=129):
        self.xL, self.xR= xL,xR
        self.xlen = xR-xL
        self.N_1x  = N_1x
        self.xChby = Chebyshev1D(xL=self.xL, xR=self.xR, M=self.N_1x-1)

        self.D1_x = self.xChby.DxCoeff(1)
        self.D2_x = self.xChby.DxCoeff(2)



    def svd_hign_sol(self,U,L):


        Usvd, s, Vh = linalg.svd(U)



        return Usvd[:,:L]

      # input phi, size [N+1], value of RB basis fun on the [N+1] mesh

    def Compute_d_dxc(self, phi):
        return np.matmul(self.D1_x,phi)

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


  #give the number of snapshot and reduced dim,
    off_ob = off_fun_eval_deri(xL=-1, xR=1, N_1x=self.N_1x)


    Psi_RB = off_ob.svd_hign_sol(sol_snap[:,:snap_num],L)


# onlinein [Nh,L]

    onlinein = Psi_RB


    return onlinein


  def compute_jaches_ksi(self,onlinein, L, xL=-1, xR=1, N_1x=129, Xdim=1):

    off_ob = off_fun_eval_deri(xL=-1, xR=1, N_1x=N_1x)


    off_jac = np.zeros((N_1x,L))

    off_hes = np.zeros((N_1x,L))


# onlinein [Nh,L]
    phi = onlinein



    off_jac[:,:] = off_ob.compute_off_no_theta(phi,'u_xc')


    off_hes[:,:] = off_ob.compute_off_no_theta(phi,'u_xc2')
# off_jac [Nh,L] off_hes [Nh,L]
    return off_jac, off_hes



  def eigen_snap(self,sol_snap, snap_num, L, theta):


      accu = 0.9999999
#(Nh-2,79)
#   [(N+1)**2,Ns]
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
#line3, = ax.plot(X_star, U_pred_pinn, dashes=[6, 2], label='Exact solution u(x)')
      plt.xlabel('L')
      plt.ylabel('Error')
      plt.legend()
      plt.show()


      return Usvd_u

class CustomedEqs():
    def __init__(self, M, h=10, Num_par1 = 10, Num_par2 = 10, Num_label_par1 = 10, Num_label_par2 = 10, mesh_num=128):
        # h frequency in the sin term
        Cheb1D  = Chebyshev1D(M=mesh_num)
        self.h = h
        self.xgrid   = Cheb1D.grid().reshape(-1,1)
        self.alpha1 = np.linspace(1.,10.,num=Num_par1)
        self.alpha2 = np.linspace(1.,10.,num=Num_par2)
        alpha1mesh,alpha2mesh = np.meshgrid(self.alpha1,self.alpha2,indexing='xy')
        self.parameters = np.concatenate((alpha1mesh.reshape(-1,1),alpha2mesh.reshape(-1,1)),axis=1)
        self.design_space = np.array([[1,1],[10,10]])

        # [Nh,Nsam]
        self.Samples = self.phix(self.xgrid.T,alpha1mesh.reshape(-1,1),alpha2mesh.reshape(-1,1)).T

        self.Np      = self.Samples.shape[0]-1
        self.NSample = self.Samples.shape[1]

        # svd decomposition
        self.Modes, self.sigma, _ = np.linalg.svd(self.Samples);
        self.Modes = self.Modes[:,:M]
        self.M = M
        # spatial discretization

        self.dx      = Cheb1D.DxCoeff();
        self.d2x     = Cheb1D.DxCoeff(2);
        self.projections = np.matmul( self.Modes.T, self.Samples)
        _, Mapping  = Normalization.Mapstatic(self.projections.T)
        self.proj_mean =  Mapping[0][None,:]
        self.proj_std  =  Mapping[1][None,:]

       # the labelled data, a different set
        self.alpha1_label = np.linspace(1.1,9.9,num=Num_label_par1)
        self.alpha2_label = np.linspace(1.1,9.9,num=Num_label_par2)
        alpha1mesh_label,alpha2mesh_label = np.meshgrid(self.alpha1_label,self.alpha2_label,indexing='xy')

        self.parameters_label = np.concatenate((alpha1mesh_label.reshape(-1,1),alpha2mesh_label.reshape(-1,1)),axis=1)

        # phix: [Nsam,Nh]
        # self.Samples_label: [Nh,Nsam]
        self.Samples_label = self.phix(self.xgrid.T,alpha1mesh_label.reshape(-1,1),alpha2mesh_label.reshape(-1,1)).T
        # [Nsam,Nh]
        self.Samples_label_normalize, self.Samples_min_val, self.Samples_max_val = self.normalize_to_minus_one_one(self.Samples_label.T)

    def normalize_to_minus_one_one(self,data):
        first_column = data[:, [0]]
        last_column = data[:, [-1]]

        # Select the middle columns for normalization
        middle_columns = data[:, 1:-1]

        # Normalize the middle columns
        min_val = np.min(middle_columns, axis=0, keepdims=True)
        max_val = np.max(middle_columns, axis=0, keepdims=True)
        normalized_middle = 2 * (middle_columns - min_val) / (max_val - min_val) - 1

        # Concatenate the columns back together
        normalized_data = np.hstack((first_column, normalized_middle, last_column))
        return normalized_data, min_val, max_val

    # Function to denormalize data from [-1, 1] back to original range
    def denormalize_from_minus_one_one(self, normalized_data, min_val, max_val):

        first_column = normalized_data[:, [0]]
        last_column = normalized_data[:, [-1]]

        # Select the middle columns for denormalization
        normalized_middle = normalized_data[:, 1:-1]

        # Denormalize the middle columns
        middle_columns = (normalized_middle + 1) / 2 * (max_val - min_val) + min_val

        # Concatenate the columns back together
        denormalized_data = np.hstack((first_column, middle_columns, last_column))
        return denormalized_data


    def plot_cheb_poly(self, x_points, coeffs):

        plot_cheb_fun(x_points,coeffs)


    def getA(self):
        V_x = np.matmul(self.dx, self.Modes);
        A = np.matmul( self.Modes.reshape((-1,self.M,1)), V_x.reshape(-1,1,self.M))
        A[0,:,:] = 0
        A[-1,:,:]=0
        A = A[None,:]*self.Modes.T.reshape((self.M, -1,1,1))
        A = A.sum(axis=1).squeeze().reshape((self.M, self.M, self.M))

        return A

    def getB(self):
        tmp =-a*self.d2x
        # add boundary conditions
        tmp[0,:] =0
        tmp[-1, :]=0
        tmp[0,0] =1
        tmp[-1,-1]=1

        tmp = np.matmul(self.Modes.T, tmp)
        B = np.matmul(tmp,self.Modes)
        return B

    def POD_G(self,Mchoose, alpha):
        alpha1 = alpha[:,0:1]
        alpha2 = alpha[:,1:2]
        n = alpha1.shape[0]
        lamda  = np.zeros((alpha.shape[0], self.M))
        def compute_eAe(A, e):
            tmp  = np.matmul(e.T, A)
            return np.matmul(tmp, e).squeeze(axis=(2))
        def compute_dA(A,e):
            return np.matmul(A+A.transpose((0, 2,1)), e).squeeze(axis=(2))

        A = self.getA()
        B = self.getB()
        for i in range(n):
            alpha1i = alpha1[i:i+1,0:1]; alpha2i = alpha2[i:i+1, 0:1];
            source = self.getsource(alpha1i, alpha2i).T
            dis = alpha[i:i+1,:] - self.parameters
            dis = np.linalg.norm(dis, axis=1);
            ind = np.where(dis == dis.min())[0][0]
            lamda0 = self.projections[0:self.M, ind:ind+1]
            #Newton iteration
            it = 0; err =1;
            while it<=Newton['iterMax'] and err>Newton['eps']:
                it +=1
                R0 = compute_eAe(A,lamda0) + np.matmul(B,lamda0) - source
                err = np.linalg.norm(R0)
                dR = compute_dA(A,lamda0) + B
                dlamda = -np.linalg.solve(dR, R0)
                lamda0 =lamda0 + dlamda
            if it>=Newton['iterMax']:
                print('Case (%f,%f) can only reach to an error of %f'%(alpha1[i,0], alpha2[i,0], err))
                lamda0 = lamda0*0 + np.inf
            lamda[i,:] = lamda0.squeeze()
        return lamda

    def GetError(self,alpha,lamda):
        alpha1 = alpha[:,0:1]
        alpha2 = alpha[:,1:2]
        phi_pred         = np.matmul( lamda, self.Modes.T)
        phi_Exact        = self.phix(self.xgrid.T, alpha1, alpha2)
        Error = np.linalg.norm(phi_Exact-phi_pred, axis = 1)/np.linalg.norm(phi_Exact, axis=1)
        Error = Error[None,:]
        Error = Error.mean()
        return Error
    def GetProjError(self, alpha):
        alpha1 = alpha[:,0:1]
        alpha2 = alpha[:,1:2]
        phi_Proj = self.phix(self.xgrid.T, alpha1, alpha2)
        lamda_Proj = np.matmul( phi_Proj, self.Modes )
        return self.GetError(alpha,lamda_Proj)


    def getsource(self,alpha1, alpha2):
      # x [1,129]
      # alpha1, alpha2[Nresi,1]
      # f1 [Nresi,129],f2 [Nresi,129],f1_x [Nresi,129]
        x = self.xgrid.T
        f1    = (1+alpha1*x)*(x**2-1)
        f2    = np.sin(-self.h*alpha2*x/3)
        f1_x  = 3*alpha1*x**2 +2*x -alpha1
        f2_x  = -self.h*alpha2/3*np.cos(-self.h*alpha2*x/3)
        f1_xx = 6*alpha1*x + 2
        f2_xx = -self.h**2*alpha2**2/9*np.sin(-self.h*alpha2*x/3)

        phi_x  = f1*f2_x + f1_x*f2
        phi_xx = 2*f1_x*f2_x + f1*f2_xx +f1_xx*f2
        # [Nresi,129]
        source = f1*f2*phi_x - phi_xx
        # [Nresi,129]
        source[:, 0:1] =self.phix(x[0,  0], alpha1, alpha2)
        source[:,-1: ] =self.phix(x[0, -1], alpha1, alpha2)
        ###source = np.matmul( source, self.Modes )

        # [Nresi,127]
        return source[:,1:-1]

    def phix_x(self,x,alpha1,alpha2):
        # x [1,x_num]
       # alpha1, alpha2[Nresi,1]
       # f1 [Nresi,x_num],f2 [Nresi,x_num],f1_x [Nresi,x_num]
        f1    = (1+alpha1*x)*(x**2-1)
        f2    = torch.sin(-self.h*alpha2*x/3)
        f1_x  = 3*alpha1*x**2 +2*x -alpha1
        f2_x  = -self.h*alpha2/3*torch.cos(-self.h*alpha2*x/3)
        phi_x  = f1*f2_x + f1_x*f2

        return phi_x

    def phix(self,x,alpha1,alpha2):
        return np.sin(-self.h*alpha2*x/3)*(1+alpha1*x)*(x**2-1)

class CustomedNet(para_Net):
    def __init__(self, rvalue, H1, layers1=None,oldnetfile1=None, reduce_net = None
                 ,roeqs=None,initial=False,initW=torch.zeros(5,5), initb=torch.zeros(5), mesh_num=128, omega_given=1.0):
        super(CustomedNet, self).__init__(layers1=layers1,OldNetfile1=oldnetfile1)
        self.M = roeqs.M
        self.A = torch.tensor( roeqs.getA() ).float().to(DEVICE)
        self.B = torch.tensor( roeqs.getB() ).float().to(DEVICE)
        self.lb = torch.tensor(roeqs.design_space[0:1,:]).float().to(DEVICE)
        self.ub = torch.tensor(roeqs.design_space[1:2,:]).float().to(DEVICE)
        self.roeqs = roeqs
        self.proj_std = torch.tensor( roeqs.proj_std ).float().to(DEVICE)
        self.proj_mean= torch.tensor( roeqs.proj_mean).float().to(DEVICE)
        self.H1 = H1
        self.rvalue = rvalue
        self.reduce_net = reduce_net

        self.labeled_inputs  = torch.tensor( roeqs.parameters ).float().to(DEVICE)
        self.labeled_outputs = torch.tensor( roeqs.projections.T ).float().to(DEVICE)
        self.source = roeqs.getsource(roeqs.parameters[:,0:1], roeqs.parameters[:,1:2])
        self.source = torch.tensor( self.source ).float().to(DEVICE)
        ###self.labeledLoss = self.loss_Eqs(self.labeled_inputs,self.labeled_outputs, self.source)
        self.initial = initial
        self.initW = initW
        self.initb = initb
        self.omega  = omega_given

        self.off_obj = off_all(mesh_num+1)
        #             onlinein [Nh,L]
        offout = self.off_obj.RB_fun(self.roeqs.Samples,self.roeqs.NSample,self.M)

        normalized_offout, min_val, max_val = self.normalize_to_minus_one_one(offout)

        offout = normalized_offout


        self.offout = torch.tensor(offout, requires_grad=True).float().to(DEVICE)
        self.offout_ins = torch.tensor(offout[1:-1,:], requires_grad=True).float().to(DEVICE)

        self.off_jac_ksi, self.off_hes_ksi = self.off_obj.compute_jaches_ksi(offout, self.M)
        # [Nh_inside,L]
        self.off_jac_ksi_ins = torch.tensor( self.off_jac_ksi[1:-1,:] ).float().to(DEVICE)
        self.off_hes_ksi_ins = torch.tensor( self.off_hes_ksi[1:-1,:] ).float().to(DEVICE)

    def normalize_to_minus_one_one(self,data):
        """
        Normalize the input array to the range [-1, 1].

        Parameters:
        data (numpy array): The input array to normalize.

        Returns:
        tuple: Normalized array, original min values, original max values.
        """
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
        return normalized_data, min_val, max_val


# from mu to weights
    def u_net(self,x):
        x = (x-(self.ub+self.lb)/2)/(self.ub-self.lb)*2
        out = self.unet1(x)
        return out


    def forward(self,x):
        return self.u_net(x).detach().cpu().numpy()

    def initial_w(self):

        if not(self.initial):

            for m in self.modules():

              if isinstance(m,nn.Linear):

                print(m.weight)

                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

        else:

            i=0

            for m in self.modules():

              if isinstance(m,nn.Linear):

                m.weight = self.initW[i]
                m.bias = self.initb[i]
                i=i+1

    def inside_x(self,Array):

        return Array[1:-1,:]

    def cheb_net_f(self,u):
        # u [Nh,Nsam], numpy
        off_ob = off_fun_eval_deri(xL=-1, xR=1, N_1x=u.shape[0])
        # off_jac [Nh,Nsam]
        off_jac = off_ob.compute_off_no_theta(u,'u_xc')

        off_hes = off_ob.compute_off_no_theta(u,'u_xc2')
        # [Nh_in,Nsam]
        return u[1:-1,:], off_jac[1:-1,:], off_hes[1:-1,:]

    def cheb_loss(self,u_on,source):
        # [Nh_in,Nsam]
        u,u_x,u_x_x  = self.cheb_net_f(u_on)
        # [Nh_in,Nsam]>>>[Nh_in]
        f1 = np.mean((u*u_x-u_x_x-source)**2,0)

        bc_value = u_on[np.array([0, -1]),:]

        fbc = np.mean((bc_value)**2,0)
         #        [Nh_in]>>R
        loss = np.mean(f1+fbc*self.omega)

        return loss


    def vmap_2nd_deri(self,offd1,offd2,ond1,ond2):

        B = torch.sum(ond2*offd1,2)

        hat_d_d = torch.sum(torch.transpose(B, 0, 1) * offd1,1) + torch.sum(ond1*offd2,1)

        return hat_d_d

    def To_phir(self,offout):

        Phi = self.reduce_net(offout)

        return Phi

    def Phi_deri(self, offout):

        compute_batch_jacobian = vmap(jacrev(self.To_phir, argnums=0), in_dims=(0), randomness='same')
        # [Num_x,r,L]
        batch_jacobian0 = compute_batch_jacobian(offout)

        compute_batch_hessian = vmap(torch.func.jacfwd(jacrev(self.To_phir, argnums=0), argnums=0, randomness='same'),
                                                       in_dims=(0), randomness='same')
        # [Num_x,r,L,L]
        batch_hessian0 = compute_batch_hessian(offout)

        return batch_jacobian0, batch_hessian0

    def Phi_x(self,phi_jac_mat, phi_hes_mat,prime_x,prime_xx):

        Phir_x = torch.sum(torch.transpose(phi_jac_mat, 0, 1)*prime_x,2)

        Phi_xx_vmap =vmap(self.vmap_2nd_deri, in_dims=(None, None, 1, 1), randomness='same')

        Phir_xx = Phi_xx_vmap(prime_x, prime_xx, phi_jac_mat, torch.transpose(phi_hes_mat, 0, 2))

        return torch.transpose(Phir_x, 0, 1), torch.transpose(Phir_xx, 0, 1)



    def muNN_out_to_weight(self, mu_NN_out):

        # mu_NN_out [Nresi,H1*L+2H1+1]
        # W1 [Nresi,L,H1]
        # b1 [Nresi,1,H1]
        # W2 [Nresi,H1,1]
        # b1 [Nresi,1,1]

        W1 = mu_NN_out[:,:self.H1*self.rvalue].view(-1,self.rvalue,self.H1)
        b1 = mu_NN_out[:,self.H1*self.rvalue:self.H1*self.rvalue+self.H1].view(-1,1,self.H1)
        W2 = mu_NN_out[:,self.H1*self.rvalue+self.H1:self.H1*self.rvalue+self.H1+self.H1].view(-1,self.H1,1)
        b2 = mu_NN_out[:,self.H1*self.rvalue+self.H1+self.H1:].view(-1,1,1)
        return W1,b1,W2,b2




    def on_fun(self, off_out_mesh, W1, b1, W2, b2):

        return (torch.tanh(off_out_mesh@W1+b1))@W2+b2


    def net_nbc(self, on_in_bc,W1,b1,W2,b2):

        allout = self.on_fun(on_in_bc,W1,b1,W2,b2)

#       u [Nresi,len(on_in_bc)]
        ulb = allout[:,0,0]

        uub = allout[:,-1,0]

        return ulb,uub

    def compute_deri_2_jaches2(self, online_in, dnn_fun, W1, b1, W2, b2):

        '''

         dnn_fun a neural network function map online_in [Num_x,L] to uhat[Num_x,1]

         return the output of online nn dnn_fun. also return the jacobian and hessian of dnn_fun output w.r.t.

        '''

        online_in = torch.tensor(online_in, requires_grad=True).float()

        compute_batch_output = vmap(vmap(vmap(dnn_fun, in_dims=(0, None, None, None, None), randomness='same'),
                                    in_dims=(None,0,0,0,0), randomness='same'), in_dims=(None,None,1,None,1), randomness='same')

        batch_out = compute_batch_output(online_in, W1, b1, W2, b2)[0]

        compute_batch_jacobian = vmap(vmap(vmap(jacrev(dnn_fun, argnums=0), in_dims=(0, None, None, None, None), randomness='same'),
                                    in_dims=(None,0,0,0,0), randomness='same'), in_dims=(None,None,1,None,1), randomness='same')


        batch_jacobian0 = compute_batch_jacobian(online_in, W1, b1, W2, b2)[0]

        compute_batch_hessian = vmap(vmap(vmap(torch.func.jacfwd(jacrev(dnn_fun, argnums=0), argnums=0, randomness='same'),
                                                       in_dims=(0, None, None, None, None), randomness='same'),
                                                       in_dims=(None,0,0,0,0), randomness='same'), in_dims=(None,None,1,None,1), randomness='same')


        batch_hessian0 = compute_batch_hessian(online_in, W1, b1, W2, b2)[0]

        return batch_out,batch_jacobian0[:,:,0,:],torch.transpose(batch_hessian0[:,:,0,:,:], 1, 2)



    def online_deri(self, online_in, W1, b1, W2, b2):
      #       [Nresi,Num_x,H1]
        l1 = torch.matmul(online_in,W1) + b1

        uhat_jac = torch.matmul((torch.transpose(W2, 1, 2)*(1-torch.tanh(l1)**2)),torch.transpose(W1, 1, 2))

        uhat_hes = torch.matmul(torch.matmul(torch.matmul(torch.matmul(W1,torch.diag_embed(W2[:,:,0])), torch.diag_embed(-2*torch.tanh(torch.transpose(l1, 0, 1)))) , torch.diag_embed(1-torch.tanh(torch.transpose(l1, 0, 1))**2)),torch.transpose(W1, 1, 2))

        return uhat_jac, torch.transpose(torch.transpose(uhat_hes, 0, 2), 0, 1)





    def compute_deri_2(self, uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, kind):

        '''
         batch_jacobian shape [Nh,L], the partial derivative of reduced basis function w.r.t. input x
         First dim is number of different collocation points, second dim is each RB functions

         batch_hess shape [Nh,L]
         First dim is number of different collocation points, second dim is each RB functions,

         dnn_fun a neural network function map online_in to u

        '''

        prime_x = batch_jacobian

        prime_xx = batch_hess

        if kind == 'u_x':
# uhat_jac_mat [Nresi,Num_x,L] , prime_x [Num_x,L], uhat_jac_mat*prime_x return [Nresi,Num_x,L]
          u_x = torch.sum(uhat_jac_mat*prime_x,2)
#.      [Nresi,Num_x]
          return u_x





########## Two ways, another way to compute uhat_y_y is vmap
###########################################################################
        elif kind == 'u_x_x':

          uhat_x_x_vmap = vmap(self.vmap_2nd_deri, in_dims=(None, None, 0, 0), randomness='same')
          uhat_x_x = uhat_x_x_vmap(prime_x, prime_xx, uhat_jac_mat, uhat_hes_mat)
          return uhat_x_x

        elif kind == 'u':

          return allout[:,:,0]

        else:
            
          print('Please type a valid kind of derivative!')


    def net_f(self,source,uhat_jac_mat,uhat_hes_mat, allout, off_tr_jac, off_tr_hes):

        u = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, off_tr_jac, off_tr_hes, 'u')

        u_x = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, off_tr_jac, off_tr_hes, 'u_x')

        u_x_x = self.compute_deri_2(uhat_jac_mat,uhat_hes_mat, allout, off_tr_jac, off_tr_hes, 'u_x_x')
        #    [Nresi,x_num]
        d = 0.1*(abs(u_x))+1

        f1 = torch.mean(((u*u_x-u_x_x-source)/d)**2,dim=1)

        return f1

    def grad_penalty(self,x):

        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(x))

        Phi = self.To_phir(self.offout)

        phi_jac_mat, phi_hes_mat = self.Phi_deri(self.offout_ins)

        Phir_x, Phir_xx = self.Phi_x(phi_jac_mat, phi_hes_mat,self.off_jac_ksi_ins, self.off_hes_ksi_ins)

        allout = self.on_fun(Phi,W1_mu, b1_mu,W2_mu, b2_mu)

        uhat_jac_mat,uhat_hes_mat = self.online_deri(Phi,W1_mu, b1_mu,W2_mu, b2_mu)

        u_x = self.compute_deri_2(uhat_jac_mat[:,1:-1,:],uhat_hes_mat[:,:,1:-1,:], allout[:,1:-1,:],Phir_x, Phir_xx, 'u_x')

        return torch.mean(u_x**2)


    def loss_Strong_Residue(self,x,source,weight=1,lamba1=1.,lamba2=1.,lamba3=1.):

        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(x))

        Phi = self.To_phir(self.offout)

        phi_jac_mat, phi_hes_mat = self.Phi_deri(self.offout_ins)

        Phir_x, Phir_xx = self.Phi_x(phi_jac_mat, phi_hes_mat,self.off_jac_ksi_ins, self.off_hes_ksi_ins)

        allout = self.on_fun(Phi,W1_mu, b1_mu,W2_mu, b2_mu)

        uhat_jac_mat,uhat_hes_mat = self.online_deri(Phi,W1_mu, b1_mu,W2_mu, b2_mu)

        f1_mean = self.net_f(source,uhat_jac_mat[:,1:-1,:],uhat_hes_mat[:,:,1:-1,:], allout[:,1:-1,:],Phir_x, Phir_xx)

        bc_value = self.on_fun(Phi[np.array([0, -1]),:],W1_mu,b1_mu,W2_mu,b2_mu)

        fbc = torch.mean((bc_value[:,:,0])**2,dim=1)

        loss = torch.mean((weight*(f1_mean+fbc))**2)

        return loss

    def loss_kernel_grad(self, x, K):

        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(x))

        Phi = self.To_phir(self.offout)

        on_out = self.on_fun(Phi, W1_mu, b1_mu, W2_mu, b2_mu)
        # [Nresi,Num_x,1]
        u_pred = on_out

        loss = (torch.transpose(u_pred, 1, 2)@K)@u_pred

        if loss.shape != (x.shape[0],1, 1):
            raise ValueError(f"loss_kernel_grad must have shape [Num_val, 1, 1], but got shape {loss.shape}")

        return torch.mean(loss**2)


    def loss_label(self, x, u_star):

        # x:labeled_inputs,  [Nsam,2]
        # u_star: labeled_outputs, [Nsam,Nh]

        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(x))

        Phi = self.To_phir(self.offout)

        on_out = self.on_fun(Phi, W1_mu, b1_mu, W2_mu, b2_mu)
        # [Nresi,Num_x,1]
        u_pred = on_out[:,:,0]

        loss = torch.mean((u_pred - u_star)**2)

        return loss

    def loss_label_grad(self, x):

        # x:labeled_inputs,  [Nsam,2]
        # u_star_x: labeled_outputs, [Nsam,Nh]

        # In burger problem, we use the gradient of exact solution. In other more general problems, we need the given approximation of gradient (Adjoint state method)
        u_star_x = self.roeqs.phix_x(torch.from_numpy(self.roeqs.xgrid[1:-1,:].T).to(DEVICE),x[:,0:1],x[:,1:2])       

        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(x))

        Phi = self.To_phir(self.offout)

        phi_jac_mat, phi_hes_mat = self.Phi_deri(self.offout_ins)

        Phir_x, Phir_xx = self.Phi_x(phi_jac_mat, phi_hes_mat,self.off_jac_ksi_ins, self.off_hes_ksi_ins)

        allout = self.on_fun(Phi,W1_mu, b1_mu,W2_mu, b2_mu)

        uhat_jac_mat,uhat_hes_mat = self.online_deri(Phi,W1_mu, b1_mu,W2_mu, b2_mu)

        u_x = self.compute_deri_2(uhat_jac_mat[:,1:-1,:],uhat_hes_mat[:,:,1:-1,:], allout[:,1:-1,:],Phir_x, Phir_xx, 'u_x')

        loss = torch.mean((u_x - u_star_x)**2)

        return loss

    def Cheb_loss(self, x, source):

        source = source.detach().cpu().numpy()

        # x:labeled_inputs,  [Nsam,2]
        # u_star: labeled_outputs, [Nsam,Nh]
        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(x))
        
        Phi = self.To_phir(self.offout)

        on_out = self.on_fun(Phi, W1_mu, b1_mu, W2_mu, b2_mu)[:,:,0].detach().cpu().numpy()
        # [Nresi,Num_x]
        err = self.cheb_loss(on_out.T,source.T)

        return err

    def error_val(self, x, u_star):

        # x:labeled_inputs,  [Nsam,2]
        # u_star: labeled_outputs, [Nsam,Nh]
        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(x))
        
        Phi = self.To_phir(self.offout)

        on_out = self.on_fun(Phi, W1_mu, b1_mu, W2_mu, b2_mu)
        # [Nresi,Num_x]
        u_pred = on_out[:,:,0].detach().cpu().numpy()

        err = LA.norm((u_pred - u_star),axis=1)/LA.norm(u_star,axis=1)

        return err
