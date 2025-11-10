import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
from Chebyshev import Chebyshev1D
import torch
from NN_bc_3NN_split import POD_Net, DEVICE
from Clenshaw_Curtis import clenshaw_curtis_quadrature_2d_discrete, clenshaw_curtis_quadrature_2d_torch
from Normalization import Normalization
from Generate_lid_data import get_inter_2d_inx
from torch.func import jacrev, jacfwd
from torch import nn
import random
from torch.func import vmap, vjp
import torch.nn.functional as F
from random import shuffle
from scipy.io import loadmat
import numpy as np
import scipy.io
import time
from scipy import linalg
import matplotlib.pyplot as plt



NVAR = 3     # the number of unknown variables: p,u,v
NVARLOAD = 3 

torch.manual_seed(1234879)
np.random.seed(1234879)


class off_fun_eval_deri():
    def __init__(self, xL=-1, xR=1, yD=-1, yU=1, N_1x=49, N_1y=49):
        self.xL, self.xR, self.yD,self.yU = xL,xR,yD,yU
        self.xlen = xR-xL
        self.ylen = yU-yD

        self.N_1x  = N_1x
        self.N_1y  = N_1y

        self.xChby = Chebyshev1D(xL=self.xL, xR=self.xR, M=self.N_1x-1)
        self.yChby = Chebyshev1D(xL=self.yD, xR=self.yU, M=self.N_1y-1)

        self.D1_x = self.xChby.DxCoeff(1)
        self.D1_y = self.yChby.DxCoeff(1)

        self.D1p_x = self.xChby.DxCoeffN2()
        self.D1p_y = self.yChby.DxCoeffN2()

    def grid(self):
        y,x = np.meshgrid(self.yChby.grid(), self.xChby.grid())
        return x,y

    def svd_hign_sol(self,U,L):

        Usvd, s, Vh = linalg.svd(U)

        Psi_u = np.zeros((L,self.N_1x,self.N_1y))

        for i in range(L):

            Psi_u[i,:,:] = Usvd[:,i].reshape((self.N_1x,self.N_1y))

        return Psi_u


    def Compute_d_dxc(self, phi):
        return np.matmul(self.D1_x,phi)
    def Compute_d_dyc(self, phi):
        return np.matmul(self.D1_y, phi.T).T
    def Compute_dp_dxc(self, phi):
        return np.matmul(self.D1p_x,phi)
    def Compute_dp_dyc(self, phi):
        return np.matmul(self.D1p_y, phi.T).T
    def Compute_d_dxc2(self, phi):
        return self.Compute_d_dxc( self.Compute_d_dxc(phi) )
    def Compute_d_dyc2(self, phi):
        return self.Compute_d_dyc( self.Compute_d_dyc(phi) )

    def Compute_d_dxcyc(self, phi):
        return self.Compute_d_dyc( self.Compute_d_dxc(phi) )

    def Compute_d_dycxc(self, phi):
        return self.Compute_d_dxc( self.Compute_d_dyc(phi) )


    def Compute_dp_dxc2(self, phi):
        return self.Compute_dp_dxc( self.Compute_dp_dxc(phi) )
    def Compute_dp_dyc2(self, phi):
        return self.Compute_dp_dyc( self.Compute_dp_dyc(phi) )

    def Compute_dp_dxcyc(self, phi):
        return self.Compute_dp_dyc( self.Compute_dp_dxc(phi) )

    def Compute_dp_dycxc(self, phi):
        return self.Compute_dp_dxc( self.Compute_dp_dyc(phi) )


    def Compute_d_d1(self, phi):
        return self.Compute_d_dxc(phi), self.Compute_d_dyc(phi)
    def Compute_d_d1p(self, phi):
        return self.Compute_dp_dxc(phi), self.Compute_dp_dyc(phi)
    def Compute_d_d2(self, phi):
        return self.Compute_d_dxc2(phi), self.Compute_d_dyc2(phi), self.Compute_d_dxcyc(phi)

    # ksi >>> u,v,p. Get u_ksi...
    def compute_off_no_theta(self,phi,kind):

        if kind == 'u_xc':

            return self.Compute_d_dxc(phi)

        elif kind == 'u_yc':

            return self.Compute_d_dyc(phi)

        elif kind == 'p_xc':

            return self.Compute_dp_dxc(phi)

        elif kind == 'p_yc':

            return self.Compute_dp_dyc(phi)

        elif kind == 'u_xc2':

            return self.Compute_d_dxc2(phi)

        elif kind == 'u_yc2':

            return self.Compute_d_dyc2(phi)

        elif kind == 'u_xcyc':

            return self.Compute_d_dxcyc(phi)

        elif kind == 'u_ycxc':

            return self.Compute_d_dycxc(phi)

        elif kind == 'p_xc2':

            return self.Compute_dp_dxc2(phi)

        elif kind == 'p_yc2':

            return self.Compute_dp_dyc2(phi)

        elif kind == 'p_xcyc':

            return self.Compute_dp_dxcyc(phi)

        elif kind == 'p_ycxc':

            return self.Compute_dp_dycxc(phi)

        else:

            print('Please give a valid type')


    def getJac(self,Theta, cos=np.cos, sin=np.sin, cat=np.concatenate):

        xCoef, yCoef = 1/2, 1/2
        #dksi1/dx
        Jac11=1/xCoef*(Theta*0+1)
        #dksi2/dx
        Jac12=0.0*Theta
        #dksi1/dy
        Jac21=-cos(Theta)/sin(Theta)/xCoef
        #dksi2/dy
        Jac22= 1/yCoef/sin(Theta)
        return Jac11,Jac12,Jac21,Jac22

class torch_off_fun_eval_deri():
    def __init__(self, xL=-1, xR=1, yD=-1, yU=1, N_1x=49, N_1y=49):
        self.xL, self.xR, self.yD,self.yU = xL,xR,yD,yU
        self.xlen = xR-xL
        self.ylen = yU-yD

        self.N_1x  = N_1x
        self.N_1y  = N_1y

        self.xChby = Chebyshev1D(xL=self.xL, xR=self.xR, M=self.N_1x-1)
        self.yChby = Chebyshev1D(xL=self.yD, xR=self.yU, M=self.N_1y-1)

        self.D1_x = torch.tensor( self.xChby.DxCoeff(1) ).float().to(DEVICE)

        self.D1_y = torch.tensor( self.yChby.DxCoeff(1) ).float().to(DEVICE)

        self.D1p_x = torch.tensor( self.xChby.DxCoeffN2() ).float().to(DEVICE)

        self.D1p_y =  torch.tensor( self.yChby.DxCoeffN2() ).float().to(DEVICE)


    def Compute_d_dxc(self, phi):
        return torch.matmul(self.D1_x,phi)
    def Compute_d_dyc(self, phi):
        return torch.transpose(torch.matmul(self.D1_y, torch.transpose(phi,1,2)),1,2)
    def Compute_dp_dxc(self, phi):
        return torch.matmul(self.D1p_x,phi)
    def Compute_dp_dyc(self, phi):
        return torch.transpose(torch.matmul(self.D1p_y, torch.transpose(phi,1,2)),1,2)
    def Compute_d_dxc2(self, phi):
        return self.Compute_d_dxc( self.Compute_d_dxc(phi) )
    def Compute_d_dyc2(self, phi):
        return self.Compute_d_dyc( self.Compute_d_dyc(phi) )

    def Compute_d_dxcyc(self, phi):
        return self.Compute_d_dyc( self.Compute_d_dxc(phi) )

    def Compute_d_dycxc(self, phi):
        return self.Compute_d_dxc( self.Compute_d_dyc(phi) )


    def Compute_dp_dxc2(self, phi):
        return self.Compute_dp_dxc( self.Compute_dp_dxc(phi) )
    def Compute_dp_dyc2(self, phi):
        return self.Compute_dp_dyc( self.Compute_dp_dyc(phi) )

    def Compute_dp_dxcyc(self, phi):
        return self.Compute_dp_dyc( self.Compute_dp_dxc(phi) )

    def Compute_dp_dycxc(self, phi):
        return self.Compute_dp_dxc( self.Compute_dp_dyc(phi) )


    def Compute_d_d1(self, phi):
        return self.Compute_d_dxc(phi), self.Compute_d_dyc(phi)
    def Compute_d_d1p(self, phi):
        return self.Compute_dp_dxc(phi), self.Compute_dp_dyc(phi)
    def Compute_d_d2(self, phi):
        return self.Compute_d_dxc2(phi), self.Compute_d_dyc2(phi), self.Compute_d_dxcyc(phi)

    def compute_off_no_theta(self,phi,kind):

        if kind == 'u_xc':

            return self.Compute_d_dxc(phi)

        elif kind == 'u_yc':

            return self.Compute_d_dyc(phi)

        elif kind == 'p_xc':

            return self.Compute_dp_dxc(phi)

        elif kind == 'p_yc':

            return self.Compute_dp_dyc(phi)

        elif kind == 'u_xc2':

            return self.Compute_d_dxc2(phi)

        elif kind == 'u_yc2':

            return self.Compute_d_dyc2(phi)

        elif kind == 'u_xcyc':

            return self.Compute_d_dxcyc(phi)

        elif kind == 'u_ycxc':

            return self.Compute_d_dycxc(phi)

        elif kind == 'p_xc2':

            return self.Compute_dp_dxc2(phi)

        elif kind == 'p_yc2':

            return self.Compute_dp_dyc2(phi)

        elif kind == 'p_xcyc':

            return self.Compute_dp_dxcyc(phi)

        elif kind == 'p_ycxc':

            return self.Compute_dp_dycxc(phi)

        else:

            print('Please give a valid type')



    def compute_jaches(self,theta, L, off_var_x, off_var_y, off_var_xx, off_var_xy, off_var_yx, off_var_yy, N_1x=49, N_1y=49):

        Nresi = theta.shape[0]

        xCoef, yCoef = 1/2, 1/2
        #dksi1/dx
        Jac11=(1/xCoef*(theta*0+1)).view(-1,1)
        #dksi2/dx
        Jac12=(0.0*theta).view(-1,1)
        #dksi1/dy
        Jac21=(-torch.cos(theta)/torch.sin(theta)/xCoef).view(-1,1)
        #dksi2/dy
        Jac22= (1/yCoef/torch.sin(theta)).view(-1,1)

        off_jac_x = off_var_x*Jac11 + off_var_y*Jac12

        off_jac_y = off_var_x*Jac21 + off_var_y*Jac22

        off_hes_xx = (off_var_xx*Jac11 + off_var_xy*Jac12)*Jac11 + (off_var_yx*Jac11 + off_var_yy*Jac12)*Jac12

        off_hes_yy = (off_var_xx*Jac21 + off_var_xy*Jac22)*Jac21 + (off_var_yx*Jac21 + off_var_yy*Jac22)*Jac22

        return off_jac_x, off_jac_y, off_hes_xx, off_hes_yy


    def compute_jaches_p(self,theta, L, off_var_x, off_var_y, N_1x=49, N_1y=49):

        Nresi = theta.shape[0]

        xCoef, yCoef = 1/2, 1/2
        #dksi1/dx
        Jac11=(1/xCoef*(theta*0+1)).view(-1,1)
        #dksi2/dx
        Jac12=(0.0*theta).view(-1,1)
        #dksi1/dy
        Jac21=(-torch.cos(theta)/torch.sin(theta)/xCoef).view(-1,1)
        #dksi2/dy
        Jac22= (1/yCoef/torch.sin(theta)).view(-1,1)

        off_jac_x = off_var_x*Jac11 + off_var_y*Jac12

        off_jac_y = off_var_x*Jac21 + off_var_y*Jac22

        return off_jac_x, off_jac_y

class off_all():

    def __init__(self, N_1x,N_1y):

        self.N_1x = N_1x

        self.N_1y = N_1y


    def RB_fun(self, sol_snap, snap_num, L,xL=-1, xR=1, yD=-1, yU=1, N_1x=49, N_1y=49):

        off_ob = off_fun_eval_deri(xL=-1, xR=1, yD=-1, yU=1, N_1x=N_1x, N_1y=N_1y)

        P_snap = sol_snap[0::3,:snap_num]

        U_snap = sol_snap[1::3,:snap_num]

        V_snap = sol_snap[2::3,:snap_num]

        P_RB = off_ob.svd_hign_sol(P_snap,L)

        U_RB = off_ob.svd_hign_sol(U_snap,L)

        V_RB = off_ob.svd_hign_sol(V_snap,L)

        onlinein = np.zeros((N_1x*N_1y,3*L))

        for i in range(L):

            onlinein[:,i] = P_RB[i,:,:].reshape(-1)

        for j in range(L):

            onlinein[:,L+j] = U_RB[j,:,:].reshape(-1)

        for h in range(L):

            onlinein[:,2*L+h] = V_RB[h,:,:].reshape(-1)

        return np.concatenate((P_RB, U_RB, V_RB), axis=0), onlinein

    # compute all jacobian and hessian w.r.t. domain \xi
    def compute_jaches_ksi(self,phi_3, L, xL=-1, xR=1, yD=-1, yU=1, N_1x=49, N_1y=49):
    # phi_3 [3*L,N+1,N+1]

        off_ob = off_fun_eval_deri(xL=-1, xR=1, yD=-1, yU=1, N_1x=49, N_1y=49)

        off_jac = np.zeros((N_1x*N_1y,3*L,2))

        off_hes = np.zeros((N_1x*N_1y,3*L,2,2))

        for i in range(L):

            phi = phi_3[i,:,:]

            off_jac[:,i,0] = off_ob.compute_off_no_theta(phi,'p_xc').reshape(-1)

            off_jac[:,i,1] = off_ob.compute_off_no_theta(phi,'p_yc').reshape(-1)

            off_hes[:,i,0,0] = off_ob.compute_off_no_theta(phi,'p_xc2').reshape(-1)

            off_hes[:,i,0,1] = off_ob.compute_off_no_theta(phi,'p_xcyc').reshape(-1)

            off_hes[:,i,1,0] = off_ob.compute_off_no_theta(phi,'p_ycxc').reshape(-1)

            off_hes[:,i,1,1] = off_ob.compute_off_no_theta(phi,'p_yc2').reshape(-1)

        # du/dxc and dv/dyc ...
        for i in range(2*L):

            phi = phi_3[L+i,:,:]

            off_jac[:,L+i,0] = off_ob.compute_off_no_theta(phi,'u_xc').reshape(-1)

            off_jac[:,L+i,1] = off_ob.compute_off_no_theta(phi,'u_yc').reshape(-1)

            off_hes[:,L+i,0,0] = off_ob.compute_off_no_theta(phi,'u_xc2').reshape(-1)

            off_hes[:,L+i,0,1] = off_ob.compute_off_no_theta(phi,'u_xcyc').reshape(-1)

            off_hes[:,L+i,1,0] = off_ob.compute_off_no_theta(phi,'u_ycxc').reshape(-1)

            off_hes[:,L+i,1,1] = off_ob.compute_off_no_theta(phi,'u_yc2').reshape(-1)

        return off_jac, off_hes

    # Given jacobian and hessian w.r.t. computational domain \xi, compute all jacobian and hessian w.r.t. origin domain x
    def compute_jaches(self,theta, L, off_jac_ksi, off_hes_ksi, N_1x=49, N_1y=49):

        Nresi = theta.shape[0]

        off_jac = np.zeros((Nresi,N_1x*N_1y,L,2))

        off_hes = np.zeros((Nresi,N_1x*N_1y,L,2,2))


        xCoef, yCoef = 1/2, 1/2
        #dksi1/dx
        Jac11=(1/xCoef*(theta*0+1)).reshape(-1,1)
        #dksi2/dx
        Jac12=(0.0*theta).reshape(-1,1)
        #dksi1/dy
        Jac21=(-np.cos(theta)/np.sin(theta)/xCoef).reshape(-1,1)
        #dksi2/dy
        Jac22= (1/yCoef/np.sin(theta)).reshape(-1,1)

        for i in range(L):

            off_jac[:,:,i,0] = off_jac_ksi[:,i,0]*Jac11 + off_jac_ksi[:,i,1]*Jac12

            off_jac[:,:,i,1] = off_jac_ksi[:,i,0]*Jac21 + off_jac_ksi[:,i,1]*Jac22
                            
            off_hes[:,:,i,0,0] = (off_hes_ksi[:,i,0,0]*Jac11 + off_hes_ksi[:,i,0,1]*Jac12)*Jac11 + (off_hes_ksi[:,i,1,0]*Jac11 + off_hes_ksi[:,i,1,1]*Jac12)*Jac12

            off_hes[:,:,i,1,1] = (off_hes_ksi[:,i,0,0]*Jac21 + off_hes_ksi[:,i,0,1]*Jac22)*Jac21 + (off_hes_ksi[:,i,1,0]*Jac21 + off_hes_ksi[:,i,1,1]*Jac22)*Jac22

        return off_jac, off_hes


    def compute_all(self, sol_snap,snap_num, L, theta):

        phi_3, onlinein = self.RB_fun(sol_snap,snap_num,L)

        off_jac_ksi, off_hes_ksi = self.compute_jaches_ksi(phi_3, L)

        off_jac, off_hes = self.compute_jaches(theta, L, off_jac_ksi, off_hes_ksi)
        #       [N_1x*N_1y,3*L], [Nresi,xnum,3L,2], [Nresi,xnum,3L,2,2]
        return onlinein, off_jac, off_hes


    def eigen_snap(self,sol_snap, snap_num, L, theta, L_lim=None):

        if L_lim:

            L_range = L_lim

        else:

            L_range = snap_num

        accu = 0.9999999

        P_snap = sol_snap[0::3,:snap_num]

        U_snap = sol_snap[1::3,:snap_num]

        V_snap = sol_snap[2::3,:snap_num]

        Usvd_p, s_p, Vh_p = linalg.svd(P_snap)

        Usvd_u, s_u, Vh_u = linalg.svd(U_snap)

        Usvd_v, s_v, Vh_v = linalg.svd(V_snap)

        sq_sum_p = np.sum(s_p**2)

        rate_p = 0

        sq_sum_u = np.sum(s_u**2)

        rate_u = 0

        sq_sum_v = np.sum(s_v**2)

        rate_v = 0

        L_i = 0

        Llist = []

        ratelist_p = []

        ratelist_u = []

        ratelist_v = []

        for i in range(L_range):

            L_i = i + 1

            Llist.append(L_i)

            rate_p = 1 - np.sum(s_p[:L_i]**2)/sq_sum_p

            ratelist_p.append(rate_p)

            rate_u = 1 - np.sum(s_u[:L_i]**2)/sq_sum_u

            ratelist_u.append(rate_u)

            rate_v = 1 - np.sum(s_v[:L_i]**2)/sq_sum_v

            ratelist_v.append(rate_v)

        Larr = np.array(Llist)

        ratearr_p = np.array(ratelist_p)

        ratearr_u = np.array(ratelist_u)

        ratearr_v = np.array(ratelist_v)

        fig, ax = plt.subplots()
        ax.plot(Larr,ratearr_p,label = 'p')
        ax.plot(Larr,ratearr_u,label = 'u')
        ax.plot(Larr,ratearr_v,label = 'v')

        fig.suptitle('Projection error with respect to L, high fidelity solution')
        plt.xlabel('L')
        plt.ylabel('Error')
        plt.legend()
        plt.show()


        return Usvd_p[:,:L], Usvd_p[:,:L], Usvd_p[:,:L]

def p_normalize_mat(p_all,N_1x,N_1y):

    p_norm_all = np.zeros_like(p_all)

    for i in range(p_all.shape[1]):

        p = p_all[:,i]

        p_int = clenshaw_curtis_quadrature_2d_discrete((p.reshape(1,N_1x-2,N_1y-2)), N_1x, N_1y, discard_endpoints=True)
        p_norm1 = p - p_int/4

        p_int2 = clenshaw_curtis_quadrature_2d_discrete((p_norm1.reshape(1,N_1x-2,N_1y-2)), N_1x, N_1y, discard_endpoints=True)
        p_norm2 = p_norm1 - p_int2/4

        p_int3 = clenshaw_curtis_quadrature_2d_discrete((p_norm2.reshape(1,N_1x-2,N_1y-2)), N_1x, N_1y, discard_endpoints=True)        
        p_norm_all[:,i] = p_norm2 - p_int3/4

    return p_norm_all

class CustomedEqs():
    def __init__(self, matfilePOD, PODNum, matfileValidation, M, xL=-1, xR=1, yD=-1, yU=1, PINN_collo = 1000, PINN_val = 200):

        self.xL, self.xR, self.yD,self.yU = xL,xR,yD,yU
        self.xlen = xR-xL
        self.ylen = yU-yD


        datas = loadmat(matfilePOD)
        # PODNum = number of snapshots
        self.Samples      = datas['Samples'][:,0:PODNum]

        self.FieldShape   = tuple(datas['FieldShape'][0])
        self.parameters   = datas['parameters'][0:PODNum,:]
        self.design_space = datas['design_space']
        self.NSample = self.Samples.shape[1]
        
        # data for validation
        datas = loadmat(matfileValidation)

        self.ValidationParameters   = datas['parameters']
        # self.ValidationSamples[i] = [p11,u11,v11,p12,u12,v12,...,p4747,u4747,v4747], solution on interior mesh for ith para
        self.ValidationSamples      = self.ExtractInteriorSnapshots( datas['Samples'] )
        # svd decomposition of p
        self.Modes_p, self.sigma_p, _ = np.linalg.svd( self.ExtractInteriorSnapshots(self.Samples)[0::3,:] );
        self.Modes_p = self.Modes_p[:,:M]
        self.M = M

        # svd decomposition of u
        self.Modes_u, self.sigma_u, _ = np.linalg.svd( self.ExtractInteriorSnapshots(self.Samples)[1::3,:] );
        self.Modes_u = self.Modes_u[:,:M]

        # svd decomposition of v
        self.Modes_v, self.sigma_v, _ = np.linalg.svd( self.ExtractInteriorSnapshots(self.Samples)[2::3,:] );
        self.Modes_v = self.Modes_v[:,:M]

        self.xChby = Chebyshev1D(xL=self.xL, xR=self.xR, M=self.FieldShape[0]-1)
        self.yChby = Chebyshev1D(xL=self.yD, xR=self.yU, M=self.FieldShape[1]-1)

        self.D1_x = self.xChby.DxCoeff(1)
        self.D1_y = self.yChby.DxCoeff(1)

        self.D1p_x = self.xChby.DxCoeffN2()
        self.D1p_y = self.yChby.DxCoeffN2()

        # projections
        self.projections_p = np.matmul( self.Modes_p.T, self.ExtractInteriorSnapshots(self.Samples)[0::3,:])
        self.projections_u = np.matmul( self.Modes_u.T, self.ExtractInteriorSnapshots(self.Samples)[1::3,:])
        self.projections_v = np.matmul( self.Modes_v.T, self.ExtractInteriorSnapshots(self.Samples)[2::3,:])

        self.Interior = np.zeros(self.FieldShape)
        self.Interior[1:-1,1:-1]=1
        self.Boundary=1-self.Interior

        self.uBC = np.reshape(self.Samples[1::NVARLOAD,0], self.FieldShape)*self.Boundary
        self.InteriorShape = (self.FieldShape[0]-2, self.FieldShape[1]-2,)

        # Compute projection error
        self.lamda_proj_p = np.matmul(self.ValidationSamples[0::3,:].T, self.Modes_p)

        self.lamda_proj_u = np.matmul(self.ValidationSamples[1::3,:].T, self.Modes_u)

        self.lamda_proj_v = np.matmul(self.ValidationSamples[2::3,:].T, self.Modes_v)

       
        self.Nh = 49*49
        self.Np_1d = 48
        self.N_1x = 49
        self.N_1y = 49
        self.on_K_dim = 12
        self.on_Y_dim = 3 
        self.Nh_list_bc1, self.Nh_list_bc21, self.Nh_list_bc22, self.Nh_list_bc23, self.Nh_list_in = get_inter_2d_inx(49,49)
        self.Nh_list_in_order = self.Nh_list_in
        num_elements = len(self.Nh_list_in_order)
        random_indices = list(range(num_elements))
        shuffle(random_indices)  # Shuffle the indices list

        # Use the random indices to create a shuffled version of `Nh_list_in_order`
        self.Nh_list_in = [self.Nh_list_in_order[i] for i in random_indices]
        self.Nh_list_1 = self.Nh_list_in[:PINN_collo]
        self.Nh_list_2 = self.Nh_list_in[PINN_collo:PINN_collo+PINN_val]
        # self.sol_snap [49*49*3,num of snapshot]
        self.sol_snap = self.ExtractAllSnapshots( self.Samples)
        self.off_obj = off_all(49,49)
        self.phi_3, self.onlinein = self.off_obj.RB_fun(self.sol_snap,PODNum,self.M)
        self.off_jac_ksi, self.off_hes_ksi = self.off_obj.compute_jaches_ksi(self.phi_3, self.M)

        self.label_p = self.Samples[0::3,:]
        self.label_p_int = self.label_p[self.Nh_list_in_order,:]
        self.label_p_norm_int = p_normalize_mat(self.label_p_int,self.N_1x,self.N_1y)
        self.label_u = self.Samples[1::3,:]
        self.label_v = self.Samples[2::3,:]

    
    def ExtractInteriorSnapshots(self,Samples):
        NSample = Samples.shape[1]
        Samples_shape = (self.FieldShape[0], self.FieldShape[1],NVAR,NSample,)
        return np.reshape( np.reshape(Samples, Samples_shape)[1:-1, 1:-1, 0:NVAR, :], (-1, NSample))

    def ExtractInteriorSnapshots_var(self,Samples):
        NSample = Samples.shape[1]
        Samples_shape = (self.FieldShape[0], self.FieldShape[1],1,NSample,)
        return np.reshape( np.reshape(Samples, Samples_shape)[1:-1, 1:-1, 0, :], (-1, NSample))
        
    def ExtractAllSnapshots(self,Samples):
        NSample = Samples.shape[1]
        Samples_shape = (self.FieldShape[0], self.FieldShape[1],NVARLOAD,NSample,)
        return np.reshape( np.reshape(Samples, Samples_shape)[:, :, 0:NVAR, :], (-1, NSample))

    def Compute_d_dxc(self, phi):
        return np.matmul(self.D1_x,phi)
    def Compute_d_dyc(self, phi):
        return np.matmul(self.D1_y, phi.T).T

    def Compute_d_dyc_label(self, phi):
        return np.transpose(np.matmul(self.D1_y, np.transpose(phi, (0, 2, 1))), (0, 2, 1))

    def Compute_dp_dyc_label(self, phi):
        return np.transpose(np.matmul(self.D1p_y, np.transpose(phi, (0, 2, 1))), (0, 2, 1))
        
    def Compute_dp_dxc(self, phi):
        return np.matmul(self.D1p_x,phi)
    def Compute_dp_dyc(self, phi):
        return np.matmul(self.D1p_y, phi.T).T
    def Compute_d_dxc2(self, phi):
        return self.Compute_d_dxc( self.Compute_d_dxc(phi) )
    def Compute_d_dyc2(self, phi):
        return self.Compute_d_dyc( self.Compute_d_dyc(phi) )
    def Compute_d_dxcyc(self, phi):
        return self.Compute_d_dyc( self.Compute_d_dxc(phi) )
    def Compute_d_dycxc(self, phi):
        return self.Compute_d_dxc( self.Compute_d_dyc(phi) )
    def Compute_dp_dxc2(self, phi):
        return self.Compute_dp_dxc( self.Compute_dp_dxc(phi) )
    def Compute_dp_dyc2(self, phi):
        return self.Compute_dp_dyc( self.Compute_dp_dyc(phi) )
    def Compute_dp_dxcyc(self, phi):
        return self.Compute_dp_dyc( self.Compute_dp_dxc(phi) )
    def Compute_dp_dycxc(self, phi):
        return self.Compute_dp_dxc( self.Compute_dp_dyc(phi) )


    def Compute_d_d1(self, phi):
        return self.Compute_d_dxc(phi), self.Compute_d_dyc(phi)
    def Compute_d_d1p(self, phi):
        return self.Compute_dp_dxc(phi), self.Compute_dp_dyc(phi)
    def Compute_d_d2(self, phi):
        return self.Compute_d_dxc2(phi), self.Compute_d_dyc2(phi), self.Compute_d_dxcyc(phi)





class CustomedNet(POD_Net):
    def __init__(self, num_parabc, rvalue, varaible, H1=3, layers1=None,oldnetfile1=None, reduce_net = None, roeqs=None,initial=False,
                             initW=torch.zeros(5,5), initb=torch.zeros(5), N=49):
        super(CustomedNet, self).__init__(layers1=layers1,OldNetfile1=oldnetfile1)
        self.M = roeqs.M

        self.lb   = torch.tensor(roeqs.design_space[0:1,:]).float().to(DEVICE)
        self.ub   = torch.tensor(roeqs.design_space[1:2,:]).float().to(DEVICE)

        self.roeqs = roeqs
        self.initial = initial
        self.initW = initW
        self.initb = initb
        self.H1 = H1
        self.rvalue = rvalue
        self.reduce_net = reduce_net
        self.num_parabc = num_parabc


        self.on_K_dim = 2
        self.on_Y_dim = 3 

        self.Np_1d = roeqs.Np_1d

        self.N = N

        self.off_obj = off_all(N,N)
        self.ksi1mesh_np = -np.cos(np.pi * np.arange(N) /(N-1))
        self.ksi1mesh = torch.tensor( self.ksi1mesh_np ).float().to(DEVICE)


        self.phi_3, offout = self.off_obj.RB_fun(self.roeqs.sol_snap,self.roeqs.NSample,self.M)
        #    [N_1x*N_1y,3*L]
        self.offout_all = torch.tensor( offout ).float().to(DEVICE)
        #    [N_1x*N_1y,3*L,2] [N_1x*N_1y,3*L,2,2]
        self.off_jac_ksi_all, self.off_hes_ksi_all = self.off_obj.compute_jaches_ksi(self.phi_3, self.M)
        self.lvalue = self.offout_all.shape[1]//3
        if self.lvalue != self.M:
            raise Exception('Expect l tobe %d'%self.M)

        if varaible == 'p':
            self.offout = self.offout_all[:,:self.lvalue]
            self.off_jac_ksi = self.off_jac_ksi_all[:,:self.lvalue,:]
            self.off_hes_ksi = self.off_hes_ksi_all[:,:self.lvalue,:,:]
        elif varaible == 'u':
            self.offout = self.offout_all[:,self.lvalue:2*self.lvalue]
            self.off_jac_ksi = self.off_jac_ksi_all[:,self.lvalue:2*self.lvalue,:]
            self.off_hes_ksi = self.off_hes_ksi_all[:,self.lvalue:2*self.lvalue,:,:]
        elif varaible == 'v':
            self.offout = self.offout_all[:,2*self.lvalue:]
            self.off_jac_ksi = self.off_jac_ksi_all[:,2*self.lvalue:,:]
            self.off_hes_ksi = self.off_hes_ksi_all[:,2*self.lvalue:,:,:]
        self.compute_deri_torch = torch_off_fun_eval_deri()
        self.varaible = varaible

    def ubc1(self,xgrid, para_bcall, n_terms):
      
        result = np.zeros_like(xgrid, dtype=float)
        for n in range(1, n_terms + 1):
            result += para_bcall[n - 1] * np.sin(n * np.pi * xgrid) * np.cos(n * np.pi * xgrid)
        ubc_para_vec = (1+xgrid+0.001*result)**2 * (1-xgrid+0.001*result)**2
        return ubc_para_vec


    def u_net(self,x):
        x = (x-(self.ub+self.lb)/2)/(self.ub-self.lb)*2
        out = self.unet(x)
        return out

    def forward(self,x):
        return self.u_net(x).detach().cpu().numpy()


    def To_phir(self,offout):
        # [xnum,L]>[xnum,r]
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

        ulb = allout[:,0,0]

        uub = allout[:,-1,0]

        return ulb,uub

    def compute_deri_2_jaches2(self, online_in, dnn_fun, W1, b1, W2, b2):

        '''

         dnn_fun a neural network function map online_in to uhat

         return the output of online nn dnn_fun. also return the jacobian and hessian 

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


    def compute_deri_2(self, uhat_jac_mat,uhat_hes_mat, allout, batch_jacobian, batch_hess, kind):

        '''
         batch_jacobian shape [Nh,L], the partial derivative of reduced basis function w.r.t. input x
         First dim is number of points, second dim is each RB functions

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

    def p_normalize_torch(self, pall):

        tensors = []
        for i in range(pall.shape[1]):

            p = pall[:,i]

            p_int = clenshaw_curtis_quadrature_2d_torch(p.reshape(1,self.roeqs.N_1x,self.roeqs.N_1y), self.roeqs.N_1x, self.roeqs.N_1y, discard_endpoints=False)
            p_norm = p - p_int/4

            p_int2 = clenshaw_curtis_quadrature_2d_torch(p_norm.reshape(1,self.roeqs.N_1x,self.roeqs.N_1y), self.roeqs.N_1x, self.roeqs.N_1y, discard_endpoints=False)
            p_norm2 = p_norm - p_int2/4


            p_int3 = clenshaw_curtis_quadrature_2d_torch(p_norm2.reshape(1,self.roeqs.N_1x,self.roeqs.N_1y), self.roeqs.N_1x, self.roeqs.N_1y, discard_endpoints=False)
            p_norm3 = p_norm2 - p_int3/4

            tensors.append(p_norm3)

        result_stack = torch.stack(tensors, dim=1)


        return result_stack


    def Phi_x(self,phi_jac_mat, phi_hes_mat,batch_jacobian, batch_hess, kind):

        # batch_jacobian[Nresi,x_num,3*L,2]
        # batch_hess[Nresi,x_num,3*L,2,2]
        # phi_jac_mat [Num_x,r,3L] , phi_hes_mat[Num_x,r,3L,3L]


        # prime_x [Nresi,1,x_num,3*L]
        prime_x = torch.unsqueeze(batch_jacobian[:,:,:,0], 1)
        # prime_y [Nresi,1,x_num,3*L]
        prime_y = torch.unsqueeze(batch_jacobian[:,:,:,1], 1)
        # prime_yy [Nresi,1,x_num,3*L]
        prime_yy = torch.unsqueeze(batch_hess[:,:,:,1,1], 1)
        # prime_xx [Nresi,1,x_num,3*L]
        prime_xx = torch.unsqueeze(batch_hess[:,:,:,0,0], 1)

        if kind == 'phi_x':
          # return [Nresi,r,x_num]
          phi_x = torch.sum(torch.transpose(phi_jac_mat, 0, 1)*prime_x,3)

          return phi_x

        if kind == 'phi_y':

          phi_y = torch.sum(torch.transpose(phi_jac_mat, 0, 1)*prime_y,3)

          return phi_y

        if kind == 'phi_xx':

            Phi_xx_vmap =vmap(vmap(self.vmap_2nd_deri, in_dims=(None, None, 1, 1), randomness='same'), in_dims=(0, 0, None, None), randomness='same')
   
            #   Nresi[Num_x,3L],  Nresi[Num_x,3L],   r*[Num_x,3L] , r*[3L,Num_x,3L] >>> [Nresi,r,x_num]
            Phir_xx = Phi_xx_vmap(batch_jacobian[:,:,:,0], batch_hess[:,:,:,0,0], phi_jac_mat, torch.transpose(phi_hes_mat, 0, 2))

            return Phir_xx

        if kind == 'phi_yy':


            Phi_yy_vmap =vmap(vmap(self.vmap_2nd_deri, in_dims=(None, None, 1, 1), randomness='same'), in_dims=(0, 0, None, None), randomness='same')
       
            Phir_yy = Phi_yy_vmap(batch_jacobian[:,:,:,1], batch_hess[:,:,:,1,1], phi_jac_mat, torch.transpose(phi_hes_mat, 0, 2))

            return Phir_yy


    
    def online_deri(self, online_in, W1, b1, W2, b2, var):

        l1 = torch.matmul(online_in,W1) + b1

        uhat_jac = torch.matmul((torch.transpose(W2, 1, 2)*(1-torch.tanh(l1)**2)),torch.transpose(W1, 1, 2))

        if var == 'u' or var == 'v':
            uhat_hes = torch.matmul(torch.matmul(torch.matmul(torch.matmul(W1,torch.diag_embed(W2[:,:,0])), torch.diag_embed(-2*torch.tanh(torch.transpose(l1, 0, 1)))) , torch.diag_embed(1-torch.tanh(torch.transpose(l1, 0, 1))**2)),torch.transpose(W1, 1, 2))

            return uhat_jac, torch.transpose(torch.transpose(uhat_hes, 0, 2), 0, 1)

        else:

            return uhat_jac

    def online_deri_first_deri(self, online_in, W1, b1, W2, b2):

        l1 = torch.matmul(online_in,W1) + b1

        uhat_jac = torch.matmul((torch.transpose(W2, 1, 2)*(1-torch.tanh(l1)**2)),torch.transpose(W1, 1, 2))

        return uhat_jac

   

    def vmap_2nd_deri(self,offd1,offd2,ond1,ond2):

        B = torch.sum(ond2*offd1,2)

        hat_d_d = torch.sum(torch.transpose(B, 0, 1) * offd1,1) + torch.sum(ond1*offd2,1)

        return hat_d_d


    def compute_off_all_online(self,x):

        geo_para = x[:,self.num_parabc+1].detach().cpu().numpy()

        theta = 2*np.pi*geo_para/360

        off_jac, off_hes = self.off_obj.compute_jaches(theta, self.M, self.off_jac_ksi, self.off_hes_ksi)

        off_tr_jac = off_jac

        off_tr_hes = off_hes

        off_tr_jac_te = torch.tensor( off_tr_jac ).float().to(DEVICE)

        off_tr_hes_te = torch.tensor( off_tr_hes ).float().to(DEVICE)

        return off_tr_jac_te, off_tr_hes_te

    def compute_off_all_label(self,x):

        geo_para = x[:,self.num_parabc+1].detach().cpu().numpy()

        theta = 2*np.pi*geo_para/360

        off_jac, off_hes = self.off_obj.compute_jaches(theta, self.M, self.off_jac_ksi, self.off_hes_ksi)

        off_tr_jac = off_jac[:,self.roeqs.Nh_list_in_order,:,:]

        off_tr_hes = off_hes[:,self.roeqs.Nh_list_in_order,:,:,:]

        off_tr_jac_te = torch.tensor( off_tr_jac ).float().to(DEVICE)

        off_tr_hes_te = torch.tensor( off_tr_hes ).float().to(DEVICE)

        return off_tr_jac_te, off_tr_hes_te


    def compute_off_all(self,x):

        geo_para = x[:,self.num_parabc+1].detach().cpu().numpy()

        theta = 2*np.pi*geo_para/360

        off_jac, off_hes = self.off_obj.compute_jaches(theta, self.M, self.off_jac_ksi, self.off_hes_ksi)

        off_tr_jac = off_jac[:,self.roeqs.Nh_list_1,:,:]

        off_tr_hes = off_hes[:,self.roeqs.Nh_list_1,:,:,:]

        off_tr_jac_te = torch.tensor( off_tr_jac ).float().to(DEVICE)

        off_tr_hes_te = torch.tensor( off_tr_hes ).float().to(DEVICE)

        return off_tr_jac_te, off_tr_hes_te


    def compute_deri_first_deri(self, var_jac_mat, phi_x, phi_y):

        phir_x = torch.transpose(phi_x, 1, 2)

        phir_y = torch.transpose(phi_y, 1, 2)       

        p_x = torch.sum(var_jac_mat*phir_x,2)

        p_y = torch.sum(var_jac_mat*phir_y,2)

        return p_x, p_y

    def loss_label_grad(self, labeled_inputs, off_tr_jac, off_tr_hes, label_varaible_x, label_varaible_y):

        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(labeled_inputs))

        Phi = self.To_phir(self.offout)

        Phi_prime_jac,Phi_prime_hes = self.Phi_deri(self.offout[self.roeqs.Nh_list_in_order,:])

        phi_x = self.Phi_x(Phi_prime_jac, Phi_prime_hes, off_tr_jac, off_tr_hes, kind='phi_x')
        phi_y = self.Phi_x(Phi_prime_jac, Phi_prime_hes, off_tr_jac, off_tr_hes, kind='phi_y')

        allout = self.on_fun(Phi,W1_mu, b1_mu,W2_mu, b2_mu)

        varaible_jac_mat = self.online_deri_first_deri(Phi[self.roeqs.Nh_list_in_order,:], W1_mu, b1_mu, W2_mu, b2_mu)

        varaible_x, varaible_y = self.compute_deri_first_deri(varaible_jac_mat, phi_x, phi_y)

        loss = torch.mean((varaible_x - label_varaible_x)**2) + torch.mean((varaible_y - label_varaible_y)**2)

        return loss

    def loss_label(self, x, var_star):

        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(x))

        Phi = self.To_phir(self.offout)

        allout = self.on_fun(Phi,W1_mu, b1_mu,W2_mu, b2_mu)

        # on_out[Nresi,Num_x,1]
        # p_org[Nresi,Num_x]
        if self.varaible == 'p':
            p_org = allout[:,:,0]
            p_org = p_org[:,self.roeqs.Nh_list_in_order]

            pdiff = (p_org - var_star)**2

            pdiff_norm = clenshaw_curtis_quadrature_2d_torch((pdiff.reshape(pdiff.shape[0],self.roeqs.N_1x-2,self.roeqs.N_1y-2)), self.roeqs.N_1x, self.roeqs.N_1y, discard_endpoints=True)
            loss = torch.mean(pdiff_norm)
        elif self.varaible == 'u':
            udiff = (allout[:,:,0] - var_star)**2
            udiff_norm = clenshaw_curtis_quadrature_2d_torch((udiff.reshape(udiff.shape[0],self.roeqs.N_1x,self.roeqs.N_1y)), self.roeqs.N_1x, self.roeqs.N_1y, discard_endpoints=False)
            loss = torch.mean(udiff_norm)
        elif self.varaible == 'v':
            vdiff = (allout[:,:,0] - var_star)**2
            vdiff_norm = clenshaw_curtis_quadrature_2d_torch((vdiff.reshape(vdiff.shape[0],self.roeqs.N_1x,self.roeqs.N_1y)), self.roeqs.N_1x, self.roeqs.N_1y, discard_endpoints=False)
            loss = torch.mean(vdiff_norm)        
        
        return loss

    def error_compute(self, Val_inputs, p_exact_norm, u_exact, v_exact, relative_exact_p, relative_exact_u, relative_exact_v):

        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(Val_inputs))

        Phi = self.To_phir(self.offout)
        allout = self.on_fun(Phi, W1_mu, b1_mu, W2_mu, b2_mu)

        if self.varaible == 'p':
            p_org = self.roeqs.ExtractInteriorSnapshots_var( allout[:,:,0].detach().cpu().numpy().T )
            p_org_norm = self.p_normalize_np(p_org)
            pdiff = ((p_org_norm).T - p_exact_norm.T)**2
            
            pdiff_norm = clenshaw_curtis_quadrature_2d_discrete((pdiff.reshape(pdiff.shape[0],self.roeqs.N_1x-2,self.roeqs.N_1y-2)), self.roeqs.N_1x, self.roeqs.N_1y, discard_endpoints=True)
            return np.sqrt(pdiff_norm)

        elif self.varaible == 'u':
            u_int = self.roeqs.ExtractInteriorSnapshots_var( allout[:,:,0].detach().cpu().numpy().T )
            udiff = (u_int.T  - u_exact.T)**2
            udiff_norm = clenshaw_curtis_quadrature_2d_discrete((udiff.reshape(udiff.shape[0],self.roeqs.N_1x-2,self.roeqs.N_1y-2)), self.roeqs.N_1x, self.roeqs.N_1y, discard_endpoints=True)
            return np.sqrt(udiff_norm)

        elif self.varaible == 'v':
            v_int = self.roeqs.ExtractInteriorSnapshots_var( allout[:,:,0].detach().cpu().numpy().T )
            vdiff = (v_int.T - v_exact.T)**2
            vdiff_norm = clenshaw_curtis_quadrature_2d_discrete((vdiff.reshape(vdiff.shape[0],self.roeqs.N_1x-2,self.roeqs.N_1y-2)), self.roeqs.N_1x, self.roeqs.N_1y, discard_endpoints=True)
            return np.sqrt(vdiff_norm)




    def error_compute_norm2(self, Val_inputs, p_exact_norm, u_exact, v_exact, relative_exact, relative_exact_p, relative_exact_u, relative_exact_v):

        W1_mu,b1_mu,W2_mu,b2_mu = self.muNN_out_to_weight(self.u_net(Val_inputs))

        Phi = self.To_phir(self.offout)
        on_out = self.on_fun(Phi, W1_mu, b1_mu, W2_mu, b2_mu)

        p_org = on_out[:,self.roeqs.Nh_list_in_order,0]

        p_org_norm = self.p_normalize_torch(p_org.T)

        pdiff = (p_org_norm - p_exact_norm)**2
        udiff = (on_out[:,self.roeqs.Nh_list_in_order,1].T - u_exact)**2
        vdiff = (on_out[:,self.roeqs.Nh_list_in_order,2].T - v_exact)**2
                                            
        Errortotal = torch.mean(torch.sqrt(torch.sum(pdiff,0) + torch.sum(udiff,0) + torch.sum(vdiff,0))/relative_exact)

        Errorpuv = [torch.mean(torch.sqrt(torch.sum(pdiff,0))/relative_exact_p), torch.mean(torch.sqrt(torch.sum(udiff,0))/relative_exact_u), torch.mean(torch.sqrt(torch.sum(vdiff,0))/relative_exact_v)]
        return Errorpuv, Errortotal


    def GetError_nonlinear(self):
        Nvalidation =self.ValidationParameters.shape[0]
        allweight = self.u_net(torch.tensor(self.ValidationParameters).float().to(DEVICE))
        W1,b1,W2,b2 = self.muNN_out_to_weight(allweight)

        Phi = self.To_phir(self.offout)

        on_out = self.on_fun(Phi, W1)

        puv_out = on_out[:,roeqs.Nh_list_in_order,:]

        phi_Num          = self.ValidationSamples.T
        Error = np.zeros((Nvalidation,NVAR))   
        for nvar in range(NVAR):
            Error[:,nvar] = np.linalg.norm(phi_Num[:,nvar::NVAR]-puv_out[:,:,nvar], axis = 1)\
                           /np.linalg.norm(phi_Num[:,nvar::NVAR], axis=1)
        Errorpuv = Error.mean(axis=0)
        phi_pred = puv_out.reshape(Nvalidation,-1)
        Errortotal =  np.linalg.norm(phi_Num[:,:]-phi_pred[:,:], axis = 1)\
                                   /np.linalg.norm(phi_Num[:,:], axis=1)

        Errortotal = Errortotal.mean(axis=0)

        return Errorpuv, Errortotal



    def p_normalize_np(self, pall):

        tensors = []
        for i in range(pall.shape[1]):

            p = pall[:,i]
            p_int = clenshaw_curtis_quadrature_2d_discrete((p.reshape(1,self.N-2,self.N-2)), self.N, self.N, discard_endpoints=True)

            p_norm = p - p_int/4

            p_int2 = clenshaw_curtis_quadrature_2d_discrete(p_norm.reshape(1,self.N-2,self.N-2), self.N, self.N, discard_endpoints=True)

            p_norm2 = p_norm - p_int2/4

            p_int3 = clenshaw_curtis_quadrature_2d_discrete(p_norm2.reshape(1,self.N-2,self.N-2), self.N, self.N, discard_endpoints=True)

            p_norm3 = p_norm2 - p_int3/4

            tensors.append(p_norm3)

        result_stack = np.stack(tensors, axis=1)


        return result_stack



