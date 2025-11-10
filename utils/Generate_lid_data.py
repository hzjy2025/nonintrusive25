import torch
import random
import numpy as np
import scipy.io




# construct a class to build all data
class tr_data():
    def __init__(self,X_dim,Y_dim,geo_para):

        self.X_dim = X_dim
        self.Y_dim = Y_dim
        self.theta1 = geo_para



    def bc_data(self,bc1,bc21,bc22,bc23):

        #x,y bc1

        x_lbc1 = -1.0
        x_ubc1 = 1.0
        y_lbc1 = 1.0
        y_ubc1 = 1.0
        bc1_ind = np.random.rand(bc1)
        xbc1 = x_lbc1 + (x_ubc1-x_lbc1)*bc1_ind
        ybc1 = y_lbc1 + (y_ubc1-y_lbc1)*bc1_ind
        Xbc1 = np.concatenate((xbc1.reshape(-1,1),ybc1.reshape(-1,1)), axis=1)

        xbcp1 = xbc1.reshape(-1)
        u_bc1 = (1+xbcp1)**2*(1-xbcp1)**2

#x,y bc21

        x_lbc21 = 1.0
        x_ubc21 = 1.0
        y_lbc21 = -1.0
        y_ubc21 = 1.0
        bc21_ind = np.random.rand(bc21)
        xbc21 = x_lbc21 + (x_ubc21-x_lbc21)*bc21_ind
        ybc21 = y_lbc21 + (y_ubc21-y_lbc21)*bc21_ind
        Xbc21 = np.concatenate((xbc21.reshape(-1,1),ybc21.reshape(-1,1)), axis=1)
#x,y bc22

        x_lbc22 = -1.0
        x_ubc22 = 1.0
        y_lbc22 = -1.0
        y_ubc22 = -1.0
        bc22_ind = np.random.rand(bc22)
        xbc22 = x_lbc22 + (x_ubc22-x_lbc22)*bc22_ind
        ybc22 = y_lbc22 + (y_ubc22-y_lbc22)*bc22_ind
        Xbc22 = np.concatenate((xbc22.reshape(-1,1),ybc22.reshape(-1,1)), axis=1)
#x,y bc23

        x_lbc23 = -1.0
        x_ubc23 = -1.0
        y_lbc23 = -1.0
        y_ubc23 = 1.0
        bc23_ind = np.random.rand(bc23)
        xbc23 = x_lbc23 + (x_ubc23-x_lbc23)*bc23_ind
        ybc23 = y_lbc23 + (y_ubc23-y_lbc23)*bc23_ind
        Xbc23 = np.concatenate((xbc23.reshape(-1,1),ybc23.reshape(-1,1)), axis=1)
        return Xbc1,u_bc1,Xbc21,Xbc22,Xbc23

    def tr_off(self,xnum_tr_off,ynum_tr_off):

        x_l_tr = -1.0
        x_u_tr = 1.0
        y_l_tr = -1.0
        y_u_tr = 1.0

        xtr_ind = np.random.rand(xnum_tr_off)
        ytr_ind = np.random.rand(ynum_tr_off)

        x_tr = x_l_tr + (x_u_tr-x_l_tr)*xtr_ind
        y_tr = y_l_tr + (y_u_tr-y_l_tr)*ytr_ind

        X_tr_off = np.concatenate((x_tr.reshape(-1,1),y_tr.reshape(-1,1)), axis=1)


        return X_tr_off

    def K_to_X(self,K):


        Tran_mat = np.array([[0.5,0],
                             [0.5*np.cos(self.theta1),0.5*np.sin(self.theta1)]])

        X = K@Tran_mat

        return X

    def tr_on(self,xnum_tr_on,ynum_tr_on):

        x_l_tr = -1.0
        x_u_tr = 1.0
        y_l_tr = -1.0
        y_u_tr = 1.0

        xtr_ind = np.random.rand(xnum_tr_on)
        ytr_ind = np.random.rand(ynum_tr_on)

        x_tr = x_l_tr + (x_u_tr-x_l_tr)*xtr_ind
        y_tr = y_l_tr + (y_u_tr-y_l_tr)*ytr_ind

        X_tr_on = np.concatenate((x_tr.reshape(-1,1),y_tr.reshape(-1,1)), axis=1)

        return X_tr_on

    def val_off(self,xnum_val,ynum_val):

        x_l_tr = -1.0
        x_u_tr = 1.0
        y_l_tr = -1.0
        y_u_tr = 1.0

        xtr_ind = np.random.rand(xnum_val)
        ytr_ind = np.random.rand(ynum_val)

        x_tr = x_l_tr + (x_u_tr-x_l_tr)*xtr_ind
        y_tr = y_l_tr + (y_u_tr-y_l_tr)*ytr_ind

        X_val = np.concatenate((x_tr.reshape(-1,1),y_tr.reshape(-1,1)), axis=1)

        return X_val

    def test_data(self,xnum_test,ynum_test):

        xl = -1
        xu = 1
        xgrid = (xl+xu)/2 - np.cos(np.arange(xnum_test+1)*np.pi/xnum_test)*(xu-xl)/2
        yl = -1
        yu = 1
        ygrid = (yl+yu)/2 - np.cos(np.arange(ynum_test+1)*np.pi/ynum_test)*(yu-yl)/2

        y1,x1 = np.meshgrid(ygrid,xgrid)

        Xtest_ch = np.concatenate((x1.reshape(-1,1),y1.reshape(-1,1)), axis=1)


        return Xtest_ch

    def test_data_uni(self,xnum_test,ynum_test):

        xgrid = np.linspace(-1, 1, xnum_test)
        ygrid = np.linspace(-1, 1, ynum_test)

        y1,x1 = np.meshgrid(ygrid,xgrid)

        Xtest_ch = np.concatenate((x1.reshape(-1,1),y1.reshape(-1,1)), axis=1)


        return Xtest_ch

    def test_data_uni_mesh(self,xnum_test,ynum_test):

        xgrid = np.linspace(-1, 1, xnum_test)
        ygrid = np.linspace(-1, 1, ynum_test)

        y1,x1 = np.meshgrid(ygrid,xgrid)

        return x1,y1


    def test_data_mesh(self,xnum_test,ynum_test):

        xl = -1
        xu = 1
        xgrid = (xl+xu)/2 - np.cos(np.arange(xnum_test+1)*np.pi/xnum_test)*(xu-xl)/2
        yl = -1
        yu = 1
        ygrid = (yl+yu)/2 - np.cos(np.arange(ynum_test+1)*np.pi/ynum_test)*(yu-yl)/2

        y1,x1 = np.meshgrid(ygrid,xgrid)

        return x1,y1



    def test_data_mesh_origin(self,xnum_test,ynum_test):


      xl = -1
      xu = 1
      xgrid = (xl+xu)/2 - np.cos(np.arange(xnum_test+1)*np.pi/xnum_test)*(xu-xl)/2
      yl = -1
      yu = 1
      ygrid = (yl+yu)/2 - np.cos(np.arange(ynum_test+1)*np.pi/ynum_test)*(yu-yl)/2

      y1,x1 = np.meshgrid(ygrid,xgrid)

      # change to real domain
      x = 0.5*x1+0.5*y1*np.cos(self.theta1)
      y = 0.5*y1*np.sin(self.theta1)


      return x,y

def choose_para(bound, para_all):

    re_list = []

    for i in range(para_all.shape[0]):

      if (para_all[i,0] < bound[0]) and (para_all[i,1] < bound[1]):
        re_list.append(i)

    return re_list

def get_inter_2d_inx(Nx,Ny):

    Nh_list = [i for i in range(Nx*Ny)]

    # Boundary

    Nh_list_bc1 = Nh_list[Ny-1::Ny]  # top
    Nh_list_bc21 = Nh_list[(Nx-1)*Ny:]  # right
    Nh_list_bc22 = Nh_list[0::Ny]  # bottom
    Nh_list_bc23 = Nh_list[0:Ny]   # left


    # Inter

    Nh_list_in = [x for x in Nh_list if x not in Nh_list_bc1]
    Nh_list_in = [x for x in Nh_list_in if x not in Nh_list_bc21]
    Nh_list_in = [x for x in Nh_list_in if x not in Nh_list_bc22]
    Nh_list_in = [x for x in Nh_list_in if x not in Nh_list_bc23]

    return Nh_list_bc1, Nh_list_bc21, Nh_list_bc22, Nh_list_bc23, Nh_list_in



####
# U [N,1]
def get_inter_2d_value(Nh_list_in,U):

  return U[Nh_list_in]

