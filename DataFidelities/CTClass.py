'''
Class for quadratic-norm on 2D Sparse-view CT
CT forward model is intensively adopted from:
https://github.com/phernst/pytorch_radon
'''
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.util import *
from DataFidelities.pytorch_radon.filters import HannFilter
from DataFidelities.pytorch_radon import *

class TrainDataset(Dataset):

    def __init__(self, train_ipt:torch.Tensor, 
                       train_gdt:torch.Tensor, 
                       train_y:torch.Tensor,
                ):
          
        super(TrainDataset, self).__init__()
        self.train_ipt = train_ipt
        self.train_gdt = train_gdt
        self.train_y = train_y

    def __len__(self):
        return self.train_gdt.shape[0]

    def __getitem__(self, item):
        return self.train_ipt[item], self.train_gdt[item], self.train_y[item]

class ValidDataset(Dataset):

    def __init__(self, valid_ipt:torch.Tensor, 
                       valid_gdt:torch.Tensor, 
                       valid_y:torch.Tensor,
                       ):
          
        super(ValidDataset, self).__init__()
        self.valid_ipt = valid_ipt
        self.valid_gdt = valid_gdt
        self.valid_y = valid_y
    def __len__(self):
        return self.valid_gdt.shape[0]

    def __getitem__(self, item):
        return self.valid_ipt[item], self.valid_gdt[item], self.valid_y[item]

class TestDataset(Dataset):

    def __init__(self, test_ipt:torch.Tensor, 
                       test_gdt:torch.Tensor,
                       test_y:torch.Tensor,
                       ):
          
        super(Dataset, self).__init__()
        self.test_ipt = test_ipt
        self.test_gdt = test_gdt
        self.test_y = test_y
    def __len__(self):
        return self.test_gdt.shape[0]

    def __getitem__(self, item):
        return self.test_ipt[item], self.test_gdt[item], self.test_y[item]

###################################################
###              Tomography Class               ###
###################################################

class CTClass(nn.Module):

    def __init__(self, y, ipt, emParams, sigSize=512, batch_size=30, numAngles=90, device='cpu'):
        super(CTClass, self).__init__()              
        self.y = y
        self.ipt = ipt

        self.sigSize = sigSize
        self.numAngles = numAngles
        self.batch_size = batch_size
        # generate angle array
        self.r = Radon_grad(self.sigSize)
        self.ir = IRadon_grad(self.sigSize)

        self.r_all_grids = emParams['r_all_grids'].to(device)
        self.ir_xy_grids = emParams['ir_xy_grids'].to(device)
        self.theta = emParams['theta'].to(device)
 
    def get_init(self):
        return self.ipt

    def size(self):
        sigSize = self.sigSize
        return sigSize

    def grad(self, x):
        with torch.no_grad():
            g = self.ftran(self.fmult(x, self.r_all_grids) - self.y, self.ir_xy_grids, self.theta)
        return g

    def sgrad(self, x, meas_list=None):
        with torch.no_grad():
            if meas_list is None:
                meas_list = self.get_subset(self.numAngles, self.batch_size)
            r_all_grids_stoc = self.r_all_grids[meas_list,...]
            theta_stoc = self.theta[meas_list]
            g = self.ftran(self.fmult(x, r_all_grids_stoc) - self.y[:,:,:,meas_list], self.ir_xy_grids, theta_stoc)
        return g

    def fmult(self, x, all_grid=None):
        sino = self.r(x, all_grid)
        return sino
    
    def ftran(self, z, ir_xy_grids, theta):
        all_grid = self.ir._create_grids(theta, ir_xy_grids[0],ir_xy_grids[1])
        reco_torch = self.ir(z, all_grid, theta)
        return reco_torch

    @staticmethod
    def get_subset(numAngles,  batch_size, num_div=5):

        sub =  torch.randperm(numAngles//num_div)[0:batch_size//num_div]
        sub, _ = torch.sort(sub)
        meas_list = torch.cat([sub + i*numAngles//num_div for i in range(num_div)])#.tolist()
        return meas_list

    @staticmethod
    def tomoCT(ipt, sigSize, all_grids, theta, inputSNR=40):

        device = ipt.device
        # generate angle array
        ipt = ipt.unsqueeze(0).unsqueeze(0).to(device)
        
        # forward project
        r = Radon_grad(sigSize)
        ir = IRadon_grad(sigSize)
        ir_hann = IRadon_grad(sigSize, use_filter=HannFilter())
    
        sino = r(ipt, all_grids[0])

        # add white noise to the sinogram
        sino = addwgn_torch(sino, inputSNR)[0]

        # backward project
        recon_bp = ir(sino, all_grids[1], theta)

        # filtered backward project
        reco_fbp_hann = ir_hann(sino, all_grids[1], theta)
        reco_fbp_hann[reco_fbp_hann<=0] = 0
        
        return sino, recon_bp, reco_fbp_hann

    @staticmethod
    def gen_grid(sigSize, device=None, numAngles=180, angleSigma=0):
        theta = np.linspace(0., 180, numAngles, endpoint=False)
        if angleSigma!=0:
            angle_noise = np.random.normal(0, angleSigma, numAngles)
        else:
            angle_noise = 0    
        theta = theta + angle_noise 
        theta = torch.tensor(theta, dtype=torch.float, device=device)
        # forward project
        r = Radon(sigSize, theta, False, device=device)
        # backward project
        ir = IRadon(sigSize, theta, False, use_filter=None, device=device)

        r_ir_all_grids = torch.stack((r.all_grids, ir.all_grids),axis=0)
        ir_xy_grids = torch.stack((ir.xgrid, ir.ygrid),axis=0)


        ir1 = IRadon_grad(sigSize)
        all_grid = ir1._create_grids(theta, ir_xy_grids[0],ir_xy_grids[1])

        res = (all_grid - ir.all_grids).sum()

        return r_ir_all_grids, r.all_grids, ir_xy_grids, theta






