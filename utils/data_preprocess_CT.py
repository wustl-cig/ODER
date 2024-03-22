from skimage.util import view_as_windows

import torch
import numpy as np
from tqdm import tqdm

from utils.util import *
from DataFidelities.CTClass import CTClass
np.random.seed(128)

################Functions######################
FFT  = lambda x: torch.fft(x,  signal_ndim=2)
iFFT = lambda x: torch.ifft(x, signal_ndim=2)
rFFT  = lambda x: torch.rfft(x,  signal_ndim=2, onesided=False)
###############################################
def compare_snr(img_test, img_true):
    return 20 * torch.log10(torch.norm(img_true.flatten()) / torch.norm(img_true.flatten() - img_test.flatten()))

def gen_x_y(config, img_raw, device, set_mode:str):

    imgList = []
    num_scales = [1]
    dtype = torch.FloatTensor
    IMG_PATCH = config['IMG_Patch']
    for index in range(img_raw.shape[0]):

        for scaling in num_scales:
            
            for m in range(1):
                img = scaling * img_raw[index].copy()
                # img = data_augmentation(img, m)
                img = np.ascontiguousarray(img)
                img = view_as_windows(img, window_shape=[IMG_PATCH, IMG_PATCH], step=40)
                img = np.ascontiguousarray(img)
                img.shape = [-1] + [IMG_PATCH, IMG_PATCH]
                img = np.rot90(img, k=1, axes=(1,2))
                img = np.expand_dims(img, 1)  # Add channel in the dimension right after batch-dimension.
                imgList.append(img)
    img = torch.from_numpy(np.concatenate(imgList, 0)).type(dtype)
    img_fbp = torch.zeros_like(img)

    r_ir_all_grids, r_all_grids, ir_xy_grids, theta = CTClass.gen_grid(IMG_PATCH, device, config['numAngles'], angleSigma=0)

    sino_temp, _, _ = CTClass.tomoCT(torch.squeeze(img[0]).to(device),
                                    IMG_PATCH, r_ir_all_grids, theta, inputSNR=config['inputSNR'])

    y_sino = torch.zeros([img.shape[0], sino_temp.shape[1], sino_temp.shape[2],sino_temp.shape[3]], dtype=torch.float32)

    for index in tqdm(range(img.shape[0]), 'CT %s Dataset'%(set_mode)):
        img_temp = img[index]
        img_temp = (img_temp - torch.min(img_temp)) /(torch.max(img_temp) - torch.min(img_temp))
        img[index] =  img_temp

        y_sino[index], _, img_fbp[index] = CTClass.tomoCT(torch.squeeze(img[index]).to(device), 
                                         IMG_PATCH, r_ir_all_grids, theta, inputSNR=config['inputSNR'])

    gdt = img.type(dtype)  # (72, 1, 90, 90)
    y = y_sino.type(dtype)
    ipt = img_fbp.type(dtype)

    print("%s Dataset: "%(set_mode))
    print('gdt: ', gdt.shape, gdt.dtype)
    print('ipt: ', ipt.shape, ipt.dtype)
    print('y: ', y.shape, y.dtype)

    print('Avg_SNR of FBP on %s dataset:  '%(set_mode), compare_snr(ipt, gdt))
    
    _, r_all_grids, ir_xy_grids, theta = CTClass.gen_grid(IMG_PATCH, device, config['numAngles'], angleSigma=0)

    return gdt, y, ipt, r_all_grids, ir_xy_grids, theta

def data_preprocess(config:dict, device=None, set_mode='test'):
    train_set, valid_set, test_set = {},{},{}
    ###############################################
    #                 Start Loading               #
    ###############################################
    if set_mode=='test':
        img_raw = sio.loadmat(config['test_datapath'])['img']
        img_raw = img_raw.astype(np.float32)
        fwd_test = config['fwd_test']
        gdt, y, ipt, r_all_grids, ir_xy_grids, theta = gen_x_y(fwd_test, img_raw, device, set_mode)

        emParamsTest = {
            "r_all_grids":r_all_grids.detach().cpu(),
            "ir_xy_grids":ir_xy_grids.detach().cpu(),
            "theta": theta.detach().cpu()
        }

        test_set = {
                    "test_ipt": ipt,
                    "test_gdt": gdt,
                    "test_y": y,
                    "emParamsTest":emParamsTest
                    }
    else:
        raise Exception("Unrecognized mode.")

    return train_set, valid_set, test_set