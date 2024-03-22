# library
import os
import shutil
import numpy as np
import time
# scripts
from tqdm import tqdm
from utils.util import *

######## Iterative Methods #######

def Nesterov(alg, xinit=None,
            numIter=100, step=100, accelerate=False, useNoise=True, create_graph=False, 
            strict=False, is_save=True, save_path='result', xtrue=None,  save_iter=100, clip=False, tau=1):
    """
    Nesterov Acc. for ODER/RED (DEQ)
    ### INPUT:
    alg        ~ ODER/RED (DEQ) per-interate block
    numIter    ~ total number of iterations
    accelerate ~ Acc. or not. Default: True
    useNoise   ~ CNN predict noise or image
    step       ~ step-size
    is_save    ~ if true save the reconstruction of each iteration
    save_path  ~ the save path for is_save
    xtrue      ~ the ground truth of the image, for tracking purpose

    ### OUTPUT:
    x     ~ reconstruction of the algorithm
    outs  ~ detailed information including cost, snr, step-size and time of each iteration

    """
    ##### HELPER FUNCTION #####
    f = lambda z : alg(z, step, tau, create_graph, strict, useNoise, clip)
    ##### INITIALIZATION #####
    
    # initialize save foler
    if is_save:
        save_path = save_path + '/tau_%.3f'%(tau)
        abs_save_path = os.path.abspath(save_path)
        if os.path.exists(save_path):
            print("Removing '{:}'".format(abs_save_path))
            shutil.rmtree(abs_save_path, ignore_errors=True)
        # make new path
        print("Allocating '{:}'".format(abs_save_path))
        os.makedirs(abs_save_path)
    #initialize info data
    if xtrue is not None:
        xtrueSet = True
        snr_iters = []
    else:
        xtrueSet = False
    # initialize variables
    if xinit == 'zeros':
        xinit = torch.zeros(alg.dObj.size(), dtype=xinit.dtype).cuda()
    else:    
        xinit = alg.dObj.get_init()

    x = xinit
    s = x  # gradient update
    t = torch.tensor(1., dtype=torch.float32)  # controls acceleration

    ##### MAIN LOOP #####
    for indIter in tqdm(range(numIter)):
        xnext = f(s)
        if xtrueSet:
            snr_iters.append(compare_snr(x.squeeze(), xtrue.squeeze()).item())

        # acceleration
        if accelerate:
            tnext = 0.5*(1+torch.sqrt(1+4*t*t))
        else:
            tnext = 1
        s = xnext + ((t-1)/tnext)*(xnext-x)
        
        # update
        t = tnext
        x = xnext

        # save & print
        if is_save and (indIter+1) % (save_iter/2) == 0:
            img_save = np.clip(x.squeeze().cpu().data.numpy(), 0, 1).astype(np.float64)
            if len(img_save.shape)==3 and img_save.shape[0]==3:
                img_save = img_save.transpose([1,2,0])
            save_img(img_save, abs_save_path+'/iter_%d_%.2fdB.tif'%(indIter+1, snr_iters[indIter]))

    # summarize outs
    x_rec = np.clip(x.squeeze().cpu().data.numpy(), 0, 1).astype(np.float64)

    outs = {
        "snr_iters": np.array(snr_iters),
        "recon": x_rec,
    }

    if is_save:
        save_mat(outs, abs_save_path+'/out.mat'.format(indIter+1))

    return x_rec, outs