import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import hydra

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from torch.utils.data import DataLoader

import numpy as np
from datetime import datetime

from iterAlgs import *
from utils.util import *
from regularizers.robjects import *
from DataFidelities.CTClass import *
from utils.data_preprocess_CT import data_preprocess

now = datetime.now()
@hydra.main(config_name='configs/config')
def main(config):
    model_type = config['model_type']
    config = config[model_type]
    fwd_set = config['fwd_test']

    if config['root_path'] == 'None':
        root = os.path.dirname(os.path.abspath(__file__))
        config['code_path'] = root
        root = root.split('/')[0:-1]
        config['root_path'] = os.path.join('/',*root)
    # set the random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ####################################################
    ####              DATA PREPARATION               ###
    ####################################################
    
    _, _, test_set = data_preprocess(config, device, set_mode='test')

    test_dataset = TestDataset(test_set['test_ipt'], 
                    test_set['test_gdt'], test_set['test_y'])
    test_dataLoader = DataLoader(test_dataset, 
                batch_size=config['testing'].batch_size)
    taus_ODER=sio.loadmat('./data/taus_ODER.mat',squeeze_me=True)['tau_all']    
    taus_REDDEQ=sio.loadmat('./data/taus_REDDEQ.mat',squeeze_me=True)['tau_all']       
    ####################################################
    ####                LOOP IMAGES                  ###
    ####################################################
    print('Start Running!')
    RED_model = config['RED_model']
    num_img = 0
    for batch_ipt, batch_gdt, batch_y  in test_dataLoader:
        batch_ipt = batch_ipt.to(device)
        batch_gdt = batch_gdt.to(device)
        batch_y = batch_y.to(device)
        if config['RED_model'].solver == "Nesterov" :

            ####################################################
            ####                     ODER                    ###
            ####################################################
            print('Start Running ODER !')
            config['inference'].load_path = './models/model_zoo/ODER.pth'
            dObj = CTClass(batch_y, batch_ipt, test_set['emParamsTest'], batch_size=fwd_set['batchAngles'], numAngles=fwd_set['numAngles'],device=device)
            rObj = UnetClass(config, config['inference'].load_path, device)
            ODER = ODER_block(dObj, rObj, config['RED_model'])
            save_results = './results/ODER/img_%d'%(num_img)

            tau, gamma_inti, numIter = RED_model['tau_inti'], RED_model['gamma_inti'], RED_model['num_iter']
            tau=taus_ODER[num_img]
            #-- Reconstruction --# 
            recon_oder, out_each_oder = Nesterov(ODER, numIter=numIter, step=gamma_inti, accelerate=True, useNoise=True, 
                            is_save=True, save_path=save_results, xtrue=batch_gdt, xinit='FBP', save_iter=180, clip=True, tau=tau)
            # save & print
            save_img(recon_oder, './results/ODER_IMG_%d_%.2fdB.tif'%(num_img, out_each_oder['snr_iters'][-1]))
            ####################################################
            ####                  RED(DEQ)                   ###
            ####################################################
            print('Start Running RED (DEQ) !')
            config['inference'].load_path = './models/model_zoo/RED(DEQ).pth'
            dObj = CTClass(batch_y, batch_ipt, test_set['emParamsTest'], batch_size=fwd_set['batchAngles'], numAngles=fwd_set['numAngles'],device=device)
            rObj = UnetClass(config, config['inference'].load_path, device)
            REDDEQ = REDDEQ_block(dObj, rObj, config['RED_model'])
            save_results = './results/RED(DEQ)/img_%d'%(num_img)
            tau, gamma_inti, numIter = RED_model['tau_inti'], RED_model['gamma_inti'], RED_model['num_iter']
            tau=taus_REDDEQ[num_img]
            #-- Reconstruction --# 
            recon_reddeq, out_each_reddeq = Nesterov(REDDEQ, numIter=numIter, step=gamma_inti, accelerate=True, useNoise=True, 
                            is_save=True, save_path=save_results, xtrue=batch_gdt, xinit='FBP', save_iter=180, clip=True, tau=tau)
            # save & print
            save_img(recon_reddeq, './results/RED(DEQ)_IMG_%d_%.2fdB.tif'%(num_img, out_each_reddeq['snr_iters'][-1]))
        else:
            raise Exception("Unrecognized mode.")
        num_img += 1

if __name__ == "__main__":
    main()