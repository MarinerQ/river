# conda activate myigwn-py39
# export OMP_NUM_THREADS=24
import numpy as np
import bilby 
#import pycbc 
import sys
import matplotlib.pyplot as plt

import zuko
from glasflow import RealNVP, CouplingNSF
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import river.data
from river.data.datagenerator import DataGeneratorBilbyFD
from river.data.dataset import DatasetStrainFD,DatasetStrainFDFromPreCalSVDWF
#import river.data.utils as datautils
from river.data.utils import *

from river.models import embedding
from river.models.utils import *
from river.models.embedding.conv import EmbeddingConv1D, EmbeddingConv2D
from river.models.embedding.mlp import EmbeddingMLP1D
from river.models.inference.cnf import GlasNSFConv1DRes

import logging
import sys
import os
import json
from copy import deepcopy
def main():
    config_path = sys.argv[1]
    with open(f"{config_path}/config.json", 'r') as f:
        config = json.load(f)

    config_datagenerator = config['data_generator_parameters']
    config_training = config['training_parameters']
    config_model = config['model_parameters']
    config_precalwf = config['precalwf_parameters']
    
    

    # Set up logger
    PID = os.getpid()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    ckpt_dir = config['ckpt_dir']
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
        logger.warning(f"{ckpt_dir} does not exist. Made dir {ckpt_dir}.")

    logfilename = f"{ckpt_dir}/logs.log"
    file_handler = logging.FileHandler(logfilename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    ckpt_path = f'{ckpt_dir}/checkpoint.pickle'

    logger.info(f'PID={PID}.')
    logger.info(f'Output path: {ckpt_dir}')

    detector_names = config_datagenerator['detector_names']

    #Nsample = config_training['Nsample']
    #Nvalid = config_training['Nvalid']
    Nsample = config_precalwf['train']['nbatch'] * config_precalwf['train']['file_per_batch'] * config_precalwf['train']['sample_per_file'] 
    Nvalid = config_precalwf['valid']['nbatch'] * config_precalwf['valid']['file_per_batch'] * config_precalwf['valid']['sample_per_file'] 
    Ntest = config_training['Ntest']

    #selection_factor = 2
    selection_factor = config_datagenerator['selection_factor'] # make sure there are enough SNR>8 samples in injection_parameters
    injection_parameters_test = generate_BNS_injection_parameters(Nsample = Ntest*selection_factor, **config_datagenerator)

    data_generator_train = DataGeneratorBilbyFD(**config_datagenerator)
    data_generator_valid = DataGeneratorBilbyFD(**config_datagenerator)
    config_datagenerator_test = config_datagenerator.copy()
    config_datagenerator_test['ipca'] = None 
    data_generator_test = DataGeneratorBilbyFD(**config_datagenerator_test)

    logger.info(f'Generating test data.')
    data_generator_test.inject_signals(injection_parameters_test, Ninj=Ntest*selection_factor, Nneeded =Ntest)
    data_generator_test.numpy_starins()


    logger.info(f'Loading precalculated waveforms.')
    train_precalwf_list = get_precalwf_list(**config_precalwf['train'])
    valid_precalwf_list = get_precalwf_list(**config_precalwf['valid'])
    
    dataset_train = DatasetStrainFDFromPreCalSVDWF(train_precalwf_list, config_datagenerator['context_parameter_names'], data_generator_train,
                                                    dmin=config_datagenerator['d_min'], dmax=config_datagenerator['d_max'], dpower=config_datagenerator['d_power'])
    dataset_valid = DatasetStrainFDFromPreCalSVDWF(valid_precalwf_list, config_datagenerator['context_parameter_names'], data_generator_valid,
                                                    dmin=config_datagenerator['d_min'], dmax=config_datagenerator['d_max'], dpower=config_datagenerator['d_power'])
    dataset_test = DatasetStrainFD(data_dict=data_generator_test.data, parameter_names=config_datagenerator['context_parameter_names'])


    batch_size_train = config_training['batch_size_train']
    batch_size_valid = config_training['batch_size_valid']
    batch_size_test = config_training['batch_size_test']
    logger.info(f'Nsample: {Nsample}, Nvalid: {Nvalid}')
    logger.info(f'batch_size_train: {batch_size_train}, batch_size_valid: {batch_size_valid}')

    train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size_valid, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False)

    ipca_path = config['model_parameters']['ipca_path']
    ipca_gen = load_model(ipca_path)
    logger.info(f'IPCA loaded from {ipca_path}')

    n_components = ipca_gen.n_components

    #downsample_rate = config_embd_noproj['downsample_rate']
    #logger.info(f'Downsample rate: {downsample_rate}')
    #n_freq = dataset_train[0:2][1][:,:,::downsample_rate].shape[-1]
    device=config_training['device']
    model = GlasNSFConv1DRes(config).to(device)

    lr = config_training['lr']
    gamma = config_training['gamma']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info(f'Initial learning rate: {lr}')
    logger.info(f'Gamma: {gamma}')

    max_epoch = config_training['max_epoch']
    epoches_pretrain = config_training['epoches_pretrain']
    epoches_save_loss = config_training['epoches_save_loss']
    epoches_adjust_lr = config_training['epoches_adjust_lr']
    epoches_adjust_lr_again = config_training['epoches_adjust_lr_again']
    #load_from_previous_train = 1
    load_from_previous_train = config_training['load_from_previous_train']
    if load_from_previous_train:
        checkpoint = torch.load(ckpt_path)
        
        best_epoch = checkpoint['epoch']
        start_epoch = best_epoch + 1
        lr_updated_epoch = start_epoch
        model.load_state_dict(checkpoint['model_state_dict']) 

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        test_losses = checkpoint['test_losses']

        logger.info(f'Loaded states from {ckpt_path}, epoch={start_epoch}.')
    else:
        best_epoch = 0
        train_losses = []
        valid_losses = []
        test_losses = []
        start_epoch = 0
        lr_updated_epoch = start_epoch

    npara_flow = count_parameters(model.flow)
    npara_embd_proj = count_parameters(model.embedding)
    npara_embd_res = count_parameters(model.resnet)
    npara_total = count_parameters(model)
    logger.info(f'Learnable parameters: flow: {npara_flow}, embedding_PCA: {npara_embd_proj}, ResNet: {npara_embd_res}. Total: {npara_total}. ')

    ###
    #for g in optimizer.param_groups:
    #    g['lr'] = 1e-5
    #    logger.info(f'Set lr to 1e-5.')

    logger.info(f'Training started.')

    for epoch in range(start_epoch, max_epoch):    

        train_loss, train_loss_std = train_GlasNSFWarpper(model, optimizer, train_loader, detector_names, ipca_gen, device=device)
        valid_loss, valid_loss_std = eval_GlasNSFWarpper(model, valid_loader, detector_names, ipca_gen, device=device)
        test_loss, test_loss_std = eval_GlasNSFWarpper(model, test_loader, detector_names, ipca_gen, device=device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)
        logger.info(f'epoch {epoch}, train loss = {train_loss}±{train_loss_std}, valid loss = {valid_loss}±{valid_loss_std}, test loss = {test_loss}±{test_loss_std}')

        if valid_loss==min(valid_losses):
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'test_losses': test_losses,
                }, ckpt_path)

            logger.info(f'Current best epoch: {best_epoch}. Checkpoint saved.')

        if epoch%epoches_save_loss == 0 and epoch!=0:
            save_loss_data(train_losses, valid_losses, ckpt_dir, test_losses=test_losses)
        
        if epoch-best_epoch>=epoches_adjust_lr and epoch-lr_updated_epoch>=epoches_adjust_lr_again:
            adjust_lr(optimizer, gamma)
            logger.info(f'Validation loss has not dropped for {epoch-best_epoch} epoches. Learning rate is decreased by a factor of {gamma}.')
            lr_updated_epoch = epoch

            '''
            checkpoint = torch.load(ckpt_path)
            embedding_proj.load_state_dict(checkpoint['embd_proj_state_dict'])
            embedding_noproj.load_state_dict(checkpoint['embd_noproj_state_dict'])
            flow.load_state_dict(checkpoint['flow_state_dict']) 
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ntime = (epoch-lr_updated_epoch) // epoches_adjust_lr_again
            adjust_lr(optimizer, gamma**ntime)
            logger.info(f'Validation loss has not dropped for {epoch-best_epoch} epoches. Learning rate is decreased by a factor of {gamma}.')
            logger.info(f'Loaded model states from {ckpt_path}, and best epoch {best_epoch}. Going from there with a smaller lr.')
            lr_updated_epoch = epoch
            '''

if __name__ == "__main__":
    main()

