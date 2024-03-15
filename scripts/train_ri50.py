# conda activate myigwn-py39
# export OMP_NUM_THREADS=4


import numpy as np
import bilby 
#import pycbc 
import sys
import matplotlib.pyplot as plt
import glob 

#import zuko
from glasflow import RealNVP, CouplingNSF
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import river.data
from river.data.datagenerator import DataGeneratorBilbyFD
from river.data.dataset import DatasetSVDStrainFDFromSVDWFonGPU, DatasetSVDStrainFDFromSVDWFonGPUBatch, DatasetConvStrainFDFromSVDWFonGPU, DatasetConvStrainFDFromSVDWFonGPUBatch
#import river.data.utils as datautils
from river.data.utils import *

from river.models import embedding
from river.models.utils import *
from river.models.embedding.conv import EmbeddingConv1D, EmbeddingConv2D
from river.models.embedding.mlp import EmbeddingMLP1D
from river.models.inference.cnf import GlasNSFConv1DRes, GlasNSFConv1D, GlasNSFTest, GlasflowEmbdding

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
    config_precaldata = config['precaldata_parameters']
    
    

    # Set up logger
    PID = os.getpid()
    device=config_training['device']
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
    #Nsample_available = config_precaldata['train']['nbatch'] * config_precaldata['train']['file_per_batch'] * config_precaldata['train']['sample_per_file'] 
    #Nvalid_available = config_precaldata['valid']['nbatch'] * config_precaldata['valid']['file_per_batch'] * config_precaldata['valid']['sample_per_file'] 

    #selection_factor = 2
    #selection_factor = config_datagenerator['selection_factor'] # make sure there are enough SNR>8 samples in injection_parameters



    logger.info(f'Loading precalculated data.')
    train_filenames = glob.glob(f"{config_precaldata['train']['folder']}/batch*/*.h5")#[:8]
    valid_filenames = glob.glob(f"{config_precaldata['valid']['folder']}/batch*/*.h5")
    #logger.info(f'{len(train_precaldata_filelist)}, {len(valid_precaldata_filelist)}')
    
    data_generator = DataGeneratorBilbyFD(**config_datagenerator)
    
    Vhfile = config_model['Vhfile']
    Nbasis = config_model['Nbasis']
    batch_size_train = config_training['batch_size_train']
    minibatch_size_train = config_training['minibatch_size_train']
    batch_size_valid = config_training['batch_size_valid']
    minibatch_size_valid = config_training['minibatch_size_valid']

    
    dataset_train = DatasetConvStrainFDFromSVDWFonGPUBatch(train_filenames, PARAMETER_NAMES_CONTEXT_PRECESSINGBNS_BILBY, data_generator,
                                     Nbasis=Nbasis, Vhfile=Vhfile, device=device, minibatch_size=minibatch_size_train, fix_extrinsic=False,
                                     shuffle=False, add_noise=True, reparameterize=True)
    dataset_valid = DatasetConvStrainFDFromSVDWFonGPUBatch(valid_filenames, PARAMETER_NAMES_CONTEXT_PRECESSINGBNS_BILBY, data_generator,
                                     Nbasis=Nbasis, Vhfile=Vhfile, device=device, minibatch_size=minibatch_size_valid, fix_extrinsic=False,
                                     shuffle=False, add_noise=True, reparameterize=True)
    '''
    
    dataset_train = DatasetSVDStrainFDFromSVDWFonGPUBatch(train_filenames, PARAMETER_NAMES_CONTEXT_PRECESSINGBNS_BILBY, data_generator,
                                     Nbasis=Nbasis, Vhfile=Vhfile, device=device, minibatch_size=minibatch_size_train, fix_extrinsic=False,
                                     shuffle=False, add_noise=True, reparameterize=True)
    dataset_valid = DatasetSVDStrainFDFromSVDWFonGPUBatch(valid_filenames, PARAMETER_NAMES_CONTEXT_PRECESSINGBNS_BILBY, data_generator,
                                     Nbasis=Nbasis, Vhfile=Vhfile, device=device, minibatch_size=minibatch_size_valid, fix_extrinsic=False,
                                     shuffle=False, add_noise=True, reparameterize=True)
    '''

    Nsample = len(dataset_train)*minibatch_size_train
    Nvalid = len(dataset_valid)*minibatch_size_valid
    logger.info(f'Nsample: {Nsample}, Nvalid: {Nvalid}.')
    logger.info(f'batch_size_train: {batch_size_train}, batch_size_valid: {batch_size_valid}')

    train_loader = DataLoader(dataset_train, batch_size=batch_size_train // minibatch_size_train, shuffle=False)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size_valid // minibatch_size_valid, shuffle=False)


    #model = GlasNSFConv1DRes(config).to(device)
    #model = GlasNSFConv1D(config).to(device)
    model = GlasflowEmbdding(config).to(device)


    lr = config_training['lr']
    gamma = config_training['gamma']
    weight_decay = config_training['weight_decay']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info(f'Initial learning rate: {lr}')
    logger.info(f'Gamma: {gamma}')

    max_epoch = config_training['max_epoch']
    #epoches_pretrain = config_training['epoches_pretrain']
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


        logger.info(f'Loaded states from {ckpt_path}, epoch={start_epoch}.')
    else:
        best_epoch = 0
        train_losses = []
        valid_losses = []

        start_epoch = 0
        lr_updated_epoch = start_epoch

    npara_flow = count_parameters(model.flow)
    #npara_embd_proj = count_parameters(model.embedding)
    #npara_embd_res = count_parameters(model.resnet)
    npara_total = count_parameters(model)
    #logger.info(f'Learnable parameters: flow: {npara_flow}, embedding_PCA: {npara_embd_proj}, ResNet: {npara_embd_res}. Total: {npara_total}. ')
    #logger.info(f'Learnable parameters: flow: {npara_flow}, embedding_PCA: {npara_embd_proj}. Total: {npara_total}. ')
    logger.info(f'Learnable parameters: flow: {npara_flow}, total: {npara_total}. ')

    ###
    #for g in optimizer.param_groups:
    #    g['lr'] = 1e-5
    #    logger.info(f'Set lr to 1e-5.')

    logger.info(f'Training started, device:{device}. ')

    for epoch in range(start_epoch, max_epoch):    
        
        train_loss, train_loss_std = train_GlasNSFWarpper(model, optimizer, train_loader, device=device, minibatch_size=minibatch_size_train)
        valid_loss, valid_loss_std = eval_GlasNSFWarpper(model, valid_loader, device=device, minibatch_size=minibatch_size_valid)


        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        logger.info(f'epoch {epoch}, train loss = {train_loss}±{train_loss_std}, valid loss = {valid_loss}±{valid_loss_std}')

        if valid_loss==min(valid_losses):
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                }, ckpt_path)

            logger.info(f'Current best epoch: {best_epoch}. Checkpoint saved.')

        if epoch%epoches_save_loss == 0 and epoch!=0:
            save_loss_data(train_losses, valid_losses, ckpt_dir)
        
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
        #dataset_train.shuffle_indexinfile()
        dataset_train.shuffle_wflist()
        train_loader = DataLoader(dataset_train, batch_size=batch_size_train // minibatch_size_train, shuffle=False)
if __name__ == "__main__":
    main()

