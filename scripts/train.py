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
from river.data.dataset import DatasetStrainFD
#import river.data.utils as datautils
from river.data.utils import *

from river.models import embedding
from river.models.utils import *
from river.models.embedding.conv import EmbeddingConv1D, EmbeddingConv2D
from river.models.embedding.mlp import EmbeddingMLP1D

import logging
import sys
import os
import json

def main():
    config_path = sys.argv[1]
    with open(f"{config_path}/config.json", 'r') as f:
        config = json.load(f)

    config_datagenerator = config['data_generator_parameters']
    config_training = config['training_parameters']
    config_flow = config['model_parameters']['flow']
    config_embd_proj = config['model_parameters']['embedding_proj']
    config_embd_resnet = config['model_parameters']['embedding_resnet']
    config_embd_noproj = config['model_parameters']['embedding_noproj']

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

    Nsample = config_training['Nsample']
    Nvalid = config_training['Nvalid']

    #selection_factor = 2
    selection_factor = config_datagenerator['selection_factor'] # make sure there are enough SNR>8 samples in injection_parameters
    injection_parameters_valid = generate_BNS_injection_parameters(Nsample = Nvalid*selection_factor, **config_datagenerator)
    injection_parameters_train = generate_BNS_injection_parameters(Nsample = Nsample*selection_factor, **config_datagenerator)

    data_generator_train = DataGeneratorBilbyFD(**config_datagenerator)
    data_generator_valid = DataGeneratorBilbyFD(**config_datagenerator)

    logger.info(f'Generating initial training data.')
    data_generator_train.inject_signals(injection_parameters_train, Ninj=Nsample*selection_factor, Nneeded = Nsample)
    data_generator_train.numpy_starins()

    data_generator_valid.inject_signals(injection_parameters_valid, Ninj=Nvalid*selection_factor, Nneeded =Nvalid)
    data_generator_valid.numpy_starins()

    dataset_train = DatasetStrainFD(data_dict=data_generator_train.data, parameter_names=config_datagenerator['context_parameter_names'])
    dataset_valid = DatasetStrainFD(data_dict=data_generator_valid.data, parameter_names=config_datagenerator['context_parameter_names'])

    data_generator_train.initialize_data()
    data_generator_valid.initialize_data()

    batch_size_train = config_training['batch_size_train']
    batch_size_valid = config_training['batch_size_valid']
    logger.info(f'Nsample: {Nsample}, Nvalid: {Nvalid}')
    logger.info(f'batch_size_train: {batch_size_train}, batch_size_valid: {batch_size_valid}')

    train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size_valid, shuffle=True)

    ipca_path = config['model_parameters']['ipca_path']
    ipca_gen = load_model(ipca_path)
    logger.info(f'IPCA loaded from {ipca_path}')

    n_components = ipca_gen.n_components

    downsample_rate = config_embd_noproj['downsample_rate']
    logger.info(f'Downsample rate: {downsample_rate}')
    n_freq = dataset_train[0:2][1][:,:,::downsample_rate].shape[-1]
    device=config_training['device']

    embedding_proj = get_model(config_embd_proj).to(device)
    #embedding_noproj = get_model(config_embd_noproj).to(device)
    embedding_resnet = get_model(config_embd_resnet).to(device)
    flow = get_model(config_flow).to(device)
    #train_func, eval_func = get_train_func(flow)
    #train_func = train_glasflow_v2
    #eval_func = eval_glasflow_v2
    train_func = train_glasflow_v3
    eval_func = eval_glasflow_v3

    lr = config_training['lr']
    gamma = config_training['gamma']
    ###optimizer = torch.optim.Adam(list(embedding_proj.parameters()) + list(embedding_noproj.parameters()) + list(flow.parameters()), lr=lr)
    optimizer = torch.optim.Adam(list(embedding_proj.parameters()) + list(flow.parameters()) + list(embedding_resnet.parameters()), lr=lr)

    logger.info(f'Initial learning rate: {lr}')
    logger.info(f'Gamma: {gamma}')

    max_epoch = config_training['max_epoch']
    epoches_update = config_training['epoches_update']
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
        embedding_proj.load_state_dict(checkpoint['embd_proj_state_dict'])
        #embedding_noproj.load_state_dict(checkpoint['embd_noproj_state_dict'])
        embedding_resnet.load_state_dict(checkpoint['embd_resnet_state_dict'])
        flow.load_state_dict(checkpoint['flow_state_dict']) 

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

    npara_flow = count_parameters(flow)
    npara_embd_proj = count_parameters(embedding_proj)
    npara_embd_res = count_parameters(embedding_resnet)
    #npara_embd_noproj = count_parameters(embedding_noproj)
    #logger.info(f'Learnable parameters: flow: {npara_flow}, embedding_PCA: {npara_embd_proj}, embedding_strain: {npara_embd_noproj}. Total: {npara_embd_noproj+npara_embd_proj+npara_flow}. ')
    #logger.info(f'Learnable parameters: flow: {npara_flow}, embedding_PCA: {npara_embd_proj}. Total: {npara_embd_proj+npara_flow}. ')
    logger.info(f'Learnable parameters: flow: {npara_flow}, embedding_PCA: {npara_embd_proj}, ResNet: {npara_embd_res}. Total: {npara_embd_proj+npara_flow+npara_embd_res}. ')

    ###
    #for g in optimizer.param_groups:
    #    g['lr'] = 1e-5
    #    logger.info(f'Set lr to 1e-5.')

    logger.info(f'Training started.')

    for epoch in range(start_epoch, max_epoch):    
        if epoch % epoches_update == 0 and epoch>=epoches_pretrain:
            data_generator_train.initialize_data()
            injection_parameters_train = generate_BNS_injection_parameters(Nsample=Nsample*selection_factor, **config_datagenerator)
            data_generator_train.inject_signals(injection_parameters_train, Ninj=Nsample*selection_factor, Nneeded = Nsample)
            data_generator_train.numpy_starins()
            dataset_train = DatasetStrainFD(data_dict=data_generator_train.data, parameter_names=config_datagenerator['context_parameter_names'])
            train_loader = DataLoader(dataset_train, batch_size_train, shuffle=True)
            logger.info(f"Training data updated at epoch={epoch}")
            data_generator_train.initialize_data()

        #train_loss, train_loss_std = train_func(flow, embedding_proj, embedding_noproj, optimizer, train_loader, detector_names, ipca_gen, device=device, downsample_rate=downsample_rate)
        #valid_loss, valid_loss_std = eval_func(flow,  embedding_proj, embedding_noproj, valid_loader, detector_names, ipca_gen, device=device, downsample_rate=downsample_rate)
        
        # v2
        #train_loss, train_loss_std = train_func(flow, embedding_proj, optimizer, train_loader, detector_names, ipca_gen, device=device, downsample_rate=downsample_rate)
        #valid_loss, valid_loss_std = eval_func(flow,  embedding_proj, valid_loader, detector_names, ipca_gen, device=device, downsample_rate=downsample_rate)

        # v3
        train_loss, train_loss_std = train_func(flow, embedding_proj, embedding_resnet, optimizer, train_loader, detector_names, ipca_gen, device=device, downsample_rate=downsample_rate)
        valid_loss, valid_loss_std = eval_func(flow,  embedding_proj, embedding_resnet, valid_loader, detector_names, ipca_gen, device=device, downsample_rate=downsample_rate)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        logger.info(f'epoch {epoch}, train loss = {train_loss}±{train_loss_std}, valid loss = {valid_loss}±{valid_loss_std}')

        if valid_loss==min(valid_losses):
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'embd_proj_state_dict': embedding_proj.state_dict(),
                'embd_resnet_state_dict': embedding_resnet.state_dict(),
                #'embd_noproj_state_dict': embedding_noproj.state_dict(),
                'flow_state_dict': flow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'valid_losses': valid_losses
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

if __name__ == "__main__":
    main()

