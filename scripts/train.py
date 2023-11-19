# conda activate myigwn-py39
import numpy as np
import bilby 
import pycbc 
import sys
import matplotlib.pyplot as plt

import zuko
from glasflow import RealNVP, CouplingNSF
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import pickle
from sklearn.decomposition import IncrementalPCA
import sklearn.decomposition 

import river.data
from river.data.datagenerator import DataGeneratorBilbyFD
from river.data.dataset import DatasetStrainFD
#import river.data.utils as datautils
from river.data.utils import *

from river.models import embedding
from river.models.utils import *
from river.models.embedding.pca import project_strain_data_FDAPhi
from river.models.embedding.conv import EmbeddingConv1D, EmbeddingConv2D

import logging
import sys
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)


#logger.addHandler(stdout_handler)

source_type = 'BNS'
detector_names = ['H1', 'L1', 'V1'] 
duration = 32
f_low = 20
f_ref = 20
sampling_frequency = 2048
waveform_approximant = 'IMRPhenomPv2_NRTidal'
parameter_names = PARAMETER_NAMES_PRECESSINGBNS_BILBY
PSD_type = 'bilby_default' #'zero_noise'
use_sealgw_detector = True


Nsample = 2000
Nvalid = 200
injection_parameters_valid = generate_BNS_injection_parameters(Nvalid,
        a_max=0.1,
        d_min=10,
        d_max=100,
        d_power=3,
        tc_min=-0.1,
        tc_max=0.1)

injection_parameters_train = generate_BNS_injection_parameters(Nsample,
        a_max=0.1,
        d_min=10,
        d_max=100,
        d_power=3,
        tc_min=-0.1,
        tc_max=0.1)

data_generator_train = DataGeneratorBilbyFD(source_type,
            detector_names, 
            duration, 
            f_low, 
            f_ref, 
            sampling_frequency, 
            waveform_approximant, 
            parameter_names,
            PSD_type=PSD_type,
            use_sealgw_detector=use_sealgw_detector)

data_generator_valid = DataGeneratorBilbyFD(source_type,
            detector_names, 
            duration, 
            f_low, 
            f_ref, 
            sampling_frequency, 
            waveform_approximant, 
            parameter_names,
            PSD_type=PSD_type,
            use_sealgw_detector=use_sealgw_detector)


data_generator_train.inject_signals(injection_parameters_train, Nsample)
data_generator_train.numpy_starins()

data_generator_valid.inject_signals(injection_parameters_valid, Nvalid)
data_generator_valid.numpy_starins()

dataset_train = DatasetStrainFD(data_dict=data_generator_train.data, parameter_names=PARAMETER_NAMES_PRECESSINGBNS_BILBY)
dataset_valid = DatasetStrainFD(data_dict=data_generator_valid.data, parameter_names=PARAMETER_NAMES_PRECESSINGBNS_BILBY)

batch_size_train = 256
batch_size_valid = 64
train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
valid_loader = DataLoader(dataset_valid, batch_size=batch_size_valid, shuffle=True)



ipca_gen = load_model('ipca_models/IPCA_BNSFD_10000to500_ExpUnwrap_fixtc.pickle')

n_components = ipca_gen.n_components

downsample_rate = 4
n_freq = dataset_train[0:2][1][:,:,::downsample_rate].shape[-1]
device='cuda'

embedding_proj = EmbeddingConv1D(ndet=3, ncomp=n_components, nout=128, middle_channel=512).to(device)
embedding_noproj = EmbeddingConv2D(ndet=3, ncomp=n_freq, nout=128, middle_channel=16).to(device)
#flow = zuko.flows.NSF(features=17, context=256, transforms=100, hidden_features=(640, 640)).to(device)
#flow = zuko.flows.CNF(features=17, context=256, hidden_features=(640, 640)).to(device)
flow = CouplingNSF(n_inputs=17,n_transforms=100, n_conditional_inputs=256, n_neurons=128, batch_norm_between_transforms=True,).to(device)

train_func = train_glasflow #train_zukoflow
eval_func = eval_glasflow #eval_zukoflow
lr = 1e-3
gamma = 0.5
optimizer = torch.optim.Adam(list(embedding_proj.parameters()) + list(embedding_noproj.parameters()) + list(flow.parameters()), lr=lr)
#sche_step_size = 20
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sche_step_size, gamma=gamma)

ckpt_dir = 'trained_models/glasnsf_fixtc_twoconv_res'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)
    print(f"Made dir {ckpt_dir}")

logfilename = f"{ckpt_dir}/logs.log"
file_handler = logging.FileHandler(logfilename)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
ckpt_path = f'{ckpt_dir}/checkpoint.pickle'

max_epoch = 2000000
epoches_update = 20
epoches_pretrain = 40
epoches_save_loss = 20
epoches_adjust_lr = 20

load_from_previous_train = 0
if load_from_previous_train:
    checkpoint = torch.load(ckpt_path)
    
    best_epoch = checkpoint['epoch']
    start_epoch = best_epoch + 1
    lr_updated_epoch = start_epoch
    embedding_proj.load_state_dict(checkpoint['embd_proj_state_dict'])
    embedding_noproj.load_state_dict(checkpoint['embd_noproj_state_dict'])
    flow.load_state_dict(checkpoint['flow_state_dict']) 

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    

    train_losses = checkpoint['train_losses']
    valid_losses = checkpoint['valid_losses']

    logger.info(f'Loaded states from {ckpt_path}, epoch={start_epoch}.')
else:
    train_losses = []
    valid_losses = []
    start_epoch = 0
    lr_updated_epoch = start_epoch


#for g in optimizer.param_groups:
#    g['lr'] = 5e-4

for epoch in range(start_epoch, max_epoch):    
    if epoch % epoches_update == 0 and epoch>=epoches_pretrain:
        data_generator_train.initialize_data()
        injection_parameters_train = generate_BNS_injection_parameters(Nsample, a_max=0.1, d_min=10, d_max=100, d_power=3, tc_min=-0.1, tc_max=0.1)
        data_generator_train.inject_signals(injection_parameters_train, Nsample)
        data_generator_train.numpy_starins()
        dataset_train = DatasetStrainFD(data_dict=data_generator_train.data, parameter_names=PARAMETER_NAMES_PRECESSINGBNS_BILBY)
        train_loader = DataLoader(dataset_train, batch_size_train, shuffle=True)
        logger.info(f"Training data updated at epoch={epoch}")

    train_loss, train_loss_std = train_func(flow, embedding_proj, embedding_noproj, optimizer, train_loader, detector_names, ipca_gen, device=device, downsample_rate=downsample_rate)
    valid_loss, valid_loss_std = eval_func(flow,  embedding_proj, embedding_noproj, valid_loader, detector_names, ipca_gen, device=device, downsample_rate=downsample_rate)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    logger.info(f'epoch {epoch}, train loss = {train_loss}±{train_loss_std}, valid loss = {valid_loss}±{valid_loss_std}')

    if valid_loss==min(valid_losses):
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'embd_proj_state_dict': embedding_proj.state_dict(),
            'embd_noproj_state_dict': embedding_noproj.state_dict(),
            'flow_state_dict': flow.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), #'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'valid_losses': valid_losses
            }, ckpt_path)

        logger.info(f'Current best epoch: {best_epoch}. Checkpoint saved.')

    if epoch%epoches_save_loss == 0 and epoch!=0:
        save_loss_data(train_losses, valid_losses, ckpt_dir)
    
    if epoch-best_epoch>epoches_adjust_lr and epoch-lr_updated_epoch>5:
        adjust_lr(optimizer, gamma)
        logger.info(f'Validation loss has not dropped for {epoches_adjust_lr} epoches. Learning rate decreased by a factor of {gamma}.')
    #scheduler.step()
