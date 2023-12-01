#conda activate myigwn-py39
#export OMP_NUM_THREADS=10
import numpy as np
import bilby 
import pycbc 
import sys
import matplotlib.pyplot as plt

import zuko
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
from river.models import utils as modelutils


Nsample_template = 500
Nround = 20 # actual sample of training will be Nsample_template * Nround
selection_factor = 2
snr_threshold = 8

source_type = 'BNS'
detector_names = ['H1', 'L1', 'V1'] 
duration = 320
f_low = 20
f_ref = 20
sampling_frequency = 2048
waveform_approximant = 'IMRPhenomPv2_NRTidal'
parameter_names = PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
PSD_type = 'zero_noise' #'zero_noise'
use_sealgw_detector = True

whiten_pca = True
n_components=256
ipca_gen = embedding.pca.IPCAGenerator( n_components, ['plus', 'cross'], decomposition='exp_unwrap', whiten = whiten_pca)

for i in range(Nround):
        print(f'round {i}.')
        injection_parameters_template = generate_BNS_injection_parameters(
                Nsample_template,
                a_max=0.1,
                d_min=10,
                d_max=200,
                d_power=1,
                tc_min=-0.1,
                tc_max=0.1)


        data_template_generator = DataGeneratorBilbyFD(source_type,
                detector_names, 
                duration, 
                f_low, 
                f_ref, 
                sampling_frequency, 
                waveform_approximant, 
                parameter_names,
                PSD_type=PSD_type,
                use_sealgw_detector=use_sealgw_detector,
                snr_threshold=snr_threshold)

        data_template_generator.generate_waveforms(injection_parameters_template)
        #ipca_gen.fit(data_template_generator.rearrange_waveforms_for_ipca(data_template_generator.waveforms['waveform_polarizations']))
        ipca_gen.fit(data_template_generator.waveforms['waveform_polarizations'])
        
        data_template_generator.initialize_waveforms()

output_dir = 'ipca_models/hphc'
#modelutils.save_model(f'{output_dir}/IPCA_BNSFD_10000to500_ExpUnwrap_fixtc_highspin_200Mpc.pickle', ipca_gen)
modelutils.save_model(f'{output_dir}/IPCA_HPHC_BNSFD_10000to256_ExpUnwrap_lowspin_200Mpc.pickle', ipca_gen)