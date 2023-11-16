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


Nsample_template = 10000
injection_parameters_template = generate_BNS_injection_parameters(
        Nsample_template,
        a_max=0.1,
        d_min=10,
        d_max=100,
        d_power=2)

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

data_template_generator = DataGeneratorBilbyFD(source_type,
            detector_names, 
            duration, 
            f_low, 
            f_ref, 
            sampling_frequency, 
            waveform_approximant, 
            parameter_names,
            PSD_type='zero_noise',
            use_sealgw_detector=use_sealgw_detector)

data_template_generator.inject_signals(injection_parameters_template, Nsample_template)

data_template_generator.numpy_starins()
data_template_generator.scale_strains()

n_components=500
ipca_gen = embedding.pca.IPCAGenerator(data_template_generator.data['strains'], n_components, detector_names, decomposition='exp_unwrap')

output_dir = 'ipca_models'
modelutils.save_model(f'{output_dir}/IPCA_BNSFD_10000to500_ExpUnwrap.pickle', ipca_gen)