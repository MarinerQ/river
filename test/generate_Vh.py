import warnings
warnings.filterwarnings("ignore", "in-band")

import numpy as np
import bilby 
import pycbc 
import sys
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import pickle
#from sklearn.utils.extmath import randomized_svd
import sklearn
from sklearn.decomposition import IncrementalPCA, randomized_svd, KernelPCA
import sklearn.decomposition 

import river.data
from river.data.datagenerator import DataGeneratorBilbyFD
#from river.data.dataset import #DatasetStrainFD, #DatasetStrainFDFromPreCalSVDWF
#import river.data.utils as datautils
from river.data.utils import *

from river.models.utils import *

import lal

source_type = 'BNS'
detector_names = ['ET', 'CE', 'CEL']
duration = 320
f_low = 50
f_high = 1024

f_ref = 20
sampling_frequency = 2048
waveform_approximant = 'IMRPhenomPv2_NRTidal'
parameter_names = PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
PSD_type = 'zero_noise' #'zero_noise' bilby_default
use_sealgw_detector = True

data_generator = DataGeneratorBilbyFD(source_type,
            detector_names, 
            duration, 
            f_low, 
            f_ref, 
            sampling_frequency, 
            waveform_approximant, 
            parameter_names,
            PSD_type=PSD_type,
            use_sealgw_detector=use_sealgw_detector,
                                     f_high=f_high)

data_generator_test = DataGeneratorBilbyFD(source_type,
            detector_names, 
            duration, 
            f_low, 
            f_ref, 
            sampling_frequency, 
            waveform_approximant, 
            parameter_names,
            PSD_type=PSD_type,
            use_sealgw_detector=use_sealgw_detector,
                                     f_high=f_high)

f = data_generator.frequency_array_masked

Nsample = 20000
Ntest = 100


injection_parameters_test = generate_BNS_injection_parameters(
        2*Ntest,
        a_max=0.8,
        d_min=10,
        d_max=2000,
        d_power=1,
        intrinsic_only = False,
        tc_min = -0.001,
        tc_max = 0.001)

injection_parameters_test['luminosity_distance'] = np.zeros(Ntest) + 1


def validate_svd(V, Vh, h_test, nbasis, mm_lowthre = 1e-30):
    
    mismatches = []
    for h in h_test:
        h_compressed = h @ V[:, :nbasis] #np.dot(h_test , V)
        h_reconstructed = h_compressed @ Vh[:nbasis] #np.dot(h_compressed , Vh)

        norm1 = np.sqrt(np.sum(np.abs(h) ** 2))
        norm2 = np.sqrt(np.sum(np.abs(h_reconstructed) ** 2))
        inner = np.sum(h.conj() * h_reconstructed).real
        mismatch = 1 - inner / (norm1 * norm2)
        if mismatch<mm_lowthre:
            mismatches.append(mm_lowthre)
        else:
            mismatches.append(mismatch)

    return mismatches

outdir = 'Vdet_output_501024512_320s'

n_components=512

Nround = 20
Nsample_template = 1000

ipca_dict = {'ET1': IncrementalPCA(n_components=n_components),
             'ET2': IncrementalPCA(n_components=n_components),
             'ET3': IncrementalPCA(n_components=n_components),
             'CE': IncrementalPCA(n_components=n_components),
             'CEL': IncrementalPCA(n_components=n_components)}

for i in range(Nround):
    print(f'round {i}.')
    injection_parameters_template = generate_BNS_injection_parameters(
        1024,
        a_max=0.8,
        d_min=10,
        d_max=200,
        d_power=2,
        intrinsic_only = False,
        tc_min = -0.001,
        tc_max = 0.001)
    injection_parameters_template['luminosity_distance'] = np.zeros(Nsample_template) + 1

    data_generator.inject_signals(injection_parameters_template, 1024, Nneeded=Nsample_template)


    for det in data_generator.ifos:        
        h_train = np.array(data_generator.data['strains'][det.name])
        ipca = ipca_dict[det.name]
    
        ipca.fit(np.real(h_train))
        ipca.fit(np.imag(h_train))
    
    
    del h_train, data_generator
    data_generator = DataGeneratorBilbyFD(source_type,
            detector_names, 
            duration, 
            f_low, 
            f_ref, 
            sampling_frequency, 
            waveform_approximant, 
            parameter_names,
            PSD_type=PSD_type,
            use_sealgw_detector=use_sealgw_detector,
                                     f_high=f_high)
    
    print(f'round {i} finished.')
    
    


data_generator_test.inject_signals(injection_parameters_test, 2*Ntest, Nneeded=Ntest)
for det in data_generator.ifos:
    detname = det.name
    ipca = ipca_dict[det.name]
    h_test = data_generator_test.data['strains'][det.name]
    V = ipca.components_.T
    Vh = ipca.components_
    
    with open(f"{outdir}/Vh_{detname}.pickle", 'wb') as f:
        pickle.dump(Vh, f)
    
    mm_list = []
    ntests = np.arange(256, 513, 16)
    for nbasis_test in ntests:
        mm_list.append(validate_svd(V, Vh, h_test, nbasis_test))
    
    
    logMM = np.log10(mm_list)
    plt.figure(figsize=(20,10))
    _ = plt.violinplot(logMM.T, quantiles=np.tile([0.01,0.99], logMM.shape[0]).reshape(2, logMM.shape[0]))
    
    plt.title(f'SVD error for {detname}')
    plt.ylabel('log10 MM')
    plt.xlabel('Number of bases')
    plt.xticks(np.arange(0,len(ntests),4)+1, labels=ntests[::4])

    plt.grid()
    #plt.savefig(f'outputs/mm_vs_nbasis_{f_low}Hz{f_high}Hz{duration}s.png')
    #plt.xscale('log')
    plt.ylim(-15,0)


    plt.savefig(f'{outdir}/nb_{detname}.png')
