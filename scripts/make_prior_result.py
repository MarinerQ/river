import numpy as np
import bilby 
import pycbc 
import sys
import matplotlib.pyplot as plt
import pandas as pd
import zuko
from glasflow import RealNVP, CouplingNSF

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import pickle
from sklearn.decomposition import IncrementalPCA
import sklearn.decomposition 
import os
import json

import river.data
from river.data.datagenerator import DataGeneratorBilbyFD
from river.data.dataset import DatasetStrainFD
#import river.data.utils as datautils
from river.data.utils import *

from river.models import embedding
from river.models.utils import *
from river.models.embedding.conv import EmbeddingConv1D,EmbeddingConv2D
config_path = "trained_models/zukonsf_conv1dconv2d_a1d200"
with open(f"{config_path}/config.json", 'r') as f:
    config = json.load(f)

config_datagenerator = config['data_generator_parameters']
config_training = config['training_parameters']
config_flow = config['model_parameters']['flow']
config_embd_proj = config['model_parameters']['embedding_proj']
config_embd_noproj = config['model_parameters']['embedding_noproj']
detector_names = config_datagenerator['detector_names']

Ntest = 8000
injection_parameters_test = generate_BNS_injection_parameters(Nsample = 2*Ntest, **config_datagenerator)
data_generator_test = DataGeneratorBilbyFD(**config_datagenerator)

data_generator_test.inject_signals(injection_parameters_test, 2*Ntest, Ntest)
data_generator_test.numpy_starins()

result_prior = make_prior(data_generator_test.data['injection_parameters'], PARAMETER_NAMES_CONTEXT_PRECESSINGBNS_BILBY)
result_prior.save_to_file(filename='prior_samples.hdf5')