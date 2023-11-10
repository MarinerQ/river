#%% 
import numpy as np
import bilby 
import pycbc 
import sys
sys.path.append('/Users/qianhu/Documents/Glasgow/research/river/data')

from datagenerator import DataGeneratorBilbyFD
from utils import *

#%%
Nsample = 10
mass_1, mass_2 = generate_random_component_mass(Nsample, 1.1, 3)
a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl = generate_random_twospins_bilbystyle(Nsample, a_max=0.1)
luminosity_distance = generate_random_distance(Nsample, low=10, high=100, power=3)
theta_jn, ra, dec, psi, phase = generate_random_extrinsic_angles(Nsample)
geocent_time = np.random.uniform(0, 3.14e7, Nsample)
lambda_tilde = np.random.uniform(0, 1000, Nsample)
delta_lambda_tilde = np.random.uniform(-5000, 5000, Nsample)
lambda_1, lambda_2 = bilby.gw.conversion.lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(lambda_tilde, delta_lambda_tilde, mass_1, mass_2)

injection_parameters_all = assemble_pBNS_parameters(mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2, theta_jn, luminosity_distance, ra, dec, psi, phase, geocent_time)

#%%
source_type = 'BNS'
detector_names = ['H1', 'L1', 'V1'] 
duration = 32
f_low = 20
f_ref = 20
sampling_frequency = 2048
waveform_approximant = 'IMRPhenomPv2_NRTidal'
parameter_names = PARAMETER_NAMES_PRECESSINGBNS_BILBY


data_generator = DataGeneratorBilbyFD(source_type,
            detector_names, 
            duration, 
            f_low, 
            f_ref, 
            sampling_frequency, 
            waveform_approximant, 
            parameter_names)

# %%
data_generator.inject_signals(injection_parameters_all, Nsample)
# %%
data_generator.save_data('test1.h5')
# %%
data_generator.load_data('test1.h5')

# %%
