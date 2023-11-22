# export LD_LIBRARY_PATH=/Users/qianhu/MultiNest/lib
# nohup python peloc_TaylorF2ecc.py >nh_TaylorF2ecc.out &
# conda activate myigwn-py39
# export OMP_NUM_THREADS=8

import bilby
import numpy as np
import matplotlib.pyplot as plt
import os
#os.environ['DYLD_LIBRARY_PATH'] = "/Users/qianhu/MultiNest/lib"
#os.environ['LD_LIBRARY_PATH'] = "/Users/qianhu/MultiNest/lib"
#import pymultinest


duration = 32
sampling_frequency = 2048
label = 'lowspin'
outdir = f'full_PE_runs/{label}'
if not os.path.exists(outdir):
    os.mkdir(outdir)


bilby.core.utils.setup_logger(outdir=outdir, label=label)

np.random.seed(11)


injection_parameters = dict(
        chirp_mass=1.74, mass_ratio=0.9, a_1=0.04, a_2=0.03, tilt_1=0.2, tilt_2=0.3,
        phi_12=0.4, phi_jl=0.1, luminosity_distance=150., theta_jn=0.4, psi=2.6,
        phase=1.3, geocent_time=0., ra=3.3, dec=-1.2)

#injection_parameters = dict(
#        chirp_mass=1.74, mass_ratio=0.9, a_1=0.4, a_2=0.3, tilt_1=1.2, tilt_2=0.8,
#        phi_12=0.4, phi_jl=0.9, luminosity_distance=150., theta_jn=0.4, psi=2.6,
#        phase=1.3, geocent_time=0., ra=3.3, dec=-1.2)
#injection_parameters = bilby.gw.conversion.generate_all_bbh_parameters(injection_parameters)

#print(injection_parameters)
waveform_arguments = dict(waveform_approximant='IMRPhenomPv2_NRTidal', minimum_frequency=20., reference_frequency=20)


waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration, sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=waveform_arguments)


# %%
#ifos = bilby.gw.detector.InterferometerList(['ET', 'CE'])
ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])

ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency, duration=duration,
    start_time=injection_parameters['geocent_time'] - duration + 1)

ifos.inject_signal(waveform_generator=waveform_generator,
                   parameters=injection_parameters, raise_error = False)

#ifos.plot_data(outdir=outdir)
ifos.save_data(outdir=outdir,label=label)
ifos.to_pickle(outdir=outdir,label=label)
priors = {}
#priors = injection_parameters.copy()
#priors = bilby.gw.prior.BNSPriorDict()
#priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=25, maximum=100)
#priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
#priors.pop('chirp_mass')
#priors.pop('mass_ratio')
priors['a_1'] = bilby.gw.prior.Uniform(name='a_1', minimum=0, maximum=0.1)
priors['a_2'] = bilby.gw.prior.Uniform(name='a_2', minimum=0, maximum=0.1)
priors['tilt_1'] = bilby.core.prior.analytical.Sine(name='tilt_1')
priors['tilt_2'] = bilby.core.prior.analytical.Sine(name='tilt_2')
priors['phi_12'] = bilby.gw.prior.Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors['phi_jl'] = bilby.gw.prior.Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic')

priors['mass_1'] = bilby.gw.prior.Uniform(name='mass_1', minimum=1, maximum=3)
#priors['mass_2'] = bilby.gw.prior.Uniform(name='mass_2', minimum=1, maximum=3)
priors['mass_2'] = bilby.core.prior.ConditionalUniform(
    name='mass_2',
    condition_func=bilby.gw.prior.secondary_mass_condition_function,
    minimum=1,
    maximum=3,
)

priors['lambda_1'] = bilby.gw.prior.Uniform(name='lambda_1', minimum=0, maximum=5000)
priors['lambda_2'] = bilby.gw.prior.Uniform(name='lambda_2', minimum=0, maximum=5000)

#priors['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=10, maximum=200)
priors['luminosity_distance'] = bilby.gw.prior.UniformComovingVolume(name='luminosity_distance', minimum=10, maximum=200)
priors['dec'] = bilby.core.prior.analytical.Cosine(name='dec')
priors['ra'] = bilby.gw.prior.Uniform(name='ra', minimum=0, maximum=2 * np.pi, boundary='periodic')
priors['theta_jn'] = bilby.core.prior.Sine(name='theta_jn')
priors['psi'] = bilby.gw.prior.Uniform(name='psi', minimum=0, maximum=np.pi, boundary='periodic')
priors['phase'] = bilby.gw.prior.Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic')

priors['geocent_time'] = bilby.core.prior.Uniform(
    minimum=injection_parameters['geocent_time'] - 0.1,
    maximum=injection_parameters['geocent_time'] + 0.1,
    name='geocent_time', latex_label='$t_c$', unit='$s$')

likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=waveform_generator, priors=priors)


#samplername = 'pymultinest'
samplername = 'dynesty'

result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler=samplername, nlive=2000, nact=10, dlogz=0.1,
    injection_parameters=injection_parameters, outdir=outdir, label=label, npool=4)


'''
plot_keys = ['chirp_mass','mass_ratio','ra', 'dec']
plot_parameters={}
for key in plot_keys:
    plot_parameters[key] = injection_parameters[key]

plot_parameters = injection_parameters.copy()
fixed_paras = ['a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl']
for key in fixed_paras:
    plot_parameters.pop(key)
'''
#result.plot_corner(parameters=plot_parameters,quantiles=[0.05,0.95])
plot_parameters = injection_parameters.copy()
result.plot_corner(parameters=plot_parameters,quantiles=[0.05,0.95])
