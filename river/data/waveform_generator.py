import numpy as np
import bilby
import lal 
import lalsimulation

from .utils import tau_of_f

LAL_MTSUN_SI = lal.MTSUN_SI
LAL_PI = lal.PI
LAL_GAMMA = lal.GAMMA
Pi_p2 = LAL_PI**2


class WaveformGeneratorMultiBandFD:
    def __init__(self,
            source_type,
            f_low, 
            f_ref, 
            f_high, 
            waveform_approximant, 
            scheme = 'search_Npoints', # uniform_fm53 or search_Npoints
            N_points = None, # number of data points within each band
            N_bands = None, # number of bands
            frequency_domain_source_model = None,
            ref_m1 = 1.,
            ref_m2 = 1.,
            _SAFE_DURATION_FACTOR = 1,
            **kwargs):

        # set properties
        self.source_type = source_type
        self.f_low = f_low 
        self.f_ref = f_ref
        self.f_high = f_high
        self.ref_m1 = ref_m1
        self.ref_m2 = ref_m2
        self.waveform_approximant = waveform_approximant
        self.scheme = scheme

        self.N_points = N_points
        self.N_bands = N_bands
        if scheme == 'uniform_fm53':
            assert self.N_bands is not None, "N_bands must be specified for scheme 'uniform_fm53'"
        if scheme == 'search_Npoints':
            assert self.N_points is not None, "N_points must be specified for scheme 'search_Npoints'"


        self._test_injection_parameters = dict(chirp_mass=1.22, mass_ratio=1, a_1=0.0, a_2=0.0,
                                               tilt_1=0.,tilt_2=0.,phi_12=0.,phi_jl=0.,
                                               lambda_1=425, lambda_2=425,luminosity_distance=1.,
                                               theta_jn=0.0,phase=0)
        self._SAFE_DURATION_FACTOR = _SAFE_DURATION_FACTOR


        if frequency_domain_source_model is None:
            if source_type == 'BNS':
                self.frequency_domain_source_model = bilby.gw.source.lal_binary_neutron_star
            elif source_type == 'BBH':
                self.frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole
            elif source_type == 'BBH_ecc':
                self.frequency_domain_source_model = bilby.gw.source.lal_eccentric_binary_black_hole_no_spins
            else:
                raise Exception("Can not assign frequency_domain_source_model!")
        self.initialize_waveform_generator()


    def tau_of_f(self, f, m1=None, m2=None, mc=None, chi=0):
        tau = tau_of_f(f, m1, m2) * self._SAFE_DURATION_FACTOR
        return tau

    def N_of_f1f2(self, f1,f2,m1=1.4, m2=1.4):
        duration = (self.tau_of_f(f1,m1, m2)-self.tau_of_f(f2,m1, m2)) 
        length = duration*(f2-f1) 
        return duration, int(length) + 1
    
    def get_bands(self, N, f0, f_final, m1, m2):
        if self.scheme == 'uniform_fm53':
            fm53 = np.linspace(self.f_high**(-5/3), self.f_low**(-5/3), self.N_bands+1)
            frequency_nodes = (fm53**(-3/5))[::-1] 
            time_nodes = -self.tau_of_f(frequency_nodes, self.ref_m1, self.ref_m2)
            durations = np.ceil(np.diff(time_nodes)).astype(np.int64)

            return frequency_nodes, durations
        elif self.scheme == 'search_Npoints':
            f_temp = f_final
            tau_temp = self.tau_of_f(f_temp, m1, m2)
            frequency_nodes = [f_final]

            durations = []

            f_to_search = np.linspace(f0, f_final, 100000)[::-1]
            tau_to_search = self.tau_of_f(f_to_search, m1, m2) 
            for i,f_search in enumerate(f_to_search):
                duration = tau_temp - tau_to_search[i]
                nn = int(duration*(f_search-f_temp)) + 1
                #duration, nn = self.N_of_f1f2(f_temp, f_search, m1, m2)
                if nn >= N:
                    durations.append(abs(duration))
                    frequency_nodes.append(f_search)
                    f_temp = f_search
                    tau_temp = self.tau_of_f(f_temp, m1, m2)
            
            dur, N = self.N_of_f1f2(f_search, frequency_nodes[-1], m1, m2)
            durations.append(dur)
            frequency_nodes.append(f_search)
            self.N_bands = len(frequency_nodes)-1

            frequency_nodes = np.array(frequency_nodes)
            durations = np.ceil(durations).astype(np.int64)

            return frequency_nodes[::-1], durations[::-1]
    
    def initialize_waveform_generator(self):
        frequency_nodes, durations = self.get_bands(self.N_points, self.f_low, self.f_high, self.ref_m1, self.ref_m2)        
        
        self.frequency_nodes = frequency_nodes
        self.durations = durations
        self.sampling_frequencies = []
        self.waveform_generator_list = []
        self.farray_list = []
        self.fmask_list = []
        self.duration_array = np.array([])

        for i in range(self.N_bands):
            temp_f_low = frequency_nodes[i]
            temp_f_high = frequency_nodes[i+1]
            temp_duration = durations[i]
            temp_sampling_frequency = int(2**np.ceil(np.log2(2*temp_f_high)))

            self.sampling_frequencies.append(temp_sampling_frequency)
            
            waveform_arguments = dict(waveform_approximant=self.waveform_approximant,
                minimum_frequency=temp_f_low,
                maximum_frequency=temp_f_high,
                reference_frequency=self.f_ref)
            
            waveform_generator = bilby.gw.WaveformGenerator(
                duration=temp_duration, sampling_frequency=temp_sampling_frequency,
                frequency_domain_source_model=self.frequency_domain_source_model,
                waveform_arguments=waveform_arguments)
            
            _test_hp = waveform_generator.frequency_domain_strain(self._test_injection_parameters)['plus']
            non_zero_mask = abs(_test_hp) != 0
            self.waveform_generator_list.append(waveform_generator)
            temp_fmask = (waveform_generator.frequency_array >= temp_f_low) * (waveform_generator.frequency_array <= 2*temp_f_high) * non_zero_mask
            self.fmask_list.append(temp_fmask)
            
            masked_farray = waveform_generator.frequency_array[temp_fmask]
            self.farray_list.append(masked_farray)

            self.duration_array = np.append(self.duration_array, np.array([temp_duration]*len(masked_farray)))
        
        f_array = np.array([])
        for ff in zip(self.farray_list):
            f_array = np.append(f_array, ff)

        self.length = len(f_array)
        self.frequency_array = f_array

        self.sampling_frequencies = np.array(self.sampling_frequencies)
        
    def frequency_domain_strain(self, injection_parameters):
        waveform_polarizations = {'plus': np.array([]), 'cross':np.array([])}
        for i in range(self.N_bands):
            waveform_generator = self.waveform_generator_list[i]
            fmask = self.fmask_list[i]
            wave_dict = waveform_generator.frequency_domain_strain(injection_parameters)
            for mode in ['plus', 'cross']:
                waveform_polarizations[mode] = np.append(waveform_polarizations[mode], wave_dict[mode][fmask])

        return waveform_polarizations

def IMRPhenomPv2_NRTidal_FrequencySequence(farray, f_ref, injection_parameters):
    
    injection_parameters['reference_frequency'] = f_ref
    mc = injection_parameters['chirp_mass']
    q = injection_parameters['mass_ratio']
    m1, m2 = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(mc,q)
    injection_parameters['mass_1'] = m1
    injection_parameters['mass_2'] = m2
    injection_parameters['frequencies'] = farray
    injection_parameters['waveform_approximant'] = 'IMRPhenomPv2_NRTidal'
    return bilby.gw.source.binary_neutron_star_frequency_sequence(farray, **injection_parameters)


class WaveformGeneratorAdaptiveDownsampleFD:
    def __init__(self,
                 f_low, 
            f_ref, 
            f_high, 
            full_duration,
            N_bands, 
            waveform_function, 
            ref_m1 = 1.,
            ref_m2 = 1.,
            _SAFE_DURATION_FACTOR = 1.1,
            **kwargs):

        # set properties
        self.f_low = f_low 
        self.f_ref = f_ref
        self.f_high = f_high
        self.N_bands = N_bands
        self.full_duration = full_duration
        self.ref_m1 = ref_m1
        self.ref_m2 = ref_m2 
        self._SAFE_DURATION_FACTOR = _SAFE_DURATION_FACTOR
        self.waveform_function = waveform_function
        
        fm53 = np.linspace(self.f_high**(-5/3), self.f_low**(-5/3), self.N_bands+1)
        self.frequency_nodes = (fm53**(-3/5))[::-1] 
        self.time_nodes = self.tau_of_f(self.frequency_nodes, self.ref_m1, self.ref_m2)

        farray_fullband = np.linspace(self.f_low, self.f_high, (self.f_high-self.f_low)*self.full_duration+1 )
        farray_downsampled = np.array([])
        for i in range(self.N_bands):
            flow = self.frequency_nodes[i] 
            fhigh = self.frequency_nodes[i+1]
            mask = (farray_fullband>=flow) * (farray_fullband<fhigh)

            tlow = self.time_nodes[i]
            thigh = self.time_nodes[i+1] 

            nk = int( self.full_duration / (tlow-thigh) )
            farray = farray_fullband[mask][::nk] 
            farray_downsampled = np.append(farray_downsampled, farray)
        
        self.farray_downsampled = farray_downsampled


    def tau_of_f(self, f, m1=None, m2=None, mc=None, chi=0):
        tau = tau_of_f(f, m1, m2) * self._SAFE_DURATION_FACTOR
        return tau

    def get_full_farray(self):
        return np.linspace(self.f_low, self.f_high, (self.f_high-self.f_low)*self.full_duration+1 ) 
    
    def get_full_fmask(self):
        full_farray = self.get_full_farray()
        full_mask = np.array([False]*len(full_farray))
        for i in range(self.N_bands):
            flow = self.frequency_nodes[i]
            fhigh = self.frequency_nodes[i+1]
            mask = (full_farray >= flow) & (full_farray < fhigh)
            tlow = self.time_nodes[i]
            thigh = self.time_nodes[i+1]
            nk = int(self.full_duration / (tlow - thigh))
            
            sub_mask = np.full(np.sum(mask), False)
            sub_mask[::nk] = True
            full_mask[mask] = sub_mask
        return full_mask
    
    def frequency_domain_strain(self, injection_parameters):
        return self.waveform_function(self.farray_downsampled, self.f_ref, injection_parameters)