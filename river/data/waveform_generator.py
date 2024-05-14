import numpy as np
import bilby
import lal 
from .utils import tau_of_f

LAL_MTSUN_SI = lal.MTSUN_SI
LAL_PI = lal.PI
LAL_GAMMA = lal.GAMMA
Pi_p2 = LAL_PI**2


class WaveformGeneratorMultiBandFD:
    def __init__(self,
            source_type,
            N_points, # number of data points within each band
            f_low, 
            f_ref, 
            f_high, 
            waveform_approximant, 
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
        self.N_points = N_points
        self.waveform_approximant = waveform_approximant
        
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
        tau = tau_of_f(f, m1, m2)
        return tau

    def N_of_f1f2(self, f1,f2,m1=1.4, m2=1.4):
        duration = (self.tau_of_f(f1,m1, m2)-self.tau_of_f(f2,m1, m2)) 
        length = duration*(f2-f1) 
        return duration, int(length) + 1
    
    def get_bands(self, N, f0, f_final, m1, m2):
        f_temp = f_final
        tau_temp = self.tau_of_f(f_temp, m1, m2)
        f_list = [f_final]

        N_list = []
        duration_list = []

        f_to_search = np.linspace(f0, f_final, 100000)[::-1]
        tau_to_search = self.tau_of_f(f_to_search, m1, m2)
        for i,f_search in enumerate(f_to_search):
            duration = tau_temp - tau_to_search[i]
            nn = int(duration*(f_search-f_temp)) + 1
            #duration, nn = self.N_of_f1f2(f_temp, f_search, m1, m2)
            if nn >= N:
                N_list.append(nn)
                duration_list.append(abs(duration))
                f_list.append(f_search)
                f_temp = f_search
                tau_temp = self.tau_of_f(f_temp, m1, m2)
        
        dur, N = self.N_of_f1f2(f_search, f_list[-1], m1, m2)
        N_list.append(N)
        duration_list.append(dur)
        f_list.append(f_search)
        return f_list[::-1], N_list[::-1], duration_list[::-1]
    
    def initialize_waveform_generator(self):
        f_list, N_list, duration_list = self.get_bands(self.N_points, self.f_low, self.f_high, self.ref_m1, self.ref_m2)
        #print(f_list, N_list, duration_list)
        self.N_bands = len(f_list)-1
        
        self.f_list = f_list
        self.duration_list = []
        self.sampling_frequency_list = []
        self.N_list = []
        self.waveform_generator_list = []
        self.farray_list = []
        self.fmask_list = []
        
        # interp so that each band has N points
        self.farray_list_interp = []

        for i in range(self.N_bands):
            temp_f_low = f_list[i]
            temp_f_high = f_list[i+1]
            temp_duration = int(duration_list[i])+1
            temp_sampling_frequency = int(min(2*self.f_high, 2*temp_f_high+1)) 

            self.duration_list.append(temp_duration)
            self.sampling_frequency_list.append(temp_sampling_frequency)
            
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
            self.N_list.append(len(masked_farray))
            
            interp_farray = np.linspace(masked_farray[0], masked_farray[-1], self.N_points)
            self.farray_list_interp.append(interp_farray)
        
        f_array = np.array([])
        f_array_interp = np.array([])
        for ff,ffintp in zip(self.farray_list,self.farray_list_interp):
            f_array = np.append(f_array, ff)
            f_array_interp = np.append(f_array_interp, ffintp)
        self.frequency_array = f_array
        self.frequency_array_interp = f_array_interp
        
    def frequency_domain_strain(self, injection_parameters, interp=False):
        waveform_polarizations = {'plus': np.array([]), 'cross':np.array([])}
        for i in range(self.N_bands):
            waveform_generator = self.waveform_generator_list[i]
            fmask = self.fmask_list[i]
            wave_dict = waveform_generator.frequency_domain_strain(injection_parameters)
            for mode in ['plus', 'cross']:
                if interp:
                    fp = self.farray_list[i]
                    hp = wave_dict[mode][fmask]
                    f = self.farray_list_interp[i]
                    #h_interp =  np.interp(f, fp, hp.real) +  np.interp(f, fp, hp.imag)*1j
                    #h_interp =  np.interp(f, fp, hp) 
                    h_interp =  np.interp(f, fp, np.abs(hp))*np.exp(1j*np.interp(f, fp, np.unwrap(np.angle(hp))))
                    waveform_polarizations[mode] = np.append(waveform_polarizations[mode], h_interp)
                else:
                    waveform_polarizations[mode] = np.append(waveform_polarizations[mode], wave_dict[mode][fmask])

        return waveform_polarizations
