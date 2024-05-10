import numpy as np
import bilby
import spiir.io
#import utils as datautils
from .utils import load_dict_from_hdf5, save_dict_to_hdf5, generate_random_distance
import sealgw.simulation as sealsim
import pickle


class DataGeneratorBilbyFD:
    def __init__(self,
            source_type,
            detector_names, 
            duration, 
            f_low, 
            f_ref, 
            sampling_frequency, 
            waveform_approximant, 
            parameter_names,
            frequency_domain_source_model = None,
            f_high=None,
            PSD_type = 'bilby_default',
            custom_psd_path=None,
            use_sealgw_detector=False,
            snr_threshold = 8,
            ipca=None,
            Vh=None,
            **kwargs):

        # set properties
        self.source_type = source_type
        self.detector_names = detector_names
        self.parameter_names = parameter_names
        self.duration = duration
        self.f_low = f_low 
        self.f_ref = f_ref
        self.sampling_frequency = sampling_frequency
        if f_high:
            self.f_high = f_high
        else:
            self.f_high = sampling_frequency / 2
        self.use_sealgw_detector = use_sealgw_detector
        self.snr_threshold = snr_threshold
        self.scaled = False
        self.whitened = False
        self.numpyed = False

        # set ifos
        if use_sealgw_detector:
            self.ifos = sealsim.sealinterferometers.SealInterferometerList(detector_names)
        else:
            self.ifos = bilby.gw.detector.InterferometerList(detector_names)

        for det in self.ifos:
            det.sampling_frequency = sampling_frequency
            det.duration = duration
            det.frequency_mask = (det.frequency_array >= self.f_low) * (det.frequency_array <= self.f_high)
            if use_sealgw_detector:
                det.antenna_response_change = False

        self.frequency_mask = det.frequency_mask
        self.frequency_array = det.frequency_array
        self.frequency_array_masked = det.frequency_array[det.frequency_mask]

        # set waveform
        self.waveform_arguments = dict(waveform_approximant=waveform_approximant,
            minimum_frequency=self.f_low,
            maximum_frequency=self.f_high,
            reference_frequency=self.f_ref)

        if frequency_domain_source_model is None:
            if source_type == 'BNS':
                frequency_domain_source_model = bilby.gw.source.lal_binary_neutron_star
            elif source_type == 'BBH':
                frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole
            elif source_type == 'BBH_ecc':
                frequency_domain_source_model = bilby.gw.source.lal_eccentric_binary_black_hole_no_spins
            else:
                raise Exception("Can not assign frequency_domain_source_model!")
        self.waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
            frequency_domain_source_model=frequency_domain_source_model,
            waveform_arguments=self.waveform_arguments)

        # set PSD
        self.PSD_type = PSD_type
        self.custom_psd_path = custom_psd_path
        if PSD_type in ['bilby_default', 'zero_noise']: 
            print(f"Using {PSD_type} PSDs to generate data.")
        elif PSD_type == 'custom':
            assert custom_psd_path is not None, 'Not using default PSD but did not provide PSD'
            print(f'Using custom PSD {custom_psd_path}')
            for i in range(len(self.ifos)):
                det = self.ifos[i]
                if type(custom_psd_path) == str and '.xml' in custom_psd_path:
                    temppsd = spiir.io.ligolw.array.load_psd_series_from_xml(
                        custom_psd_path
                    )[detector_names[i]]
                    psdarray = temppsd.to_numpy()
                    freqarray = temppsd.index.to_numpy()
                    psdarray[
                        freqarray >= 972.5
                    ] = 1.0  
                    psd = bilby.gw.detector.PowerSpectralDensity(
                        frequency_array=freqarray, psd_array=psdarray
                    )
                elif type(custom_psd_path) == list:
                    psd_file = custom_psd_path[i]
                    psd = bilby.gw.detector.PowerSpectralDensity(psd_file=psd_file)
                det.power_spectral_density = psd
        elif PSD_type == 'real_noise':
            raise Exception('Under development!')
        else:
            raise Exception('Wrong PSD type!')

        # set data
        self.initialize_data()

        # set precalculated waveforms
        self.initialize_waveforms()
        if type(ipca) == str:
            with open(ipca, 'rb') as f:
                model = pickle.load(f)
            self.ipca = model.pca_dict
        else:
            self.ipca = ipca

        if type(Vh) == str:
            with open(Vh, 'rb') as f:
                self.Vh = pickle.load(f)
        else:
            self.Vh = Vh
        if self.Vh is not None:
            self.V = self.Vh.T.conj()

        if (self.Vh is not None) and (self.ipca is not None):
            raise ValueError("Got both IPCA and Vh!")

    def initialize_data(self):
        self.data = {}
        self.data['farray'] = self.frequency_array_masked
        self.data['strains'] = {}
        self.data['PSDs'] = {}
        self.data['injection_parameters'] = {}
        self.data['SNRs'] = {}
        for detname in self.detector_names:
            if detname == 'ET':
                for dn in ['ET1', 'ET2', 'ET3']:
                    self.data['strains'][dn] = []
                    self.data['PSDs'][dn] = []
                    self.data['SNRs'][dn] = []
            else:
                self.data['strains'][detname] = []
                self.data['PSDs'][detname] = []
                self.data['SNRs'][detname] = []
        for paraname in self.parameter_names:
            self.data['injection_parameters'][paraname] = []
        self.data['Nsample'] = [0]
        self.Nsample=0

        
    def update_data(self, injection_parameters):
        for det in self.ifos:
            detname = det.name
            self.data['strains'][detname].append(det.frequency_domain_strain[det.frequency_mask])
            self.data['PSDs'][detname].append(det.power_spectral_density_array[det.frequency_mask])
            self.data['SNRs'][detname].append(abs(det.meta_data['matched_filter_SNR']))
        for paraname in self.parameter_names:
            self.data['injection_parameters'][paraname].append(injection_parameters[paraname])
        self.Nsample+=1
        self.data['Nsample'][0] += 1

    def get_injected_snr(self):
        netsnr = 0
        for key, data in self.ifos.meta_data.items():
            netsnr += np.abs(data['matched_filter_SNR'])**2
        netsnr = netsnr**0.5
        return netsnr
    
    def inject_one_signal(self, injection_parameters):
        if self.PSD_type in ['bilby_default', 'custom']:
            self.ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=self.sampling_frequency, duration=self.duration,
                start_time=injection_parameters['geocent_time'] - self.duration + 1)
        elif self.PSD_type == 'zero_noise':
            self.ifos.set_strain_data_from_zero_noise(
                sampling_frequency=self.sampling_frequency, duration=self.duration,
                start_time=injection_parameters['geocent_time'] - self.duration + 1)
        else:
            raise Exception("Under development!")

        if self.use_sealgw_detector:
            self.ifos.inject_signal(waveform_generator=self.waveform_generator,
                            parameters=injection_parameters,raise_error=False, print_snr=False, print_para=False)
        else:
            self.ifos.inject_signal(waveform_generator=self.waveform_generator,
                            parameters=injection_parameters,raise_error=False)

        netsnr = self.get_injected_snr()
        if netsnr>=self.snr_threshold:
            self.update_data(injection_parameters)
        else:
            print(f'SNR={netsnr}<{self.snr_threshold}, this injection is not recorded.')

    def inject_signals(self, injection_parameters_all, Ninj, Nneeded=None):
        if Nneeded is None:
            Nneeded = Ninj
            print("Nneeded not set. Actual number of injection may be less than Ninjection due to SNR threshold. ")
        elif Nneeded>Ninj:
            raise ValueError("Needed > Ninj!")
        else:
            pass

        for i_inj in range(Ninj):
            if self.Nsample >= Nneeded:
                break 
            print(f"Injecting {i_inj}-th signal, {round(100*i_inj/Ninj,2)}% done")
            injection_parameters = {}
            for paraname in self.parameter_names:
                injection_parameters[paraname] = injection_parameters_all[paraname][i_inj]
            self.inject_one_signal(injection_parameters)

        if self.Nsample < Nneeded:
            print(f"Actual number of injection ({self.Nsample}) is less than Ninjection due to SNR threshold. ")

    def get_data(self, i_data):
        if i_data>self.Nsample:
            raise ValueError(f"i_data ({i_data}) > Nsample ({self.Nsample})!")
        injection_parameters = {}
        for paraname in self.parameter_names:
            injection_parameters[paraname] = self.data.injection_parameters[paraname][i_data]
        strains = {}
        PSDs = {}
        SNRs = {}
        for detname in self.detector_names:
            strains[detname] = self.data['strains'][detname][i_data]
            PSDs[detname] = self.data['PSDs'][detname][i_data]
            SNRs[detname] = self.data['SNRs'][detname][i_data]

        return injection_parameters, strains, PSDs
        
    def save_data(self, filename):
        if self.Nsample == 0:
            raise Exception("Data is empty!")
        
        save_dict_to_hdf5(self.data, filename)
        print(f"File saved to {filename}")
    
    def load_data(self, filename):
        new_data = load_dict_from_hdf5(filename)
        if (new_data['farray'] != self.frequency_array_masked).any():
            raise Exception("Frequency arrays do not match!")
        if set(list(self.data['injection_parameters'].keys())) != set(list(new_data['injection_parameters'].keys())):
            raise Exception(("Parameter names do not match!"))
        
        for detname in self.detector_names:
            self.data['strains'][detname]+= new_data['strains'][detname]
            self.data['PSDs'][detname]+= new_data['PSDs'][detname]
            self.data['SNRs'][detname]+= new_data['SNRs'][detname]
        for paraname in self.parameter_names:
            self.data['injection_parameters'][paraname]+= new_data['injection_parameters'][paraname]

        self.Nsample += new_data['Nsample'][0]
        self.data['Nsample'][0] += new_data['Nsample'][0]
    
    def numpy_starins(self):
        for detname in self.detector_names:
            self.data['strains'][detname] = np.array(self.data['strains'][detname])
            self.data['PSDs'][detname] = np.array(self.data['PSDs'][detname])
            self.data['SNRs'][detname] = np.array(self.data['SNRs'][detname])
        self.numpyed = True

    def scale_strains(self):
        assert self.whitened == False, 'Strain already whitened!'
        for detname in self.detector_names:
            self.data['strains'][detname] *= 1e23
        self.scaled = True
    
    def whiten_strains(self):
        assert self.scaled == False, 'Strain already scaled!'
        for detname in self.detector_names:
            self.data['strains'][detname] /= self.data['PSDs'][detname]**0.5
        self.whitened = True


    #### to deal with large training set, we should generate waveforms and store them in disk
    #### when inject, read those waveforms and apply extrinsic parameters
    def initialize_waveforms(self):
        self.waveforms = {}
        #self.waveforms['farray'] = self.frequency_array_masked
        self.waveforms['waveform_polarizations'] = {}
        for mode in ['plus', 'cross']:
            self.waveforms['waveform_polarizations'][mode] = {}
            for part in ['amplitude', 'phase']:
                self.waveforms['waveform_polarizations'][mode][part] = []

        self.waveforms['injection_parameters'] = {}
        for paraname in self.parameter_names:
            if paraname not in ['luminosity_distance', 'ra', 'dec', 'psi', 'geocent_time']:
                self.waveforms['injection_parameters'][paraname] = []
        self.Nwaveform = 0
        #self.waveforms['SNR_at_1Mpc'] = []

    def update_waveforms(self, injection_parameters, wave_dict):
        for paraname in self.parameter_names:
            if paraname not in ['luminosity_distance', 'ra', 'dec', 'psi', 'geocent_time']:
                self.waveforms['injection_parameters'][paraname].append(injection_parameters[paraname])

        for mode in ['plus', 'cross']:
            for part in ['amplitude', 'phase']:
                self.waveforms['waveform_polarizations'][mode][part].append(wave_dict[mode][part])

        self.Nwaveform += 1

    def generate_one_waveform(self, injection_parameters):
        '''
        Generate a waveform with "intrisic" parameters. 
        "Extrisic" parameters (t_c, ra, dec, psi, d_L) are not involved in waveform generation. They will be applied during injection. 
        '''
        injection_parameters['luminosity_distance'] = 1 # fix distance as it scales amplitude 
        wf = self.waveform_generator.frequency_domain_strain(injection_parameters)
        wf_masked = {}
        for key, mode in wf.items():
            h = mode[self.frequency_mask]
            amp = np.abs(h) * 1e23
            phase = np.unwrap(np.angle(h))
            wf_masked[key] = {}

            if self.Vh is not None:
                h_proj = h @ self.V
                wf_masked[key]['amplitude'] = np.abs(h_proj) 
                wf_masked[key]['phase'] = np.angle(h_proj)
            elif self.ipca:
                amp_proj = np.dot(amp, self.ipca[key]['amplitude'].components_.T)
                phase_proj = np.dot(phase, self.ipca[key]['phase'].components_.T)
                wf_masked[key]['amplitude'] = amp_proj
                wf_masked[key]['phase'] = phase_proj
            else:
                wf_masked[key]['amplitude'] = amp 
                wf_masked[key]['phase'] = phase

        self.update_waveforms(injection_parameters, wf_masked)

        
        

    def generate_waveforms(self, injection_parameters_all):
        N = len(injection_parameters_all['chirp_mass'])
        for i in range(N):
            injection_parameters = {}
            for paraname in self.parameter_names:
                injection_parameters[paraname] = injection_parameters_all[paraname][i]

            self.generate_one_waveform(injection_parameters)
            
    
    def reconstruct_waveforms(self, wave_dict, dL = 1):
        '''
        wave_dict should have to keys 'plus' and 'cross', values of which are also dict, with compressed amplidute and phase
        '''
        waveform_polarizations = {}
        for polarization, waveform_component in wave_dict.items():
            if self.ipca:
                ipca_A = self.ipca[polarization]['amplitude']
                ipca_phi = self.ipca[polarization]['phase']
                #A_reconstructed = np.dot(ipca_A.transform([waveform_component['amplitude']])[0], ipca_A.components_) / dL
                #phi_reconstructed = np.dot(ipca_phi.transform([waveform_component['phase']])[0], ipca_phi.components_)
                A_reconstructed = np.dot(waveform_component['amplitude'], ipca_A.components_) / dL / 1e23
                phi_reconstructed = np.dot(waveform_component['phase'], ipca_phi.components_)
                waveform_polarizations[polarization] = A_reconstructed * np.exp(1j *  phi_reconstructed)
            else:
                waveform_polarizations[polarization] = waveform_component['amplitude'] / dL / 1e23 * np.exp(1j *  waveform_component['phase'])

        return waveform_polarizations
    
    def inject_one_signal_from_waveforms(self, injection_parameters, injection_polarizations_compressed):
        if self.PSD_type in ['bilby_default', 'custom']:
            self.ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=self.sampling_frequency, duration=self.duration,
                start_time=injection_parameters['geocent_time'] - self.duration + 1)
        elif self.PSD_type == 'zero_noise':
            self.ifos.set_strain_data_from_zero_noise(
                sampling_frequency=self.sampling_frequency, duration=self.duration,
                start_time=injection_parameters['geocent_time'] - self.duration + 1)
        else:
            raise Exception("Under development!")

        injection_polarizations = self.reconstruct_waveforms(injection_polarizations_compressed,
                                                             dL = injection_parameters['luminosity_distance'])
        # unmask waveforms
        for kk,pp in injection_polarizations.items():
            append_zeros = np.zeros(len(self.frequency_array) - len(self.frequency_array_masked))
            injection_polarizations[kk] = np.append(append_zeros, pp)
        
        if self.use_sealgw_detector:
            self.ifos.inject_signal(injection_polarizations=injection_polarizations,
                            parameters=injection_parameters,raise_error=False, print_snr=False, print_para=False)
        else:
            self.ifos.inject_signal(injection_polarizations=injection_polarizations,
                            parameters=injection_parameters,raise_error=False)

        netsnr = self.get_injected_snr()
        if netsnr>=self.snr_threshold:
            self.update_data(injection_parameters)
            return 1
        else:
            print(f'SNR={netsnr}<{self.snr_threshold}, this injection is not recorded.')
            return 0
    

    def inject_signals_from_waveforms(self, injection_parameters_all, Ninj, Nneeded=None):
        '''
        injection_parameters_all: injection para dict that contains t_c, ra, dec, psi, d_L. Others are not used.
        '''
        if Nneeded is None:
            Nneeded = Ninj
            print("Nneeded not set. Actual number of injection may be less than Ninjection due to SNR threshold. ")
        elif Nneeded>Ninj:
            raise ValueError("Needed > Ninj!")
        else:
            pass

        for i_inj in range(Ninj):
            if self.Nsample >= Nneeded:
                break 
            print(f"Injecting {i_inj}-th signal, {round(100*i_inj/Ninj,2)}% done")
            injection_parameters = self.get_one_injection_parameters(i_inj, injection_parameters_all)
            injection_polarizations_compressed = self.get_one_waveform(i_inj, self.waveforms['waveform_polarizations'])
            self.inject_one_signal_from_waveforms(injection_parameters, injection_polarizations_compressed)

        if self.Nsample < Nneeded:
            print(f"Actual number of injection ({self.Nsample}) is less than Ninjection due to SNR threshold. ")

        return 


    def rearrange_waveforms_for_ipca(self, datagenwaveform):
        '''
        datagenwaveform: self.waveforms['waveform_polarizations']

        the output can be used in ipca_gen.fit() as input
        '''
        rearranged_waveforms = {}
        rearranged_waveforms['plus'] = []
        rearranged_waveforms['cross'] = []

        for ii,dd in enumerate(datagenwaveform):
            rearranged_waveforms['plus'].append(dd['plus']['amplitude'] * np.exp(1j*dd['plus']['phase']))
            rearranged_waveforms['cross'].append(dd['cross']['amplitude'] * np.exp(1j*dd['cross']['phase']))

        return rearranged_waveforms
    
    def save_waveform_data(self, filepath):
        save_dict_to_hdf5(self.waveforms, filepath)
        
    def load_waveform_data(self, filepath):
        wf = load_dict_from_hdf5(filepath)
        self.waveforms = wf 
    


    def get_one_injection_parameters(self, index, parameter_list, is_intrinsic_only=False):
        injection_parameters = {}
        
        for paraname in self.parameter_names:
            if is_intrinsic_only:
                if paraname in [ 'ra', 'dec', 'psi', 'luminosity_distance', 'geocent_time']:
                    continue
            injection_parameters[paraname] = parameter_list[paraname][index]

        return injection_parameters
    
    def get_one_waveform(self, index, waveform_list):
        polarizations = {}
        for mode in ['plus', 'cross']:
            polarizations[mode] = {}
            for part in ['amplitude', 'phase']:
                polarizations[mode][part] = waveform_list[mode][part][index]

        return polarizations

