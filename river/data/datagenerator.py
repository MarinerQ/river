import numpy as np
import bilby
import spiir.io
#import utils as datautils
from .utils import load_dict_from_hdf5, save_dict_to_hdf5, generate_random_distance, inner_product
import sealgw.simulation as sealsim
import pickle
from .waveform_generator import WaveformGeneratorMultiBandFD

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
            antenna_response_change=False,
            antenna_response_change_timescale=8,
            snr_threshold = 8,
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
        self.antenna_response_change = antenna_response_change
        self.antenna_response_change_timescale = antenna_response_change_timescale

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
                det.antenna_response_change = self.antenna_response_change
                det.antenna_response_change_timescale = self.antenna_response_change_timescale

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
        
    def get_one_injection_parameters(self, index, parameter_list, is_intrinsic_only=False):
        injection_parameters = {}
        
        for paraname in self.parameter_names:
            if is_intrinsic_only:
                if paraname in [ 'ra', 'dec', 'psi', 'luminosity_distance', 'geocent_time']:
                    continue
            injection_parameters[paraname] = parameter_list[paraname][index]

        return injection_parameters
    

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



    


class DataGeneratorBilbyFDMB(DataGeneratorBilbyFD):
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
                 PSD_type = 'zero_noise',
                 custom_psd_path=None,
                 use_sealgw_detector=True,
                 antenna_response_change=True,
                 antenna_response_change_timescale=8,
                 snr_threshold = 1,
                 N_points = 256,
                 SAFE_DURATION_FACTOR = 2,
                 scheme = 'search_Npoints',
                 **kwargs):
        super().__init__(
            source_type=source_type,
            detector_names=detector_names,
            duration=duration,
            f_low=f_low,
            f_ref=f_ref,
            sampling_frequency=sampling_frequency,
            waveform_approximant=waveform_approximant,
            parameter_names=parameter_names,
            frequency_domain_source_model=frequency_domain_source_model,
            f_high=f_high,
            PSD_type=PSD_type,
            custom_psd_path=custom_psd_path,
            use_sealgw_detector=use_sealgw_detector,
            antenna_response_change=antenna_response_change,
            antenna_response_change_timescale=antenna_response_change_timescale,
            snr_threshold=snr_threshold,
            **kwargs
        )

        assert self.PSD_type == 'zero_noise'
        self.waveform_generator = WaveformGeneratorMultiBandFD(source_type=source_type,
                                    N_points=N_points, 
                                    f_low=f_low,
                                    f_ref=f_ref, 
                                    f_high=f_high, 
                                    waveform_approximant = waveform_approximant,
                                    _SAFE_DURATION_FACTOR=SAFE_DURATION_FACTOR,
                                    scheme=scheme)
        self.farray_mb = self.waveform_generator.frequency_array
        self.N_bands = self.waveform_generator.N_bands
        self.data_length = len(self.farray_mb)
        self.scheme = scheme
        self.duration_array = self.waveform_generator.duration_array



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
    

    def inject_one_signal(self, injection_parameters):

        waveform_polarizations = self.waveform_generator.frequency_domain_strain(injection_parameters)
        temp_data_dict = {}
        temp_data_dict['strains'] = {}
        temp_data_dict['SNRs'] = {}

        netsnr = 0
        if self.PSD_type in ['bilby_default', 'custom']:
            raise Exception("Under development!")
        elif self.PSD_type == 'zero_noise':
            for det in self.ifos:
                detname = det.name
                h_resp = det.get_detector_response(
                        waveform_polarizations, injection_parameters, frequencies=self.farray_mb
                    )
                snr = inner_product(h_resp, h_resp, self.duration_array,
                                     det.power_spectral_density.power_spectral_density_interpolated(self.farray_mb))**0.5
                temp_data_dict['strains'][detname] = h_resp
                temp_data_dict['SNRs'][detname] = snr
                netsnr += snr**2
        else:
            raise Exception("Under development!")


        netsnr = netsnr ** 0.5
        if netsnr>=self.snr_threshold:
            for det in self.ifos:
                detname = det.name 
                self.data['strains'][detname].append(temp_data_dict['strains'][detname])
                self.data['SNRs'][detname].append(temp_data_dict['SNRs'][detname])
            for paraname in self.parameter_names:
                self.data['injection_parameters'][paraname].append(injection_parameters[paraname])
            self.Nsample+=1
            self.data['Nsample'][0] += 1
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