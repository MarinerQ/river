import numpy as np
import bilby
import spiir.io
#import utils as datautils
from .utils import load_dict_from_hdf5, save_dict_to_hdf5
import sealgw.simulation as sealsim

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
            PSD_type = 'bilby_default',
            custom_psd_path=None,
            use_sealgw_detector=False,
            **kwargs):

        # set properties
        self.source_type = source_type
        self.detector_names = detector_names
        self.parameter_names = parameter_names
        self.duration = duration
        self.f_low = f_low 
        self.f_ref = f_ref
        self.sampling_frequency = sampling_frequency
        self.use_sealgw_detector = use_sealgw_detector
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
            det.frequency_mask = det.frequency_array >= f_low
            if use_sealgw_detector:
                det.antenna_response_change = False

        self.frequency_mask = det.frequency_mask
        self.frequency_array = det.frequency_array
        self.frequency_array_masked = det.frequency_array[det.frequency_mask]

        # set waveform
        self.waveform_arguments = dict(waveform_approximant=waveform_approximant,
            minimum_frequency=f_low,
            reference_frequency=f_ref)

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

        self.update_data(injection_parameters)

    def inject_signals(self, injection_parameters_all, Ninj):
        for i_inj in range(Ninj):
            print(f"Injecting {i_inj}-th signal, {round(100*i_inj/Ninj,2)}% done")
            injection_parameters = {}
            for paraname in self.parameter_names:
                injection_parameters[paraname] = injection_parameters_all[paraname][i_inj]
            self.inject_one_signal(injection_parameters)

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