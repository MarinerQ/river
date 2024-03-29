import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#from .utils import PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
from .utils import * 
from .reparameterize import *
import pickle
import random
import glob
import sealgw.simulation as sealsim

class DatasetMBStrainFDFromMBWFonGPU(Dataset):
    '''
    Simulate multiband FD data from pre-calculated multiband waveforms, optimized for GPU or CPU computation.
    '''
    def __init__(self, wf_folder, asd_folder, parameter_names, full_duration, detector_names, device='cuda',
                dmin=10, dmax=200, dpower=1,  timing_std = 0.01, tc_min=-1, tc_max=1,
                add_noise=True, fix_extrinsic=False, shuffle=False, reparameterize=True,
                use_sealgw_detector=False, is_asd=True, random_asd = False):
        
        self.precalwf_filelist = glob.glob(f"{wf_folder}/batch*/*.h5")
        self.parameter_names = parameter_names
        self.full_duration = full_duration
        
        self.dmin = dmin
        self.dmax = dmax
        self.dpower = dpower
        self.device = device
        self.add_noise = add_noise
        self.fix_extrinsic = fix_extrinsic
        self.shuffle = shuffle
        self.reparameterize = reparameterize
        self.random_asd = random_asd
        self.timing_std = timing_std
        self.tc_min = tc_min
        self.tc_max = tc_max
        
        self.detector_names = detector_names
        if use_sealgw_detector:
            self.ifos = sealsim.sealinterferometers.SealInterferometerList(detector_names)
        else:
            self.ifos = bilby.gw.detector.InterferometerList(detector_names)
            
        #for det in self.ifos:
        #    det.sampling_frequency = 2048
        #    det.duration = self.full_duration
            #det.frequency_mask = (det.frequency_array >= self.f_low) * (det.frequency_array <= self.f_high)
        #    if use_sealgw_detector:
        #        det.antenna_response_change = True
        
        
        #self.det_data = self.prepare_detector_data()
        
        testfile = load_dict_from_hdf5(self.precalwf_filelist[0])
        self.sample_per_file = len(testfile['injection_parameters']['chirp_mass'])
        self.Nfile = len(self.precalwf_filelist)
        self.Nsample = self.Nfile * self.sample_per_file 
        self.cached_wf_file = testfile
        self.cached_wf_file_index = 0
        self.farray = torch.from_numpy(testfile['frequency_array']).float().to(self.device)
        self.Npoint = len(self.farray)
        self.asd_dict = load_asd(asd_folder, detector_names, freqs_interp=testfile['frequency_array'], is_asd=is_asd)
        
        self.shuffle_indexinfile()
    '''
    def prepare_detector_data(self):
        det_data = {}
        for det in self.ifos:
            detname = det.name
            psd = det.power_spectral_density_array[self.data_generator.frequency_mask]
            psd = torch.from_numpy(psd).double().to(self.device)
            whitened_V = (self.V.T * 1/(psd*det.duration/4)**0.5).T
            det_data[detname] = {'whitened_V': whitened_V.type(torch.complex64)}
        return det_data'''

    def __len__(self):
        return len(self.precalwf_filelist) * self.sample_per_file

    def __getitem__(self, index):
        index_of_file, index_in_file = self.get_index(index, self.sample_per_file)
        wf_dict = self.get_precalwf_dict(index_of_file)
        hp_mb, hc_mb = self.get_waveform_tensors(wf_dict, index_in_file)

        injection_parameters = self.get_injection_parameters(wf_dict,index_in_file)
        injection_parameters = self.update_injection_parameters(injection_parameters)
        hp_mb = hp_mb/injection_parameters['luminosity_distance']
        hc_mb = hc_mb/injection_parameters['luminosity_distance']
        x = self.compute_strain_tensors(hp_mb, hc_mb, injection_parameters)

        theta = self.get_theta(injection_parameters)
        return theta, torch.cat((x.real, x.imag)).float()

    def get_index(self, index, sample_per_file):
        index_of_file = index // sample_per_file
        index_in_file = index - index_of_file*sample_per_file
        
        return index_of_file, index_in_file
    
    def get_precalwf_dict(self, index_of_file):
        if self.cached_wf_file_index == index_of_file:
            return self.cached_wf_file
        else:
            wf_dict = load_dict_from_hdf5(self.precalwf_filelist[index_of_file])
            self.cached_wf_file = wf_dict
            self.cached_wf_file_index = index_of_file
            return wf_dict
        
    def get_waveform_tensors(self, wf_dict, index_in_file):
        index_in_file = self.random_index_in_file[index_in_file]
        hp = (torch.from_numpy(wf_dict['waveforms'][2*index_in_file]).type(torch.complex64)).to(self.device)
        hc = (torch.from_numpy(wf_dict['waveforms'][2*index_in_file+1]).type(torch.complex64)).to(self.device)
        return hp, hc

    def get_injection_parameters(self, wf_dict, index_in_file):
        index_in_file = self.random_index_in_file[index_in_file]
        injection_parameters = {key: wf_dict['injection_parameters'][key][index_in_file] for key in ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                    'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'phase']}
        return injection_parameters

    def get_noise_tensors(self, ):
        white_noise = (torch.randn(self.Npoint, device=self.device) + 1j * torch.randn(self.Npoint, device=self.device)).type(torch.complex64)

        return white_noise
    
    def compute_strain_tensors(self, hp_mb, hc_mb, injection_parameters):
        num_ifos = len(self.ifos)
        x = torch.zeros((num_ifos, self.Npoint), dtype=torch.complex64, device=self.device)
        tc_est = injection_parameters['geocent_time'] + injection_parameters['timing_error']
        for i, det in enumerate(self.ifos):
            detname = det.name
            
            fp, fc, dt = self.compute_detector_factors(det, injection_parameters)
            phase2add = torch.exp(-1j * 2 * np.pi * (dt-tc_est) * self.farray)            
            h_mb = (fp*hp_mb + fc*hc_mb).type(torch.complex64) * phase2add
            if self.random_asd:
                whiten_factor = 1/random.choice(self.asd_dict[detname]) * 1/(self.full_duration/4)**0.5
            else:
                whiten_factor = 1/self.asd_dict[detname][0] * 1/(self.full_duration/4)**0.5
                
            whiten_factor = torch.from_numpy(whiten_factor).to(self.device)
            h_mb_whitened = h_mb * whiten_factor
            if self.add_noise:
                n_mb = self.get_noise_tensors()
                d_mb = h_mb_whitened + n_mb
            else:
                d_mb = h_mb_whitened
            
            x[i] = d_mb
        return x

    def compute_detector_factors(self, det, injection_parameters):
        # These calculations remain on CPU as they cannot be efficiently vectorized or moved to GPU
        ra = injection_parameters['ra']
        dec = injection_parameters['dec']
        tc = injection_parameters['geocent_time']
        psi = injection_parameters['psi']
        fp = det.antenna_response(ra , dec, tc, psi, 'plus')
        fc = det.antenna_response(ra , dec, tc, psi, 'cross')
        time_shift = det.time_delay_from_geocenter(ra , dec, tc)
        
        dt_geocent = tc #- self.strain_data.start_time
        dt = dt_geocent + time_shift
            
        return fp, fc, dt

    def get_theta(self, injection_parameters):
        if self.fix_extrinsic:
            parameter_names = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                    'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'phase']
        else:
            parameter_names = self.parameter_names
        
        theta_list = []
        for paraname in parameter_names:
            if paraname == 'geocent_time':
                paraname = 'timing_error'
            if self.reparameterize:
                para_re = reparameterize(injection_parameters[paraname], paraname)
            else:
                para_re = injection_parameters[paraname]
            theta_list.append(para_re)
        theta_array = np.array(theta_list)
        theta = torch.tensor(theta_array, dtype=torch.float32).to(self.device).T
    
        return theta
    
    def update_injection_parameters(self, injection_parameters):
        if self.fix_extrinsic:
            injection_parameters['ra'] = 1
            injection_parameters['dec'] = 1
            injection_parameters['psi'] = 1
            injection_parameters['geocent_time'] = 0
            injection_parameters['luminosity_distance'] = 100
            injection_parameters['timing_error'] = 0
        else:
            injection_parameters['ra'] = np.random.uniform(0, 2*np.pi)
            injection_parameters['dec'] = np.arcsin(np.random.uniform(-1, 1))
            injection_parameters['psi'] = np.random.uniform(0, np.pi)
            injection_parameters['geocent_time'] = np.random.uniform(self.tc_min, self.tc_max)
            injection_parameters['luminosity_distance'] = generate_random_distance(Nsample=1, low=self.dmin, high=self.dmax, power=self.dpower)[0]
            injection_parameters['timing_error'] = np.random.randn() * self.timing_std

        return injection_parameters
    
    def shuffle_wflist(self):
        #if self.shuffle:
        random.shuffle(self.precalwf_filelist)
        
    def shuffle_indexinfile(self):
        if self.shuffle:
            raise Exception('Not implemeted')
            #self.random_index_in_file = np.random.permutation(self.sample_per_file)
        else:
            self.random_index_in_file = np.arange(self.sample_per_file)


class DatasetMBStrainFDFromMBWFonGPUBatch(DatasetMBStrainFDFromMBWFonGPU):
    '''
    Simulate multiband FD data from pre-calculated multiband waveforms, optimized for GPU or CPU computation.
    '''
    def __init__(self, wf_folder, asd_folder, parameter_names, full_duration, detector_names,
                dmin=10, dmax=200, dpower=1, device='cuda', minibatch_size=1, timing_std = 0.01, tc_min=-1, tc_max=1,
                add_noise=True, fix_extrinsic=False, shuffle=False, reparameterize=True,
                 use_sealgw_detector=False, is_asd=True, random_asd = False):
        super().__init__(wf_folder=wf_folder,
                        asd_folder=asd_folder,
                        parameter_names = parameter_names, 
                        full_duration = full_duration, 
                        detector_names = detector_names, 
                        timing_std = timing_std,
                        tc_min = tc_min,
                        tc_max = tc_max,
                        dmin = dmin, 
                        dmax = dmax, 
                        dpower = dpower, 
                        device = device,
                        add_noise = add_noise,
                        fix_extrinsic =  fix_extrinsic, 
                        shuffle = shuffle, 
                        reparameterize = reparameterize, 
                        use_sealgw_detector = use_sealgw_detector, 
                        is_asd = is_asd, 
                        random_asd = random_asd)
        self.minibatch_size = minibatch_size
        
    def __len__(self):
        return len(self.precalwf_filelist) * self.sample_per_file // self.minibatch_size
    
    def __getitem__(self, index):
        index = index*self.minibatch_size
        
        index_end = index + self.minibatch_size
        index_of_file, index_in_file = self.get_index(index, self.sample_per_file)
        index_of_file_end, index_in_file_end = self.get_index(index_end, self.sample_per_file)
        if index_of_file_end>=len(self.precalwf_filelist):
            index_of_file_end = len(self.precalwf_filelist)-1
            index_in_file_end = self.sample_per_file
        wf_dict_list = []
        for i in range(index_of_file, index_of_file_end+1):
            wf_dict_list.append(self.get_precalwf_dict(i))
        
        hp_mb, hc_mb = self.get_waveform_tensors_batch(wf_dict_list, index_in_file, index_in_file_end)
        injection_parameters = self.get_injection_parameters_batch(wf_dict_list,index_in_file, index_in_file_end)
        injection_parameters = self.update_injection_parameters_batch(injection_parameters)
        
        dL = torch.from_numpy(injection_parameters['luminosity_distance']).to(self.device).unsqueeze(-1)
        hp_mb = hp_mb/dL
        hc_mb = hc_mb/dL

        x = self.compute_strain_tensors_batch(hp_mb, hc_mb, injection_parameters)

        theta = self.get_theta(injection_parameters)

        return theta, torch.cat((x.real, x.imag), axis=1).float()
    
    def get_precalwf_dict(self, index_of_file):
        if self.cached_wf_file_index == index_of_file:
            return self.cached_wf_file
        else:
            try:
                wf_dict = load_dict_from_hdf5(self.precalwf_filelist[index_of_file])
            except:
                raise Exception(f'index_of_file: {index_of_file}')
            self.cached_wf_file = wf_dict
            self.cached_wf_file_index = index_of_file
        return wf_dict
    
    def get_waveform_tensors_batch(self, wf_dict_list, index_in_file, index_in_file_end):
        for i, wf_dict in enumerate(wf_dict_list):
            if i==len(wf_dict_list)-1:
                end_index = index_in_file_end
            else:
                end_index = self.sample_per_file
                
            index = self.random_index_in_file[index_in_file:end_index]
            index_p = 2*index
            index_c = 2*index + 1
            if i==0:
                hp = (torch.from_numpy(wf_dict['waveforms'][index_p]).type(torch.complex64)).to(self.device)
                hc = (torch.from_numpy(wf_dict['waveforms'][index_c]).type(torch.complex64)).to(self.device)
        
            else:
                hp_new = (torch.from_numpy(wf_dict['waveforms'][index_p]).type(torch.complex64)).to(self.device)
                hc_new = (torch.from_numpy(wf_dict['waveforms'][index_c]).type(torch.complex64)).to(self.device)
                
                hp = torch.cat((hp,hp_new))
                hc = torch.cat((hc,hc_new))
                    
        return hp, hc    
    
    def get_injection_parameters_batch(self, wf_dict_list, index_in_file, index_in_file_end):
        para_name_list = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                    'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'phase']
        for i, wf_dict in enumerate(wf_dict_list):
            if i==len(wf_dict_list)-1:
                end_index = index_in_file_end
            else:
                end_index = self.sample_per_file

            index_random = self.random_index_in_file[index_in_file:end_index]
            if i==0:
                injection_parameters = {key: wf_dict['injection_parameters'][key][index_random] for key in para_name_list}
            else:
                injection_parameters = {key: np.append(injection_parameters[key], wf_dict['injection_parameters'][key][index_random]) for key in para_name_list}

        return injection_parameters
    
    def get_noise_tensors_batch(self, ):
        white_noise = (torch.randn((self.minibatch_size, self.Npoint), device=self.device) + \
                       1j * torch.randn((self.minibatch_size, self.Npoint), device=self.device)).type(torch.complex64)

        return white_noise
    
    
    def compute_strain_tensors_batch(self, hp_mb, hc_mb, injection_parameters):
        num_ifos = len(self.ifos)
        x = torch.zeros((self.minibatch_size, num_ifos, self.Npoint), dtype=torch.complex64, device=self.device)
        tc_est = injection_parameters['geocent_time'][0] + injection_parameters['timing_error'][0]
        for i,det in enumerate(self.ifos):
            detname = det.name
        
            fp, fc, dt = self.compute_detector_factors_batch(det, injection_parameters)
            phase2add = torch.exp(-1j * 2 * np.pi * (dt-tc_est) * self.farray)

            h_mb = (fp*hp_mb + fc*hc_mb).type(torch.complex64) * phase2add#.unsqueeze(0)
            
            if self.random_asd:
                whiten_factor = 1/random.choice(self.asd_dict[detname]) * 1/(self.full_duration/4)**0.5
            else:
                whiten_factor = 1/self.asd_dict[detname][0] * 1/(self.full_duration/4)**0.5
                
            whiten_factor = torch.from_numpy(whiten_factor).to(self.device)
            h_mb_whitened = h_mb * whiten_factor
            
            if self.add_noise:
                n_mb = self.get_noise_tensors_batch()
                d_mb = h_mb_whitened + n_mb
            else:
                d_mb = h_mb_whitened
            
            x[:,i,:] = d_mb
            
        return x
        
        
    def compute_detector_factors_batch(self, det, injection_parameters):
        ra = injection_parameters['ra'][0]
        dec = injection_parameters['dec'][0]
        tc = injection_parameters['geocent_time'][0]
        psi = injection_parameters['psi'][0]

        fp = det.antenna_response(ra, dec, tc, psi, 'plus')
        fc = det.antenna_response(ra, dec, tc, psi, 'cross')
        time_shift = det.time_delay_from_geocenter(ra, dec, tc)

        dt_geocent = tc #- self.strain_data.start_time
        dt = dt_geocent + time_shift
        
        return fp, fc, dt
    
    def update_injection_parameters_batch(self, injection_parameters):
        if self.fix_extrinsic:
            injection_parameters['ra'] = np.zeros(self.minibatch_size) + 1
            injection_parameters['dec'] = np.zeros(self.minibatch_size) + 1
            injection_parameters['psi'] = np.zeros(self.minibatch_size) + 1
            injection_parameters['geocent_time'] = np.zeros(self.minibatch_size) + 0
            injection_parameters['luminosity_distance'] = np.zeros(self.minibatch_size) + 100
            injection_parameters['timing_error'] = np.zeros(self.minibatch_size) + 0
    
        else:
            injection_parameters['ra'] = np.zeros(self.minibatch_size) + np.random.uniform(0, 2*np.pi)
            injection_parameters['dec'] = np.zeros(self.minibatch_size) + np.arcsin(np.random.uniform(-1, 1))
            injection_parameters['psi'] = np.zeros(self.minibatch_size) + np.random.uniform(0, np.pi)
            injection_parameters['geocent_time'] = np.zeros(self.minibatch_size) + np.random.uniform(self.tc_min, self.tc_max)
            injection_parameters['luminosity_distance'] = generate_random_distance(Nsample=self.minibatch_size, low=self.dmin, high=self.dmax, power=self.dpower)
            injection_parameters['timing_error'] = np.zeros(self.minibatch_size) + np.random.randn() * self.timing_std

        return injection_parameters