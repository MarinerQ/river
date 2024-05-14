import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
#from .utils import PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
from .utils import * 
from .reparameterize import *
from .antenna import *
import pickle
import random



def loadVandVh(Vhfilepath, Nbasis):
    with open(Vhfilepath, 'rb') as f:
        Vh = pickle.load(f)
    if len(Vh)<Nbasis:
        raise ValueError(f'required Nbasis ({Nbasis}) > len(Vh) ({len(Vh)})!')
    Vh = Vh[:Nbasis]
    V = Vh.T.conj()
        
    return V, Vh

class DatasetSVDStrainFDFromSVDWFonGPUBatch(Dataset):
    '''
    Simulate FD data in SVD space from pre-calculated SVD waveforms, optimized for GPU or CPU computation.

    Load a batch of data, i.e. return [minibatch_size, dim1, dim2, ...]. The batch size should be 2^N. 
    '''
    def __init__(self, precalwf_filelist, parameter_names, data_generator, Nbasis, Vhfile,
                dmin=10, dmax=200, dpower=1, device='cuda',
                add_noise=True, minibatch_size=1, fix_extrinsic=False, shuffle=True, reparameterize=True):
        self.precalwf_filelist = precalwf_filelist
        self.parameter_names = parameter_names
        self.data_generator = data_generator
        self.Nbasis = Nbasis
        self.dmin = dmin
        self.dmax = dmax
        self.dpower = dpower
        self.device = device
        self.minibatch_size = minibatch_size
        self.add_noise = add_noise
        self.fix_extrinsic = fix_extrinsic
        self.shuffle = shuffle
        self.reparameterize = reparameterize

        # Load V and Vh matrices and convert to tensors
        self.V, self.Vh = loadVandVh(Vhfile, Nbasis)
        self.V = torch.from_numpy(self.V).to(self.device).type(torch.complex64)
        self.Vh = torch.from_numpy(self.Vh).to(self.device).type(torch.complex64)

        self.antenna = GWAntennaOnCPU(data_generator.detector_names)
        self.farray = torch.from_numpy(data_generator.frequency_array_masked).float().to(self.device)
        self.ifos = data_generator.ifos
        self.det_data = self.prepare_detector_data()
        
        testfile = load_dict_from_hdf5(precalwf_filelist[0])
        self.sample_per_file = len(testfile['injection_parameters']['chirp_mass'])
        #if self.sample_per_file<self.minibatch_size:
        #    raise ValueError("Sample per file < batch size!")
        self.Nfile = len(self.precalwf_filelist)
        self.Nsample = self.Nfile * self.sample_per_file 
        self.cached_wf_file = testfile
        self.cached_wf_file_index = 0
            
        self.shuffle_indexinfile()
        
        assert self.sample_per_file % self.minibatch_size == 0
        
    def prepare_detector_data(self):
        det_data = {}
        for det in self.ifos:
            detname = det.name
            psd = det.power_spectral_density_array[self.data_generator.frequency_mask]
            psd = torch.from_numpy(psd).double().to(self.device)
            whitened_V = (self.V.T * 1/(psd*det.duration/4)**0.5).T
            det_data[detname] = {'whitened_V': whitened_V.type(torch.complex64)}
        return det_data

    def __len__(self):
        return len(self.precalwf_filelist) * self.sample_per_file // self.minibatch_size

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError("Index out of range")
        
        
        index = index*self.minibatch_size
        index_end = index + self.minibatch_size
        index_of_file, index_in_file = self.get_index(index, self.sample_per_file)
        index_of_file_end, index_in_file_end = self.get_index(index_end, self.sample_per_file, is_end=True)
        
        assert index_of_file == index_of_file_end
        assert index_of_file < len(self.precalwf_filelist)

        
        wf_dict = self.get_precalwf_dict(index_of_file)
        hp_svd, hc_svd, injection_parameters = self.get_waveform_tensors_batch(wf_dict, index_in_file, index_in_file_end)
        injection_parameters = self.update_injection_parameters_batch(injection_parameters)
        
        dL = torch.from_numpy(injection_parameters['luminosity_distance']).to(self.device).unsqueeze(-1)
        hp_svd = hp_svd/dL
        hc_svd = hc_svd/dL

        x = self.compute_strain_tensors_batch(hp_svd, hc_svd, injection_parameters)
        theta = self.get_theta(injection_parameters)
        return theta, torch.cat((x.real, x.imag), axis=1).float()

    def get_index(self, index, sample_per_file, is_end=False):
        index_of_file = index // sample_per_file
        index_in_file = index - index_of_file*sample_per_file
        
        if is_end and index_in_file==0:
            index_in_file = None
            index_of_file -= 1
        return index_of_file, index_in_file
    
    
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
    
    
    def get_waveform_tensors_batch(self, wf_dict, index_in_file, index_in_file_end):
        index = self.random_index_in_file[index_in_file:index_in_file_end]
        hp_svd = (torch.from_numpy(wf_dict['waveform_polarizations']['plus'][index])).type(torch.complex64).to(self.device)
        hc_svd = (torch.from_numpy(wf_dict['waveform_polarizations']['cross'][index])).type(torch.complex64).to(self.device)
        
        para_name_list = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                    'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'phase']
        injection_parameters = injection_parameters = {key: wf_dict['injection_parameters'][key][index] for key in para_name_list}
        
        # last wf in this file
        if (index_in_file_end is None) and (self.cached_wf_file_index < len(self.precalwf_filelist)-1): 
            _ = self.get_precalwf_dict(self.cached_wf_file_index + 1)
            print('Automatically updated cached wf.')

        return hp_svd, hc_svd, injection_parameters

    def get_noise_tensors_batch(self, ):
        white_noise = (torch.randn((self.minibatch_size, self.Nbasis), device=self.device) + \
                       1j * torch.randn((self.minibatch_size, self.Nbasis), device=self.device)).type(torch.complex64)

        return white_noise
    
    def compute_strain_tensors_batch(self, hp_svd, hc_svd, injection_parameters):
        num_ifos = len(self.ifos)
        x = torch.zeros((self.minibatch_size, num_ifos, self.Nbasis), dtype=torch.complex64, device=self.device)
        resp_dt_dict = self.compute_detector_factors_batch(injection_parameters)
        for i,det in enumerate(self.ifos):
            detname = det.name

            fp, fc, dt = resp_dt_dict[detname]
            fp = torch.from_numpy(fp).to(self.device).unsqueeze(1)
            fc = torch.from_numpy(fc).to(self.device).unsqueeze(1)
            dt = torch.from_numpy(dt).to(self.device).unsqueeze(1)

            phase2add = torch.exp(-1j * 2 * np.pi * dt * self.farray).type(torch.complex64)
            Vh_recons = self.Vh
            hh = (fp*hp_svd + fc*hc_svd).type(torch.complex64)

            hh_reconstruct = torch.einsum('ij,jk->ik', hh, Vh_recons)

            hh_shifted = hh_reconstruct * phase2add
            h_svd = torch.einsum('ij,jk->ik', hh_shifted, self.det_data[detname]['whitened_V'])

            if self.add_noise:
                n_svd = self.get_noise_tensors_batch()
                d_svd = h_svd + n_svd
            else:
                d_svd = h_svd

            x[:,i,:] = d_svd
            

        return x
    
    def compute_detector_factors_batch(self, injection_parameters):

        ra = injection_parameters['ra']
        dec = injection_parameters['dec']
        tc = injection_parameters['geocent_time']
        psi = injection_parameters['psi']

        resp_dt_dict = self.antenna.resp_and_dt(ra,dec,tc,psi)    
        dt_geocent = tc  #- self.strain_data.start_time
        for key,value in resp_dt_dict.items():
            value[2] += dt_geocent        

        return resp_dt_dict

    def get_theta(self, injection_parameters):
        if self.fix_extrinsic:
            parameter_names = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                    'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'phase']
        else:
            parameter_names = self.parameter_names
        
        theta_list = []
        for paraname in parameter_names:
            if self.reparameterize:
                para_re = reparameterize(injection_parameters[paraname], paraname)
            else:
                para_re = injection_parameters[paraname]
            theta_list.append(para_re)
        theta_array = np.array(theta_list)
        theta = torch.tensor(theta_array, dtype=torch.float32).to(self.device).T
    
        return theta
    
    def update_injection_parameters_batch(self, injection_parameters):
        if self.fix_extrinsic:
            injection_parameters['ra'] = np.zeros(self.minibatch_size) + 1#np.random.uniform(0, 2*np.pi) #1
            injection_parameters['dec'] = np.zeros(self.minibatch_size) + 1#np.arcsin(np.random.uniform(-1, 1)) #1
            injection_parameters['psi'] = np.zeros(self.minibatch_size) + 1#np.random.uniform(0, np.pi) #1
            injection_parameters['geocent_time'] = np.zeros(self.minibatch_size) 
            injection_parameters['luminosity_distance'] = np.zeros(self.minibatch_size) + 100
    
        else:
            injection_parameters['ra'] = np.random.uniform(0, 2*np.pi, self.minibatch_size)
            injection_parameters['dec'] = np.arcsin(np.random.uniform(-1, 1, self.minibatch_size))
            injection_parameters['psi'] = np.random.uniform(0, np.pi, self.minibatch_size)
            injection_parameters['geocent_time'] = np.random.uniform(-0.01, 0.01, self.minibatch_size)
            injection_parameters['luminosity_distance'] = generate_random_distance(Nsample=self.minibatch_size, low=self.dmin, high=self.dmax, power=self.dpower)
        
        return injection_parameters
    
    def shuffle_wflist(self):
        #if self.shuffle:
        random.shuffle(self.precalwf_filelist)
        
    def shuffle_indexinfile(self):
        if self.shuffle:
            self.random_index_in_file = np.random.permutation(self.sample_per_file)
        else:
            self.random_index_in_file = np.arange(self.sample_per_file)