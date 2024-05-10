import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
#from .utils import PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
from .utils import * 
from .reparameterize import *
#from ..models.utils import project_strain_data_FDAPhi
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


class DatasetSVDStrainFDFromSVDWFonGPU(Dataset):
    '''
    Simulate FD data in SVD space from pre-calculated SVD waveforms, optimized for GPU or CPU computation.
    '''
    def __init__(self, precalwf_filelist, parameter_names, data_generator, Nbasis, Vhfile,
                dmin=10, dmax=200, dpower=1, loadwf=False, loadnoise=False, device='cuda',
                is_complex = False, add_noise=True, fix_extrinsic=False, shuffle=True, reparameterize=True):
        self.precalwf_filelist = precalwf_filelist
        self.parameter_names = parameter_names
        self.data_generator = data_generator
        self.Nbasis = Nbasis
        self.dmin = dmin
        self.dmax = dmax
        self.dpower = dpower
        self.loadwf = loadwf
        self.loadnoise = loadnoise
        self.device = device
        self.complex = is_complex
        self.add_noise = add_noise
        self.fix_extrinsic = fix_extrinsic
        self.shuffle = shuffle
        self.reparameterize = reparameterize
        
        # Load V and Vh matrices and convert to tensors
        self.V, self.Vh = loadVandVh(Vhfile, Nbasis)
        self.V = torch.from_numpy(self.V).to(self.device).type(torch.complex64)
        self.Vh = torch.from_numpy(self.Vh).to(self.device).type(torch.complex64)

        
        self.farray = torch.from_numpy(data_generator.frequency_array_masked).float().to(self.device)
        self.ifos = data_generator.ifos
        self.det_data = self.prepare_detector_data()
        
        testfile = load_dict_from_hdf5(precalwf_filelist[0])
        self.sample_per_file = len(testfile['injection_parameters']['chirp_mass'])
        self.Nfile = len(self.precalwf_filelist)
        self.Nsample = self.Nfile * self.sample_per_file 
        self.cached_wf_file = testfile
        self.cached_wf_file_index = 0
        
        self.shuffle_indexinfile()
            
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
        return len(self.precalwf_filelist) * self.sample_per_file

    def __getitem__(self, index):
        index_of_file, index_in_file = self.get_index(index, self.sample_per_file)
        wf_dict = self.get_precalwf_dict(index_of_file)
        hp_svd, hc_svd = self.get_waveform_tensors(wf_dict, index_in_file)

        injection_parameters = self.get_injection_parameters(wf_dict,index_in_file)
        injection_parameters = self.update_injection_parameters(injection_parameters)
        hp_svd = hp_svd/injection_parameters['luminosity_distance']
        hc_svd = hc_svd/injection_parameters['luminosity_distance']
        #x_real, x_imag = self.compute_strain_tensors(hp_svd, hc_svd, injection_parameters)
        x = self.compute_strain_tensors(hp_svd, hc_svd, injection_parameters)

        theta = self.get_theta(injection_parameters)
        if self.complex:
            return theta, x 
        else:
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
        hp_svd = (torch.from_numpy(wf_dict['waveform_polarizations']['plus']['amplitude'][index_in_file]) *\
            torch.exp(1j*torch.from_numpy(wf_dict['waveform_polarizations']['plus']['phase'][index_in_file])).type(torch.complex64)).to(self.device)
        hc_svd = (torch.from_numpy(wf_dict['waveform_polarizations']['cross']['amplitude'][index_in_file]) *\
            torch.exp(1j*torch.from_numpy(wf_dict['waveform_polarizations']['cross']['phase'][index_in_file])).type(torch.complex64)).to(self.device)
        
        return hp_svd, hc_svd

    def get_injection_parameters(self, wf_dict, index_in_file):
        index_in_file = self.random_index_in_file[index_in_file]
        injection_parameters = {key: wf_dict['injection_parameters'][key][index_in_file] for key in ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                    'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'phase']}
        return injection_parameters

    def get_noise_tensors(self, ):
        white_noise = (torch.randn(self.Nbasis, device=self.device) + 1j * torch.randn(self.Nbasis, device=self.device)).type(torch.complex64)

        return white_noise
    
    def compute_strain_tensors(self, hp_svd, hc_svd, injection_parameters):
        num_ifos = len(self.ifos)
        #x_real = torch.zeros((num_ifos, self.Nbasis), dtype=torch.float32, device=self.device)
        #x_imag = torch.zeros((num_ifos, self.Nbasis), dtype=torch.float32, device=self.device)
        x = torch.zeros((num_ifos, self.Nbasis), dtype=torch.complex64, device=self.device)
        for i, det in enumerate(self.ifos):
            detname = det.name

            fp, fc, dt = self.compute_detector_factors(det, injection_parameters)
            phase2add = torch.exp(-1j * 2 * np.pi * dt * self.farray)
            Vh_recons = self.Vh * phase2add.unsqueeze(0)  # Ensure proper broadcasting
            
            h_svd = torch.matmul(torch.matmul((fp*hp_svd + fc*hc_svd).type(torch.complex64), Vh_recons),
                                 self.det_data[detname]['whitened_V'])

            if self.add_noise:
                n_svd = self.get_noise_tensors()
                d_svd = h_svd + n_svd
            else:
                d_svd = h_svd
            
            #x_real[i] = d_svd.real
            #x_imag[i] = d_svd.imag
            x[i] = d_svd
        #return x_real, x_imag
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
            injection_parameters['ra'] = 1#np.random.uniform(0, 2*np.pi) #1
            injection_parameters['dec'] = 1#np.arcsin(np.random.uniform(-1, 1)) #1
            injection_parameters['psi'] = 1#np.random.uniform(0, np.pi) #1
            injection_parameters['geocent_time'] = 0
            #injection_parameters['luminosity_distance'] = generate_random_distance(Nsample=1, low=self.dmin, high=self.dmax, power=self.dpower)[0]
            injection_parameters['luminosity_distance'] = 100
        else:
            injection_parameters['ra'] = np.random.uniform(0, 2*np.pi)
            injection_parameters['dec'] = np.arcsin(np.random.uniform(-1, 1))
            injection_parameters['psi'] = np.random.uniform(0, np.pi)
            injection_parameters['geocent_time'] = np.random.uniform(-0.01, 0.01)
            injection_parameters['luminosity_distance'] = generate_random_distance(Nsample=1, low=self.dmin, high=self.dmax, power=self.dpower)[0]
    
        return injection_parameters
    
    def shuffle_wflist(self):
        #if self.shuffle:
        random.shuffle(self.precalwf_filelist)
        
    def shuffle_indexinfile(self):
        if self.shuffle:
            self.random_index_in_file = np.random.permutation(self.sample_per_file)
        else:
            self.random_index_in_file = np.arange(self.sample_per_file)

class DatasetSVDStrainFDFromSVDWFonGPUBatch(Dataset):
    '''
    Simulate FD data in SVD space from pre-calculated SVD waveforms, optimized for GPU or CPU computation.

    Load a batch of data, i.e. return [minibatch_size, dim1, dim2, ...]. The batch size should be 2^N. 
    '''
    def __init__(self, precalwf_filelist, parameter_names, data_generator, Nbasis, Vhfile,
                dmin=10, dmax=200, dpower=1, loadwf=False, loadnoise=False, device='cuda',
                is_complex=False, add_noise=True, minibatch_size=1, fix_extrinsic=False, shuffle=True, reparameterize=True):
        self.precalwf_filelist = precalwf_filelist
        self.parameter_names = parameter_names
        self.data_generator = data_generator
        self.Nbasis = Nbasis
        self.dmin = dmin
        self.dmax = dmax
        self.dpower = dpower
        self.loadwf = loadwf
        self.loadnoise = loadnoise
        self.device = device
        self.minibatch_size = minibatch_size
        self.complex = is_complex
        self.add_noise = add_noise
        self.fix_extrinsic = fix_extrinsic
        self.shuffle = shuffle
        self.reparameterize = reparameterize

        # Load V and Vh matrices and convert to tensors
        self.V, self.Vh = loadVandVh(Vhfile, Nbasis)
        self.V = torch.from_numpy(self.V).to(self.device).type(torch.complex64)
        self.Vh = torch.from_numpy(self.Vh).to(self.device).type(torch.complex64)

        
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
        
        hp_svd, hc_svd = self.get_waveform_tensors_batch(wf_dict_list, index_in_file, index_in_file_end)
        injection_parameters = self.get_injection_parameters_batch(wf_dict_list,index_in_file, index_in_file_end)
        injection_parameters = self.update_injection_parameters_batch(injection_parameters)
        
        dL = torch.from_numpy(injection_parameters['luminosity_distance']).to(self.device).unsqueeze(-1)
        hp_svd = hp_svd/dL
        hc_svd = hc_svd/dL

        #x_real, x_imag = self.compute_strain_tensors_batch(hp_svd, hc_svd, injection_parameters)
        x = self.compute_strain_tensors_batch(hp_svd, hc_svd, injection_parameters)

        theta = self.get_theta(injection_parameters)
        if self.complex:
            return theta, x 
        else:
            return theta, torch.cat((x.real, x.imag), axis=1).float()

    def get_index(self, index, sample_per_file):
        index_of_file = index // sample_per_file
        index_in_file = index - index_of_file*sample_per_file

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
        
    def get_waveform_tensors_batch(self, wf_dict_list, index_in_file, index_in_file_end):
        for i, wf_dict in enumerate(wf_dict_list):
            if i==len(wf_dict_list)-1:
                end_index = index_in_file_end
            else:
                end_index = self.sample_per_file
            if i==0:
                index = self.random_index_in_file[index_in_file:end_index]
                hp_svd = (torch.from_numpy(wf_dict['waveform_polarizations']['plus']['amplitude'][index]) *\
                    torch.exp(1j*torch.from_numpy(wf_dict['waveform_polarizations']['plus']['phase'][index])).type(torch.complex64)).to(self.device)
                hc_svd = (torch.from_numpy(wf_dict['waveform_polarizations']['cross']['amplitude'][index]) *\
                    torch.exp(1j*torch.from_numpy(wf_dict['waveform_polarizations']['cross']['phase'][index])).type(torch.complex64)).to(self.device)
            else:
                index = self.random_index_in_file[index_in_file:end_index]
                hp_svd_new = (torch.from_numpy(wf_dict['waveform_polarizations']['plus']['amplitude'][index]) *\
                    torch.exp(1j*torch.from_numpy(wf_dict['waveform_polarizations']['plus']['phase'][index])).type(torch.complex64)).to(self.device)
                hc_svd_new = (torch.from_numpy(wf_dict['waveform_polarizations']['cross']['amplitude'][index]) *\
                    torch.exp(1j*torch.from_numpy(wf_dict['waveform_polarizations']['cross']['phase'][index])).type(torch.complex64)).to(self.device)

                hp_svd = torch.cat((hp_svd,hp_svd_new))
                hc_svd = torch.cat((hc_svd,hc_svd_new))
                    
            
        return hp_svd, hc_svd

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
        white_noise = (torch.randn((self.minibatch_size, self.Nbasis), device=self.device) + \
                       1j * torch.randn((self.minibatch_size, self.Nbasis), device=self.device)).type(torch.complex64)

        return white_noise
    
    def compute_strain_tensors_batch(self, hp_svd, hc_svd, injection_parameters):
        num_ifos = len(self.ifos)
        #x_real = torch.zeros((self.minibatch_size, num_ifos, self.Nbasis), dtype=torch.float32, device=self.device)
        #x_imag = torch.zeros((self.minibatch_size, num_ifos, self.Nbasis), dtype=torch.float32, device=self.device)
        x = torch.zeros((self.minibatch_size, num_ifos, self.Nbasis), dtype=torch.complex64, device=self.device)
        for i,det in enumerate(self.ifos):
            detname = det.name
        
            fp, fc, dt = self.compute_detector_factors_batch(det, injection_parameters)
            phase2add = torch.exp(-1j * 2 * np.pi * dt * self.farray)
            Vh_recons = (self.Vh * phase2add.unsqueeze(0)).type(torch.complex64)  # Ensure proper broadcasting            
            hh = (fp*hp_svd + fc*hc_svd).type(torch.complex64)

            #h_svd = torch.matmul(torch.bmm(hh.unsqueeze(1), Vh_recons).squeeze(1),
            #                     self.det_data[detname]['whitened_V'])
            h_svd = torch.matmul(torch.matmul(hh, Vh_recons),
                                 self.det_data[detname]['whitened_V'])
            
            
            if self.add_noise:
                n_svd = self.get_noise_tensors_batch()
                d_svd = h_svd + n_svd
            else:
                d_svd = h_svd
            
            #x_real[:,i,:] = d_svd.real
            #x_imag[:,i,:] = d_svd.imag
            x[:,i,:] = d_svd
            

        return x
    
    def compute_detector_factors_batch(self, det, injection_parameters):
        # These calculations remain on CPU as they cannot be efficiently vectorized or moved to GPU
        #fp_tensor = torch.zeros((self.minibatch_size), dtype=torch.float32, device=self.device)
        #fc_tensor = torch.zeros((self.minibatch_size), dtype=torch.float32, device=self.device)
        #dt_tensor = torch.zeros((self.minibatch_size), dtype=torch.float32, device=self.device)
        '''
        for i in range(len(injection_parameters['ra'])):
            
            ra = injection_parameters['ra'][i]
            dec = injection_parameters['dec'][i]
            tc = injection_parameters['geocent_time'][i]
            psi = injection_parameters['psi'][i]
            
            fp = det.antenna_response(ra, dec, tc, psi, 'plus')
            fc = det.antenna_response(ra, dec, tc, psi, 'cross')
            time_shift = det.time_delay_from_geocenter(ra, dec, tc)
            
            dt_geocent = tc #- self.strain_data.start_time
            dt = dt_geocent + time_shift
            
            fp_tensor[i] = fp
            fc_tensor[i] = fc
            dt_tensor[i] = dt
        '''
        ra = injection_parameters['ra'][0]
        dec = injection_parameters['dec'][0]
        tc = injection_parameters['geocent_time'][0]
        psi = injection_parameters['psi'][0]

        fp = det.antenna_response(ra, dec, tc, psi, 'plus')
        fc = det.antenna_response(ra, dec, tc, psi, 'cross')
        time_shift = det.time_delay_from_geocenter(ra, dec, tc)

        dt_geocent = tc #- self.strain_data.start_time
        dt = dt_geocent + time_shift
        
        #return fp_tensor.unsqueeze(-1), fc_tensor.unsqueeze(-1), dt_tensor.unsqueeze(-1)
        return fp, fc, dt
    
    '''
    def get_theta(self, injection_parameters):
        if self.fix_extrinsic:
            reduced_parameter_names = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                    'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'phase']
            theta = torch.tensor(np.array([injection_parameters[paraname] for paraname in reduced_parameter_names]), dtype=torch.float32).to(self.device).T
        else:
            theta = torch.tensor(np.array([injection_parameters[paraname] for paraname in self.parameter_names]), dtype=torch.float32).to(self.device).T
        return theta
    '''

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
            #injection_parameters['luminosity_distance'] = generate_random_distance(Nsample=self.minibatch_size, low=self.dmin, high=self.dmax, power=self.dpower)
            injection_parameters['luminosity_distance'] = np.zeros(self.minibatch_size) + 100
    
        else:
            injection_parameters['ra'] = np.zeros(self.minibatch_size) + np.random.uniform(0, 2*np.pi)
            injection_parameters['dec'] = np.zeros(self.minibatch_size) + np.arcsin(np.random.uniform(-1, 1))
            injection_parameters['psi'] = np.zeros(self.minibatch_size) + np.random.uniform(0, np.pi)
            injection_parameters['geocent_time'] = np.zeros(self.minibatch_size) + np.random.uniform(-0.01, 0.01)
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

