import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as transforms
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
        self.det_aux = self.prepare_detector_aux()
        
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
        
    def prepare_detector_aux(self):
        det_aux = {}
        for det in self.ifos:
            detname = det.name
            psd = det.power_spectral_density_array[self.data_generator.frequency_mask]
            psd = torch.from_numpy(psd).double().to(self.device)
            whitened_V = (self.V.T * 1/(psd*det.duration/4)**0.5).T
            det_aux[detname] = {'whitened_V': whitened_V.type(torch.complex64)}
        return det_aux

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
            h_svd = torch.einsum('ij,jk->ik', hh_shifted, self.det_aux[detname]['whitened_V'])

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


class DatasetSVDMBStrainFDFromSVDWFonGPUBatch(Dataset):
    '''
    Simulate FD data in SVD space from pre-calculated SVD waveforms, optimized for GPU or CPU computation.

    Load a batch of data, i.e. return [minibatch_size, dim1, dim2, ...]. The batch size should be 2^N. 
    '''
    def __init__(self, precalwf_filelist, parameter_names, data_generator, Nbasis_wf, Nbasis_det, Vhfile_dict,
                snr_min=10, snr_max=200, tmin=-0.01, tmax=0.01, device='cuda',
                add_noise=True, minibatch_size=1, fix_extrinsic=False, shuffle=False, reparameterizer=None,
                scale_amp=False):
        self.precalwf_filelist = precalwf_filelist
        self.parameter_names = parameter_names
        self.data_generator = data_generator
        self.Nbasis_wf = Nbasis_wf
        self.Nbasis_det = Nbasis_det
        self.Vhfile_dict = Vhfile_dict

        self.snr_min = snr_min
        self.snr_max = snr_max
        self.snr_generator = torch.distributions.Uniform(snr_min, snr_max)
        self.tmin = tmin
        self.tmax = tmax
        self.device = device
        self.minibatch_size = minibatch_size
        self.add_noise = add_noise
        self.fix_extrinsic = fix_extrinsic
        self.shuffle = shuffle
        self.reparameterizer = reparameterizer
        self.scale_amp = scale_amp

        # Load V and Vh matrices and convert to tensors
        self.V_wf, self.Vh_wf = loadVandVh(Vhfile_dict['waveform'], Nbasis_wf)
        self.V_wf = torch.from_numpy(self.V_wf).to(self.device).type(torch.complex64)
        self.Vh_wf = torch.from_numpy(self.Vh_wf).to(self.device).type(torch.complex64)

        self.antenna = GWAntennaOnGPU(data_generator.detector_names, device=self.device,
                                     gmst_fit=True, gps_start=tmin-12000, gps_end=tmax+12000)
        #self.farray = torch.from_numpy(data_generator.frequency_array_masked).float().to(self.device)
        self.ifos = data_generator.ifos
        self.Ndet = len(self.ifos)
        self.farray_mb = torch.from_numpy(data_generator.farray_mb).float().to(self.device)
        self.duration_array = torch.from_numpy(data_generator.waveform_generator.duration_array).float().to(self.device)
        #self.frequency_nodes = torch.from_numpy(data_generator.waveform_generator.frequency_nodes).float().to(self.device)
        self.det_aux = self.prepare_detector_aux()
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
        
    def prepare_detector_aux(self):
        det_aux = {}
        for det in self.ifos:
            detname = det.name
            psd = det.power_spectral_density.power_spectral_density_interpolated(self.farray_mb.cpu())
            psd = torch.from_numpy(psd).double().to(self.device)
            V, Vh = loadVandVh(self.Vhfile_dict[detname], self.Nbasis_det)
            V = torch.from_numpy(V).to(self.device)
            whitened_V = (V.T * 1/(psd*self.duration_array/4)**0.5).T
            det_aux[detname] = {'whitened_V': whitened_V.type(torch.complex64),
                                'PSD': psd,
                                'V': V}
        return det_aux

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
        
        #dL = injection_parameters['luminosity_distance'].unsqueeze(-1)
        #hp_svd = hp_svd/dL
        #hc_svd = hc_svd/dL

        x, injection_parameters, netsnr = self.compute_strain_tensors_batch(hp_svd, hc_svd, injection_parameters)
        theta = self.get_theta(injection_parameters)
        return theta.float(), torch.cat((x.real, x.imag), axis=1).float(), netsnr.float()

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
                raise Exception(f'index_of_file not found: {index_of_file}')
            self.cached_wf_file = wf_dict
            self.cached_wf_file_index = index_of_file
            return wf_dict
    
    
    def get_waveform_tensors_batch(self, wf_dict, index_in_file, index_in_file_end):
        index = self.random_index_in_file[index_in_file:index_in_file_end]
        #hp_svd = (torch.from_numpy(wf_dict['waveform_polarizations']['plus'][index])).type(torch.complex64).to(self.device)
        #hc_svd = (torch.from_numpy(wf_dict['waveform_polarizations']['cross'][index])).type(torch.complex64).to(self.device)
        hp_svd = (torch.from_numpy(wf_dict['waveforms'][2*index][:,:self.Nbasis_wf])).type(torch.complex64).to(self.device)
        hc_svd = (torch.from_numpy(wf_dict['waveforms'][2*index+1][:,:self.Nbasis_wf])).type(torch.complex64).to(self.device)

        para_name_list = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                    'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'phase','mass_1', 'mass_2']
        injection_parameters = {key: torch.tensor(wf_dict['injection_parameters'][key][index], device=self.device) for key in para_name_list}
        # last wf in this file
        if (index_in_file_end is None) and (self.cached_wf_file_index < len(self.precalwf_filelist)-1): 
            _ = self.get_precalwf_dict(self.cached_wf_file_index + 1)
            print('Automatically updated cached wf.')

        return hp_svd, hc_svd, injection_parameters

    def get_noise_tensors_batch(self):
        if not self.scale_amp:
            white_noise = (torch.randn((self.minibatch_size, self.Ndet, self.Nbasis_det), device=self.device) + \
                           1j * torch.randn((self.minibatch_size, self.Ndet, self.Nbasis_det), device=self.device)).type(torch.complex64)
        else:
            white_noise = (torch.randn((self.minibatch_size, self.Ndet, len(self.farray_mb)), device=self.device) + \
                           1j * torch.randn((self.minibatch_size, self.Ndet, len(self.farray_mb)), device=self.device)).type(torch.complex64)
        return white_noise
    
    def inner_product(self, a, b, psd):
        integrand = torch.conj(a) * b / psd / self.duration_array
        integral = torch.sum(integrand, axis=-1)
        return 4. * torch.real(integral)
    
    def compute_strain_tensors_batch(self, hp_svd, hc_svd, injection_parameters):
        num_ifos = len(self.ifos)
        x = torch.zeros((self.minibatch_size, num_ifos, self.Nbasis_det), dtype=torch.complex64, device=self.device)
        resp_dt_dict = self.compute_detector_factors_batch(injection_parameters)
        
        ref_snr_sq = torch.zeros(self.minibatch_size, device=self.device)
        snr_sample = self.snr_generator.sample((self.minibatch_size,)).to(self.device)
        
        
        Vh_recons = self.Vh_wf
        hp_reconstruct = torch.einsum('ij,jk->ik', hp_svd.type(torch.complex64), Vh_recons.type(torch.complex64))
        hc_reconstruct = torch.einsum('ij,jk->ik', hc_svd.type(torch.complex64), Vh_recons.type(torch.complex64))
        mc = injection_parameters['chirp_mass']
        for i,det in enumerate(self.ifos):
            detname = det.name
            fp, fc, dt = resp_dt_dict[detname]
            h_reconstruct = fp*hp_reconstruct + fc*hc_reconstruct
            
            phase2add = torch.exp(-1j * 2 * np.pi * dt * self.farray_mb).type(torch.complex64)
            hh_shifted = h_reconstruct * phase2add
            
            if self.scale_amp:
                hh_dehet = hh_shifted * mc.unsqueeze(1)**(5/6) * self.farray_mb.unsqueeze(0)**(-7/6)
            else:
                hh_dehet = hh_shifted
            #ref_snr_sq += inner_product(hh_dehet, hh_dehet, self.duration_array, self.det_aux[detname]['PSD'], torch)
            ref_snr_sq += self.inner_product(hh_dehet, hh_dehet, self.det_aux[detname]['PSD'])
            h_svd = torch.einsum('ij,jk->ik', hh_shifted.type(torch.complex64), self.det_aux[detname]['whitened_V'].type(torch.complex64))
            x[:,i,:] = h_svd 

        ref_snr = ref_snr_sq**0.5
        distance = ref_snr/snr_sample 
        injection_parameters['luminosity_distance'] = distance
        x = x / distance.unsqueeze(1).unsqueeze(2)

        if self.add_noise:
            n_svd = self.get_noise_tensors_batch()
            if self.scale_amp:
                scaled_noise = torch.zeros((self.minibatch_size, self.Ndet, self.Nbasis_det), device=self.device).type(torch.complex64)
                for i,det in enumerate(self.ifos):
                    n_svd[:,i,:] = n_svd[:,i,:] / mc.unsqueeze(1)**(5/6) / self.farray_mb.unsqueeze(0)**(-7/6)
                    scaled_noise[:,i,:] = torch.einsum('ij,jk->ik', n_svd[:,i,:].type(torch.complex64), self.det_aux[detname]['V'].type(torch.complex64))
                n_svd = scaled_noise
                
            x = x + n_svd
            
        return x, injection_parameters, snr_sample

    def get_example_distances(self, Nexample=10000, Nstart=0):        
        ref_snr_sq = torch.zeros(Nexample, device=self.device)
        snr_sample = self.snr_generator.sample((Nexample,)).to(self.device)
        
        index = self.random_index_in_file[Nstart:Nstart+Nexample]
        wf_dict = self.cached_wf_file
        hp_svd = (torch.from_numpy(wf_dict['waveforms'][2*index][:,:self.Nbasis_wf])).type(torch.complex64).to(self.device)
        mc = torch.tensor(wf_dict['injection_parameters']['chirp_mass'][index], device=self.device) 
        Vh_recons = self.Vh_wf
        hp_reconstruct = torch.einsum('ij,jk->ik', hp_svd.type(torch.complex64), Vh_recons.type(torch.complex64))
        if self.scale_amp:
            hp_reconstruct = hp_reconstruct * mc.unsqueeze(1)**(5/6) * self.farray_mb.unsqueeze(0)**(-7/6)

        h_resp_average = hp_reconstruct * 0.2**0.5 * 2
        for i,det in enumerate(self.ifos):
            #h_reconstruct = fp*hp_reconstruct + fc*hc_reconstruct
            ref_snr_sq += self.inner_product(h_resp_average, h_resp_average, self.det_aux[det.name]['PSD'])

        ref_snr = ref_snr_sq**0.5
        distance = ref_snr/snr_sample 

        return distance

    def compute_detector_factors_batch(self, injection_parameters):
        L = len(self.farray_mb)
        bs = self.minibatch_size
        m1 = injection_parameters['mass_1']
        m2 = injection_parameters['mass_2']
        tau_array = tau_of_f_gpubatch(self.farray_mb,
                           m1=m1,
                           m2=m2,
                           chi = torch.zeros_like(m1).to(self.device),
                                     device=self.device) 
        
        # tau_array: [minibatch, len(f)] 
        # ra, dec, tc, psi: [minibatch] 
        ra = injection_parameters['ra'].type(torch.double)
        dec = injection_parameters['dec'].type(torch.double)
        tc = injection_parameters['geocent_time'].type(torch.double)
        psi = injection_parameters['psi'].type(torch.double)

        tc_array = (tc.unsqueeze(1) - tau_array).reshape(bs * L)
        ra_reshape = ra.unsqueeze(1).repeat(1, L).reshape(bs * L)
        dec_reshape = dec.unsqueeze(1).repeat(1, L).reshape(bs * L)
        psi_reshape = psi.unsqueeze(1).repeat(1, L).reshape(bs * L)
        
        resp_dt_dict_reshape = self.antenna.resp_and_dt(ra_reshape, dec_reshape, tc_array, psi_reshape)
        
        resp_dt_dict = {}  
        dt_geocent = tc.unsqueeze(1)
        for detname, result_list in resp_dt_dict_reshape.items():
            resp_dt_dict[detname] = []
            for dd in result_list:
                resp_dt_dict[detname].append(dd.reshape(bs, L))
            resp_dt_dict[detname][2] += dt_geocent        
        
        return resp_dt_dict
    
    

    def get_theta(self, injection_parameters):
        
        if self.reparameterizer is not None:
            theta = self.reparameterizer.reparameterize(injection_parameters)
        else:
            if self.fix_extrinsic:
                parameter_names = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                        'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'phase']
            else:
                parameter_names = self.parameter_names
            theta_list = []
            for paraname in parameter_names:
                para_re = injection_parameters[paraname]
                theta_list.append(para_re)
            theta_array = np.array(theta_list)
            theta = torch.tensor(theta_array, dtype=torch.float32).to(self.device).T
    
        return theta
    
    def update_injection_parameters_batch(self, injection_parameters):
        if self.fix_extrinsic:
            injection_parameters['ra'] = torch.zeros(self.minibatch_size, device=self.device) + 1
            injection_parameters['dec'] = torch.zeros(self.minibatch_size, device=self.device) + 1
            injection_parameters['psi'] = torch.zeros(self.minibatch_size, device=self.device) + 1
            injection_parameters['geocent_time'] = torch.zeros(self.minibatch_size, device=self.device)
            injection_parameters['luminosity_distance'] = torch.zeros(self.minibatch_size, device=self.device) + 100
    
        else:
            injection_parameters['ra'] = torch.rand(self.minibatch_size, device=self.device) * 2 * np.pi
            injection_parameters['dec'] = torch.asin(torch.rand(self.minibatch_size, device=self.device) * 2 - 1)
            injection_parameters['psi'] = torch.rand(self.minibatch_size, device=self.device) * np.pi
            injection_parameters['geocent_time'] = torch.rand(self.minibatch_size, device=self.device) * (self.tmax - self.tmin) + self.tmin
            #injection_parameters['luminosity_distance'] = (self.dmax - self.dmin) * np.random.power(a=self.dpower, size=self.minibatch_size) + self.dmin
            #injection_parameters['luminosity_distance'] = torch.tensor(injection_parameters['luminosity_distance']).to(self.device)
        
        return injection_parameters
    
    def shuffle_wflist(self):
        #if self.shuffle:
        random.shuffle(self.precalwf_filelist)
        
    def shuffle_indexinfile(self):
        if self.shuffle:
            self.random_index_in_file = np.random.permutation(self.sample_per_file)
        else:
            self.random_index_in_file = np.arange(self.sample_per_file)


class DatasetSVDMBStrainFDFromSVDWFonGPUBatchParallelDet(DatasetSVDMBStrainFDFromSVDWFonGPUBatch):
    '''
    Same as DatasetSVDMBStrainFDFromSVDWFonGPUBatch, but waveform projection is done in parallel for different detectors.
    Fast when minibatchsize is large whrn loading large amount of data, but also more memory consuming. 
    '''
    def __init__(self, precalwf_filelist, parameter_names, data_generator, Nbasis_wf, Nbasis_det, Vhfile_dict,
                snr_min=10, snr_max=200, tmin=-0.01, tmax=0.01, device='cuda',
                add_noise=True, minibatch_size=1, fix_extrinsic=False, shuffle=False, reparameterizer=None):
        super().__init__(precalwf_filelist, parameter_names, data_generator, Nbasis_wf, Nbasis_det, Vhfile_dict,
                         snr_min, snr_max, tmin, tmax, device,
                         add_noise, minibatch_size, fix_extrinsic, shuffle, reparameterizer)

    def compute_strain_tensors_batch(self, hp_svd, hc_svd, injection_parameters):
        num_ifos = len(self.ifos)
        x = torch.zeros((self.minibatch_size, num_ifos, self.Nbasis_det), dtype=torch.complex64, device=self.device)
        resp_dt_dict = self.compute_detector_factors_batch(injection_parameters)
        
        snr_sample = self.snr_generator.sample((self.minibatch_size,)).to(self.device)
        
        
        Vh_recons = self.Vh_wf
        hp_reconstruct = torch.einsum('ij,jk->ik', hp_svd.type(torch.complex64), Vh_recons.type(torch.complex64))
        hc_reconstruct = torch.einsum('ij,jk->ik', hc_svd.type(torch.complex64), Vh_recons.type(torch.complex64))
        mc = injection_parameters['chirp_mass']
        
        fp_list = torch.stack([resp_dt_dict[det.name][0] for det in self.ifos])
        fc_list = torch.stack([resp_dt_dict[det.name][1] for det in self.ifos])
        dt_list = torch.stack([resp_dt_dict[det.name][2] for det in self.ifos])
        whitened_V_list = torch.stack([self.det_aux[det.name]['whitened_V'].type(torch.complex64) for det in self.ifos])

        hp_reconstruct_expanded = hp_reconstruct.unsqueeze(0)
        hc_reconstruct_expanded = hc_reconstruct.unsqueeze(0)
        h_reconstruct = fp_list * hp_reconstruct_expanded + fc_list * hc_reconstruct_expanded
        
        phase2add = torch.exp(-1j * 2 * np.pi * dt_list * self.farray_mb.unsqueeze(0)).type(torch.complex64)
        hh_shifted = h_reconstruct * phase2add
        
        hh_dehet = hh_shifted #* mc.unsqueeze(0).unsqueeze(-1)**(5/6) * self.farray_mb.unsqueeze(0).unsqueeze(0)**(-7/6)
        

        psd_list = torch.stack([self.det_aux[det.name]['PSD'] for det in self.ifos]) # (Ndet, L)
        ref_snr_sq = torch.sum(self.inner_product(hh_dehet, hh_dehet, psd_list.unsqueeze(1)), dim=0)

        h_svd = torch.einsum('ijk,ikl->ijl', hh_shifted.type(torch.complex64), whitened_V_list.type(torch.complex64))
        x = h_svd
        
        

        ref_snr = ref_snr_sq**0.5
        distance = ref_snr/snr_sample 
        injection_parameters['luminosity_distance'] = distance
        x = x / distance.unsqueeze(0).unsqueeze(-1)

        if self.add_noise:
            n_svd = self.get_noise_tensors_batch()
            x = x.view(self.minibatch_size, self.Ndet, self.Nbasis_det) + n_svd
            
        return x, injection_parameters, snr_sample

