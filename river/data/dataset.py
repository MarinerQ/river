import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
#from .utils import PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
from .utils import * 
#from ..models.utils import project_strain_data_FDAPhi
import pickle
import random


def reparameterize_mass(mass):
    return np.log10(mass)

class DatasetStrainFD(Dataset):
    def __init__(self, data_dict, parameter_names):
        self.farray = torch.from_numpy(data_dict['farray']).float()
        self.Nsample = data_dict['Nsample'][0]
        self.paradim = len(parameter_names)
        self.detector_names = list(data_dict['strains'].keys())

        self.injection_parameters = np.zeros((self.Nsample, self.paradim))
        for i, parameter_name in enumerate(parameter_names):
            if parameter_name in ['chirp_mass']:
                self.injection_parameters[:,i] = reparameterize_mass(data_dict['injection_parameters'][parameter_name])
            else:
                self.injection_parameters[:,i] = data_dict['injection_parameters'][parameter_name]
        self.injection_parameters = torch.from_numpy(self.injection_parameters).float()

        s = np.array(list(data_dict['strains'][detname] for detname in self.detector_names ))
        psd = np.array(list(data_dict['PSDs'][detname] for detname in self.detector_names ))
        inv_asd = np.float32(1 / (psd**0.5))
        ###s_whitened = np.complex64(s*inv_asd)
        s_whitened = np.complex64(s*1e23)
        self.inv_asd = torch.from_numpy(inv_asd*1e-23).movedim(0,1).float()
        self.strain = torch.from_numpy(s_whitened).movedim(0,1) # is complex

        #strain_r = np.real(s_whitened)
        #strain_i = np.imag(s_whitened)
        #self.strain1 = torch.from_numpy(strain1).movedim(0,1).float()
        #self.strain2 = torch.from_numpy(strain2).movedim(0,1).float()


    def __len__(self):
        return self.Nsample

    def __getitem__(self, index):
        theta = self.injection_parameters[index]
        inv_asd = self.inv_asd[index]
        strain = self.strain[index]

        return theta, strain, inv_asd


class DatasetXFromPreCalSVDData(Dataset):
    '''
    From pre-calculated data, stored in SVD that can be feed into embedding layer directly
    '''
    def __init__(self, precaldata_list,  parameter_names):
        #self.farray = torch.from_numpy(data_dict['farray']).float()

        self.precaldata_list = precaldata_list
        self.sample_per_file = len(precaldata_list[0]['injection_parameters']['chirp_mass'])
        self.nfile = len(self.precaldata_list)
        self.Nsample = self.nfile * self.sample_per_file 

        self.parameter_names = parameter_names

    def __len__(self):
        return self.Nsample

    def __getitem__(self, index):
        index_precaldata_list, index_data_dict = self.get_data_index(index)
        x = torch.from_numpy( self.precaldata_list[index_precaldata_list]['x'][index_data_dict] ).float()
        theta = self.get_theta(self.precaldata_list[index_precaldata_list]['injection_parameters'], index_data_dict)

        return theta, x

    def get_data_index(self, index):
        index_precaldata_list = index // self.sample_per_file
        index_data_dict = index - index_precaldata_list*self.sample_per_file

        return index_precaldata_list, index_data_dict

    def get_theta(self, injection_parameters_all, index):
        theta = []
        for paraname in self.parameter_names:
            tt = injection_parameters_all[paraname][index]
            theta.append(tt)
            '''
            if paraname in ['chirp_mass']:
                theta.append(reparameterize_mass(tt))
            elif paraname in ['ra', 'dec', 'psi', 'phi', 'phi_12', 'phi_jl', 'tilt_1', 'tilt_2', 'theta_jn']:
                theta.append(tt/np.pi)
            elif paraname in ['luminosity_distance']:
                theta.append(tt/100)
            elif paraname in ['lambda_tilde', 'delta_lambda_tilde']:
                theta.append(tt/1000)
            else:
                theta.append(tt)
            '''
        return torch.from_numpy(np.array(theta)).float()



class DatasetXFromPreCalSVDData2D(Dataset):
    '''
    From pre-calculated data, stored in SVD that can be feed into embedding layer directly
    '''
    def __init__(self, precaldata_list,  parameter_names):
        #self.farray = torch.from_numpy(data_dict['farray']).float()

        self.precaldata_list = precaldata_list
        self.sample_per_file = len(precaldata_list[0]['injection_parameters']['chirp_mass'])
        self.nfile = len(self.precaldata_list)
        self.Nsample = self.nfile * self.sample_per_file 

        self.parameter_names = parameter_names

    def __len__(self):
        return self.Nsample

    def __getitem__(self, index):
        index_precaldata_list, index_data_dict = self.get_data_index(index)
        x = torch.from_numpy( self.precaldata_list[index_precaldata_list]['x'][index_data_dict] ).float()
        theta = self.get_theta(self.precaldata_list[index_precaldata_list]['injection_parameters'], index_data_dict)

        return theta, x

    def get_data_index(self, index):
        index_precaldata_list = index // self.sample_per_file
        index_data_dict = index - index_precaldata_list*self.sample_per_file

        return index_precaldata_list, index_data_dict

    def get_theta(self, injection_parameters_all, index):
        theta = []
        for paraname in self.parameter_names:
            if paraname in ['chirp_mass', 'mass_ratio']:
                tt = injection_parameters_all[paraname][index]
                theta.append(tt)
            '''
            if paraname in ['chirp_mass']:
                theta.append(reparameterize_mass(tt))
            elif paraname in ['ra', 'dec', 'psi', 'phi', 'phi_12', 'phi_jl', 'tilt_1', 'tilt_2', 'theta_jn']:
                theta.append(tt/np.pi)
            elif paraname in ['luminosity_distance']:
                theta.append(tt/100)
            elif paraname in ['lambda_tilde', 'delta_lambda_tilde']:
                theta.append(tt/1000)
            else:
                theta.append(tt)
            '''
        return torch.from_numpy(np.array(theta)).float()
    

class DatasetSVDStrainFDFromSVDWF(Dataset):
    '''
    Simulate FD data in SVD space from pre-calculated SVD waveforms. 
    '''
    def __init__(self, precalwf_list,  parameter_names, data_generator, Nbasis, V, dmin=10, dmax=200, dpower=1):
        #self.farray = torch.from_numpy(data_dict['farray']).float()

        self.precalwf_list = precalwf_list
        self.sample_per_file = len(precalwf_list[0]['injection_parameters']['chirp_mass'])
        self.nfile = len(self.precalwf_list)
        self.Nsample = self.nfile * self.sample_per_file 

        self.dmin = dmin
        self.dmax = dmax
        self.dpower = dpower
        
        self.parameter_names = parameter_names
        self.paradim = len(parameter_names)
        self.data_generator = data_generator
        self.detector_names = data_generator.detector_names
        #self.ipca_gen = ipca_gen
        self.V = V
        self.Vh = V.T.conj()
        self.Nbasis = Nbasis

    def __len__(self):
        return self.Nsample

    def __getitem__(self, index):
        index_precalwf_list, index_wf_dict = self.get_wf_index(index)
        
        wf_dict = load_dict_from_hdf5(pp)

        hp_svd = wf_dict['waveform_polarizations']['plus']['amplitude'][testid] * np.exp(1j*wf_dict['waveform_polarizations']['plus']['phase'][testid])

        
        #wf_dict = self.data_generator.get_one_waveform(index_wf_dict, self.precalwf_list[index_precalwf_list]['waveform_polarizations'])
        injection_parameters = self.data_generator.get_one_injection_parameters(index_wf_dict,  self.precalwf_list[index_precalwf_list]['injection_parameters'], is_intrinsic_only=True)
        injection_parameters = self.update_injection_parameters(injection_parameters)

        #while not self.data_generator.inject_one_signal_from_waveforms(injection_parameters, wf_dict):
        #    injection_parameters = self.update_injection_parameters(injection_parameters) # until this injection can be detected
        
        data_dict = self.data_generator.data
        s = np.array(list(data_dict['strains'][detname] for detname in self.detector_names ))
        psd = np.array(list(data_dict['PSDs'][detname] for detname in self.detector_names ))
        inv_asd = np.float32(1 / (psd**0.5))
        ###s_whitened = np.complex64(s*inv_asd)
        s_whitened = np.complex64(s*1e23)

        inv_asd = torch.from_numpy(inv_asd*1e-23).movedim(0,1).float()[0]
        strain = torch.from_numpy(s_whitened).movedim(0,1)[0] # is complex
        theta = self.get_theta(injection_parameters)

        x = self._project_strain_data_FDAPhi(strain, inv_asd, self.detector_names, self.ipca_gen)
        self.data_generator.initialize_data()
        #return theta.clone().detach(), strain.clone().detach(), inv_asd.clone().detach()
        return theta.clone().detach(), x.squeeze(0).clone().detach()

    def get_wf_index(self, index):
        index_precalwf_list = index // self.sample_per_file
        index_wf_dict = index - index_precalwf_list*self.sample_per_file

        return index_precalwf_list, index_wf_dict

    def update_injection_parameters(self, injection_parameters):
        injection_parameters['ra'] = np.random.uniform(0, np.pi)
        injection_parameters['dec'] = np.arcsin(np.random.uniform(-1, 1))
        injection_parameters['psi'] = np.random.uniform(0, np.pi)
        injection_parameters['geocent_time'] = np.random.uniform(-0.1, 0.1)
        injection_parameters['luminosity_distance'] = generate_random_distance(Nsample=1, low=self.dmin, high=self.dmax, power=self.dpower)[0]

        return injection_parameters
   
    def get_theta(self, injection_parameters):
        theta = []
        for paraname in self.parameter_names:
            tt = injection_parameters[paraname]
            if paraname in ['chirp_mass']:
                theta.append(reparameterize_mass(tt))
            else:
                theta.append(tt)
        
        return torch.from_numpy(np.array(theta)).float()

    def _project_strain_data_FDAPhi(self, strain, psd, detector_names, ipca_gen, project=True, downsample_rate=1, dim=1):
        '''
        strain: DatasetStrainFD in batches, e.g. DatasetStrainFD[0:10]
        psd: strain-like
        detector_names: DatasetStrainFD.detector_names
        ipca_gen: IPCAGenerator
        '''
        strain = np.expand_dims(strain, 0)
        psd = np.expand_dims(psd, 0)
        strain_amp = np.abs(strain)
        strain_phi = np.unwrap(np.angle(strain) , axis=-1)
        strain_real = np.real(strain)
        strain_imag = np.imag(strain)

        n_components = ipca_gen.n_components
        batch_size = strain.shape[0]
        ndet = len(detector_names)

        output_amp = []
        output_phi = []
        output_psd = []
        for i,detname in enumerate(detector_names):
            if project:
                output_amp.append(ipca_gen.project(strain_amp[:,i,:], detname, 'amplitude'))
                output_phi.append(ipca_gen.project(strain_phi[:,i,:], detname, 'phase'))
                output_psd.append(ipca_gen.project(psd[:,i,:], detname, 'amplitude'))
            else:
                output_amp.append(strain_amp.numpy()[:,i,:][:,::downsample_rate])
                output_phi.append(strain_phi[:,i,:][:,::downsample_rate])
                #output_amp.append(strain_real.numpy()[:,i,:][:,::downsample_rate])
                #output_phi.append(strain_imag.numpy()[:,i,:][:,::downsample_rate])
                output_psd.append(psd.numpy()[:,i,:][:,::downsample_rate])

        output_amp = torch.from_numpy(np.array(output_amp))
        output_phi = torch.from_numpy(np.array(output_phi))
        output_psd = torch.from_numpy(np.array(output_psd))
        data_length = output_amp.shape[-1]
        if dim==1:
            return torch.cat((output_amp, output_phi, output_psd)).movedim(0,1).float()
        elif dim==2:
            return torch.cat((output_amp, output_phi, output_psd)).movedim(0,1).float().view((batch_size,3,ndet,data_length))

class DatasetStrainFDFromFolder(Dataset):
    def __init__(self, data_folder, filename_prefix,  parameter_names, ipca, data_generator, nbatch = 100, file_per_batch = 1000, sample_per_file = 10):
        #self.farray = torch.from_numpy(data_dict['farray']).float()

        self.data_folder = data_folder
        self.filename_prefix = filename_prefix
        self.nbatch = nbatch
        self.file_per_batch = file_per_batch
        self.sample_per_file = sample_per_file
        self.Nsample = self.nbatch * self.file_per_batch * self.sample_per_file
        self.paradim = len(parameter_names)
        self.data_generator = data_generator


        self.detector_names = list(data_dict['strains'].keys())

        self.injection_parameters = np.zeros((self.Nsample, self.paradim))
        for i, parameter_name in enumerate(parameter_names):
            if parameter_name in ['chirp_mass']:
                self.injection_parameters[:,i] = reparameterize_mass(data_dict['injection_parameters'][parameter_name])
            else:
                self.injection_parameters[:,i] = data_dict['injection_parameters'][parameter_name]
        self.injection_parameters = torch.from_numpy(self.injection_parameters).float()

        s = np.array(list(data_dict['strains'][detname] for detname in self.detector_names ))
        psd = np.array(list(data_dict['PSDs'][detname] for detname in self.detector_names ))
        inv_asd = np.float32(1 / (psd**0.5))
        ###s_whitened = np.complex64(s*inv_asd)
        s_whitened = np.complex64(s*1e23)
        self.inv_asd = torch.from_numpy(inv_asd*1e-23).movedim(0,1).float()
        self.strain = torch.from_numpy(s_whitened).movedim(0,1) # is complex

        #strain_r = np.real(s_whitened)
        #strain_i = np.imag(s_whitened)
        #self.strain1 = torch.from_numpy(strain1).movedim(0,1).float()
        #self.strain2 = torch.from_numpy(strain2).movedim(0,1).float()

    def __len__(self):
        return self.Nsample

    def __getitem__(self, index):
        filepath, index_inside_file = self.get_file_path_and_index(index)
        file_dict = load_dict_from_hdf5(filepath)
        wf_dict = file_dict['waveform_polarizations'][index_inside_file]


        self.data_generator.inject_one_signal_from_waveforms(injection_parameters_dict, wf_dict)


        theta = self.injection_parameters[index]
        inv_asd = self.inv_asd[index]
        strain = self.strain[index]

        return theta, strain, inv_asd

    def get_file_path_and_index(self, index):
        sample_per_batch = self.sample_per_file * self.file_per_batch
        batch_number = index // sample_per_batch
        file_number = (index - sample_per_batch*batch_number) // self.file_per_batch
        index_inside_file = int( index - sample_per_batch*batch_number - self.sample_per_file*file_number )

        return f"{self.data_folder}/batch{batch_number}/{self.filename_prefix}{index_inside_file}.h5", index_inside_file



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
                 dmin=10, dmax=200, dpower=1, loadwf=False, loadnoise=False, device='cuda'):
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

        # Load V and Vh matrices and convert to tensors
        self.V, self.Vh = loadVandVh(Vhfile, Nbasis)
        self.V = torch.from_numpy(self.V).to(self.device).type(torch.complex128)
        self.Vh = torch.from_numpy(self.Vh).to(self.device).type(torch.complex128)

        
        self.farray = torch.from_numpy(data_generator.frequency_array_masked).float().to(self.device)
        self.ifos = data_generator.ifos
        self.det_data = self.prepare_detector_data()
        
        testfile = load_dict_from_hdf5(precalwf_filelist[0])
        self.sample_per_file = len(testfile['injection_parameters']['chirp_mass'])
        self.Nfile = len(self.precalwf_filelist)
        self.Nsample = self.Nfile * self.sample_per_file 
        self.cached_wf_file = testfile
        self.cached_wf_file_index = 0
        
        self.random_index_in_file = np.random.permutation(self.sample_per_file)
            
    def prepare_detector_data(self):
        det_data = {}
        for det in self.ifos:
            detname = det.name
            psd = det.power_spectral_density_array[self.data_generator.frequency_mask]
            psd = torch.from_numpy(psd).double().to(self.device)
            whitened_V = (self.V.T * 1/(psd*det.duration/4)**0.5).T
            det_data[detname] = {'whitened_V': whitened_V.type(torch.complex128)}
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
        x_real, x_imag = self.compute_strain_tensors(hp_svd, hc_svd, injection_parameters)

        theta = self.get_theta(injection_parameters)
        return theta, torch.cat((x_real, x_imag)).float()

    def get_index(self, index, sample_per_file):
        index_of_file = index // sample_per_file
        index_in_file = index - index_of_file*sample_per_file
        index_in_file = self.random_index_in_file[index_in_file]
        
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
        hp_svd = (torch.from_numpy(wf_dict['waveform_polarizations']['plus']['amplitude'][index_in_file]) *\
            torch.exp(1j*torch.from_numpy(wf_dict['waveform_polarizations']['plus']['phase'][index_in_file])).type(torch.complex128)).to(self.device)
        hc_svd = (torch.from_numpy(wf_dict['waveform_polarizations']['cross']['amplitude'][index_in_file]) *\
            torch.exp(1j*torch.from_numpy(wf_dict['waveform_polarizations']['cross']['phase'][index_in_file])).type(torch.complex128)).to(self.device)
        
        return hp_svd, hc_svd

    def get_injection_parameters(self, wf_dict, index_in_file):
        injection_parameters = {key: wf_dict['injection_parameters'][key][index_in_file] for key in ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl',
                    'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'phase']}
        return injection_parameters

    def get_noise_tensors(self, ):
        white_noise = (torch.randn(self.Nbasis, device=self.device) + 1j * torch.randn(self.Nbasis, device=self.device)).type(torch.complex128)

        return white_noise
    
    def compute_strain_tensors(self, hp_svd, hc_svd, injection_parameters):
        x_real, x_imag = [], []
        for det in self.ifos:
            detname = det.name

            fp, fc, dt = self.compute_detector_factors(det, injection_parameters)
            phase2add = torch.exp(-1j * 2 * np.pi * dt * self.farray)
            Vh_recons = self.Vh * phase2add.unsqueeze(0)  # Ensure proper broadcasting
            
            h_svd = torch.matmul(torch.matmul((fp*hp_svd + fc*hc_svd).type(torch.complex128), Vh_recons),
                                 self.det_data[detname]['whitened_V'])
            n_svd = self.get_noise_tensors()
            d_svd = h_svd + n_svd
            
            x_real.append(d_svd.real)
            x_imag.append(d_svd.imag)
        return torch.stack(x_real), torch.stack(x_imag)

    def compute_detector_factors(self, det, injection_parameters):
        # These calculations remain on CPU as they cannot be efficiently vectorized or moved to GPU
        fp = det.antenna_response(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['geocent_time'], injection_parameters['psi'], 'plus')
        fc = det.antenna_response(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['geocent_time'], injection_parameters['psi'], 'cross')
        time_shift = det.time_delay_from_geocenter(injection_parameters['ra'], injection_parameters['dec'], injection_parameters['geocent_time'])
        
        dt_geocent = injection_parameters['geocent_time'] #- self.strain_data.start_time
        dt = dt_geocent + time_shift
            
        return fp, fc, dt

    def get_theta(self, injection_parameters):
        theta = torch.tensor([injection_parameters[paraname] for paraname in self.parameter_names], dtype=torch.float32).to(self.device)
        return theta
    
    def update_injection_parameters(self, injection_parameters):
        injection_parameters['ra'] = np.random.uniform(0, np.pi)
        injection_parameters['dec'] = np.arcsin(np.random.uniform(-1, 1))
        injection_parameters['psi'] = np.random.uniform(0, np.pi)
        injection_parameters['geocent_time'] = np.random.uniform(-0.1, 0.1)
        injection_parameters['luminosity_distance'] = generate_random_distance(Nsample=1, low=self.dmin, high=self.dmax, power=self.dpower)[0]
    
        return injection_parameters
    
    def shuffle_wflist(self):
        random.shuffle(self.precalwf_filelist)
        
    def shuffle_indexinfile(self):
        self.random_index_in_file = np.random.permutation(self.sample_per_file)