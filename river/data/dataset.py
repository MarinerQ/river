import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
#from .utils import PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY
from .utils import * 
#from ..models.utils import project_strain_data_FDAPhi

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

class DatasetStrainFDFromPreCalSVDWF(Dataset):
    '''
    From pre-calculated waveforms, stored in SVD
    '''
    def __init__(self, precalwf_list,  parameter_names, data_generator, ipca_gen, dmin=10, dmax=200, dpower=1):
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
        self.ipca_gen = ipca_gen

    def __len__(self):
        return self.Nsample

    def __getitem__(self, index):
        index_precalwf_list, index_wf_dict = self.get_wf_index(index)

        wf_dict = self.data_generator.get_one_waveform(index_wf_dict, self.precalwf_list[index_precalwf_list]['waveform_polarizations'])
        injection_parameters = self.data_generator.get_one_injection_parameters(index_wf_dict,  self.precalwf_list[index_precalwf_list]['injection_parameters'], is_intrinsic_only=True)
        injection_parameters = self.update_injection_parameters(injection_parameters)

        while not self.data_generator.inject_one_signal_from_waveforms(injection_parameters, wf_dict):
            injection_parameters = self.update_injection_parameters(injection_parameters) # until this injection can be detected
        
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