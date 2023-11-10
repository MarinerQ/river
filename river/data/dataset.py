import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from .utils import PARAMETER_NAMES_PRECESSINGBNS_BILBY

class DatasetStrainFD(Dataset):
    def __init__(self, data_dict, parameter_names):
        self.farray = torch.from_numpy(data_dict['farray']).float()
        self.Nsample = data_dict['Nsample'][0]
        self.paradim = len(parameter_names)


        self.injection_parameters = np.zeros((self.Nsample, self.paradim))
        for i, parameter_name in enumerate(parameter_names):
            self.injection_parameters[:,i] = data_dict['injection_parameters'][parameter_name]
        self.injection_parameters = torch.from_numpy(self.injection_parameters).float()

        s = np.array(list(data_dict['strains'].values()))
        psd = np.array(list(data_dict['PSDs'].values()))
        inv_asd = np.float32(1 / (psd**0.5))
        s_whitened = np.complex64(s*inv_asd)
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
