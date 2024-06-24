import numpy as np
import torch

MEAN_VAR_DICT_BNS = {
    'log10_chirp_mass': (0.21893410187780157, 0.009063873587706993), #assume m1, m2 ~ U(1,3)
    'chirp_mass': (1.6950131506973254, 0.1322087464067809),
    'mass_ratio': (0.7259431133909592, 0.029381495904049214),
    'a_1': (0.5, 1/12),
    'a_2': (0.5, 1/12),
    'phi_12': (np.pi,np.pi**2/3),
    'phi_jl': (np.pi,np.pi**2/3),
    'tilt_1': (np.pi/2,0.467),
    'tilt_2': (np.pi/2,0.467),
    'lambda_tilde': (2763.790376592589, 1791002.30359124),
    'delta_lambda_tilde': (-408.7502400946281, 286825.89080037485),
    'theta_jn': (np.pi/2,0.467),
    'ra': (np.pi,np.pi**2/3),
    'dec': (0, 0.467),
    'psi': (np.pi/2,np.pi**2/12),
    'geocent_time': (0,0.2**2 / 12), # (-0.1, 0.1)
    'luminosity_distance': (105, 3008.333333),
    'phase': (np.pi,np.pi**2/3),
    'X': (1.5, 5/12), # X = (psi+phase) / pi
    'psiprime': (1, 1/3),  # psiprime = psi/ (pi/2),
    'timing_error': (0, 0.01**2),
}

def normalize(x, mean, var):
    xbar = (x-mean) / var**0.5
    return xbar

def inverse_normalize(xbar, mean, var):
    x = xbar*var**0.5 + mean
    return x


def reparameterize(para, paraname, all_para_dict=None):
    if paraname == 'chirp_mass':
        mean ,var = MEAN_VAR_DICT_BNS['log10_chirp_mass']
        new_para = normalize(np.log10(para), mean, var)
        #mean ,var = MEAN_VAR_DICT_BNS['chirp_mass']
        #new_para = normalize(para, mean, var)
    elif paraname == 'psi' and all_para_dict:
        # psi -> psiprime
        mean ,var = MEAN_VAR_DICT_BNS['psiprime']
        psi = para
        psiprime = psi/ (np.pi/2)
        new_para = normalize(psiprime, mean, var)
    elif paraname == 'phase' and all_para_dict:
        # phase -> X 
        mean ,var = MEAN_VAR_DICT_BNS['X']
        phase = para
        psi = all_para_dict['psi']
        X = (psi+phase) / np.pi
        new_para = normalize(X, mean, var)
    else:
        mean, var = MEAN_VAR_DICT_BNS[paraname]
        new_para = normalize(para, mean, var)

    return new_para

def inverse_reparameterize(new_para, paraname, all_para_dict=None):
    if paraname == 'chirp_mass':
        mean ,var = MEAN_VAR_DICT_BNS['log10_chirp_mass']
        #mean ,var = MEAN_VAR_DICT_BNS['chirp_mass']
        para = inverse_normalize(new_para, mean, var)
        para = 10**para
    elif paraname == 'psi' and all_para_dict:
        mean ,var = MEAN_VAR_DICT_BNS['psiprime']
        psiprime = inverse_normalize(new_para, mean, var)
        psi = psiprime*np.pi/2
        para = psi 
    elif paraname == 'phase' and all_para_dict:
        mean ,var = MEAN_VAR_DICT_BNS['X']
        X = inverse_normalize(new_para, mean, var)
        psi = inverse_reparameterize(all_para_dict['psi'], 'psi', all_para_dict)
        phase = X*np.pi - psi 
        para = phase 
    else:
        mean, var = MEAN_VAR_DICT_BNS[paraname]
        para = inverse_normalize(new_para, mean, var)

    return para



class ReparametrizerGPU:
    def __init__(self, context_parameter_names, example_injection_parameters, device='cuda'):
        '''
        example_injection_parameters: dict of numpy.ndarray
        '''
        self.device = device
        self.context_parameter_names = context_parameter_names 

        example_injection_parameters["X"] = (example_injection_parameters["psi"] + example_injection_parameters["phase"]) / np.pi
        example_injection_parameters["psiprime"] = example_injection_parameters["psi"] / (np.pi/2)
        self.MEAN_VAR_DICT = {}
        for paraname, para in example_injection_parameters.items():
            para = torch.tensor(para, device=self.device)
            mean, var = torch.mean(para), torch.var(para)
            self.MEAN_VAR_DICT[paraname] = (mean, var)
        
        self.mu = torch.tensor([self.MEAN_VAR_DICT[name][0] for name in self.context_parameter_names], device=self.device)
        self.sigma = torch.tensor([self.MEAN_VAR_DICT[name][1]**0.5 for name in self.context_parameter_names], device=self.device)


    def reparameterize(self, injection_parameters, tensor_input=False):
        '''
        injection_parameters should be the normal bilby injection parameters
        '''
        injection_parameters["X"] = (injection_parameters["psi"] + injection_parameters["phase"]) / np.pi
        injection_parameters["psiprime"] = injection_parameters["psi"] / (np.pi/2)
        theta = torch.stack([injection_parameters[name].clone().detach() if tensor_input else torch.tensor(injection_parameters[name], device=self.device) for name in self.context_parameter_names]).T
        #theta = torch.stack([torch.tensor(injection_parameters[name], device=self.device) for name in self.context_parameter_names]).T
        theta_bar = (theta - self.mu.unsqueeze(0)) / self.sigma.unsqueeze(0)
        return theta_bar

    def inverse_reparameterize(self, theta_bar):
        theta = theta_bar * self.sigma.unsqueeze(0) + self.mu.unsqueeze(0)
        injection_parameters = {}
        for i, name in enumerate(self.context_parameter_names):
            injection_parameters[name] = theta[:, i].detach().cpu().numpy()

        injection_parameters['psi'] = injection_parameters['psiprime'] * (np.pi/2)
        injection_parameters['phase'] = injection_parameters['X'] * np.pi - injection_parameters['psi']

        return injection_parameters


class ReparametrizerGPUOriginPara:
    def __init__(self, context_parameter_names, example_injection_parameters, device='cuda'):
        '''
        example_injection_parameters: dict of numpy.ndarray
        '''
        self.device = device
        self.context_parameter_names = context_parameter_names 

        self.MEAN_VAR_DICT = {}
        for paraname, para in example_injection_parameters.items():
            para = torch.tensor(para, device=self.device)
            mean, var = torch.mean(para), torch.var(para)
            self.MEAN_VAR_DICT[paraname] = (mean, var)
        
        self.mu = torch.tensor([self.MEAN_VAR_DICT[name][0] for name in self.context_parameter_names], device=self.device)
        self.sigma = torch.tensor([self.MEAN_VAR_DICT[name][1]**0.5 for name in self.context_parameter_names], device=self.device)


    def reparameterize(self, injection_parameters, tensor_input=False):
        '''
        injection_parameters should be the normal bilby injection parameters
        '''
        theta = torch.stack([injection_parameters[name].clone().detach() if tensor_input else torch.tensor(injection_parameters[name], device=self.device) for name in self.context_parameter_names]).T
        #theta = torch.stack([torch.tensor(injection_parameters[name], device=self.device) for name in self.context_parameter_names]).T
        theta_bar = (theta - self.mu.unsqueeze(0)) / self.sigma.unsqueeze(0)
        return theta_bar

    def inverse_reparameterize(self, theta_bar):
        theta = theta_bar * self.sigma.unsqueeze(0) + self.mu.unsqueeze(0)
        injection_parameters = {}
        for i, name in enumerate(self.context_parameter_names):
            injection_parameters[name] = theta[:, i].detach().cpu().numpy()

        return injection_parameters