import numpy as np

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