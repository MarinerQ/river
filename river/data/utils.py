import numpy as np
import h5py 
import bilby

PARAMETER_NAMES_PRECESSINGBNS_BILBY = ['mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'lambda_1', 'lambda_2', 'theta_jn', 'luminosity_distance', 'ra', 'dec', 'psi', 'phase', 'geocent_time' ]

PARAMETER_NAMES_PRECESSINGBBH_BILBY = ['mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'theta_jn', 'luminosity_distance', 'ra', 'dec', 'psi', 'phase', 'geocent_time' ]

##################### file IO #####################
def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, list)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans




##################### source parameters #####################

def generate_random_component_mass(Nsample, Mmin, Mmax):
    m1 = np.random.uniform(Mmin, Mmax, Nsample)
    m2 = np.random.uniform(Mmin, Mmax, Nsample)

    change_index = np.where(m2>m1)[0]
    m1[change_index], m2[change_index] = m2[change_index], m1[change_index]

    return m1, m2


def generate_random_twospins_bilbystyle(Nsample, a_max=0.99):
    '''
    Generate random spins for binaries in bilby convention. 
    '''
    a_1 = np.random.uniform(0, a_max, Nsample)
    a_2 = np.random.uniform(0, a_max, Nsample)
    cos_tilt_1 = np.random.uniform(-1, 1, Nsample)
    cos_tilt_2 = np.random.uniform(-1, 1,  Nsample)
    tilt_1 = np.arccos(cos_tilt_1)
    tilt_2 = np.arccos(cos_tilt_2)
    phi_12 = np.random.uniform(0, 2*np.pi, Nsample)
    phi_jl = np.random.uniform(0, 2*np.pi, Nsample)

    return a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl


def generate_random_distance(Nsample, low, high, power=3):
    '''
    PDF propto power-1. 

    Power=3 -> uniform in volume
    Power=1 -> uniform in distance
    '''
    dl = (high - low) * np.random.power(a=3, size=Nsample) + low
    return dl

def generate_random_extrinsic_angles(Nsample):
    cos_theta_jn = np.random.uniform(-1, 1, Nsample)
    theta_jn = np.arccos(cos_theta_jn)

    ra = np.random.uniform(0, np.pi, Nsample)
    sindec = np.random.uniform(0, 1, Nsample)
    dec = np.arcsin(sindec)
    psi = np.random.uniform(0, np.pi, Nsample)
    phi = np.random.uniform(0, 2*np.pi, Nsample)
    
    
    return theta_jn, ra, dec, psi, phi 


def assemble_pBNS_parameters(mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2,
    theta_jn, luminosity_distance, ra, dec, psi, phase, geocent_time):
    injection_parameters = {}
    injection_parameters['mass_1'] = mass_1
    injection_parameters['mass_2'] = mass_2
    injection_parameters['a_1'] = a_1
    injection_parameters['a_2'] = a_2
    injection_parameters['tilt_1'] = tilt_1
    injection_parameters['tilt_2'] = tilt_2
    injection_parameters['phi_12'] = phi_12
    injection_parameters['phi_jl'] = phi_jl
    injection_parameters['lambda_1'] = lambda_1
    injection_parameters['lambda_2'] = lambda_2
    injection_parameters['theta_jn'] = theta_jn
    injection_parameters['luminosity_distance'] = luminosity_distance
    injection_parameters['ra'] = ra
    injection_parameters['dec'] = dec
    injection_parameters['psi'] = psi
    injection_parameters['phase'] = phase
    injection_parameters['geocent_time'] = geocent_time

    return injection_parameters


def generate_BNS_injection_parameters(
        Nsample,
        a_max=0.1,
        d_min=10,
        d_max=100,
        d_power=2):
    mass_1, mass_2 = generate_random_component_mass(Nsample, 1.1, 3)
    a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl = generate_random_twospins_bilbystyle(Nsample, a_max=a_max)
    luminosity_distance = generate_random_distance(Nsample, low=d_max, high=d_max, power=d_power)
    theta_jn, ra, dec, psi, phase = generate_random_extrinsic_angles(Nsample)
    geocent_time = np.random.uniform(0, 3.14e7, Nsample)
    lambda_tilde = np.random.uniform(0, 1000, Nsample)
    delta_lambda_tilde = np.random.uniform(-5000, 5000, Nsample)
    lambda_1, lambda_2 = bilby.gw.conversion.lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(lambda_tilde, delta_lambda_tilde, mass_1, mass_2)

    injection_parameters_all = assemble_pBNS_parameters(mass_1, mass_2, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2, theta_jn, luminosity_distance, ra, dec, psi, phase, geocent_time)

    return injection_parameters_all
