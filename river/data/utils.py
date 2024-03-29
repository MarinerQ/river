import numpy as np
import h5py 
import bilby
import glob

#PARAMETER_NAMES_PRECESSINGBNS_BILBY = ['mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'lambda_1', 'lambda_2', 'theta_jn', 'luminosity_distance', 'ra', 'dec', 'psi', 'phase', 'geocent_time' ]
PARAMETER_NAMES_ALL_PRECESSINGBNS_BILBY = ['mass_1', 'mass_2', 'chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'lambda_1', 'lambda_2', 'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'luminosity_distance', 'ra', 'dec', 'psi', 'phase', 'geocent_time' ]
PARAMETER_NAMES_CONTEXT_PRECESSINGBNS_BILBY = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'lambda_tilde', 'delta_lambda_tilde', 'theta_jn', 'luminosity_distance', 'ra', 'dec', 'psi', 'phase', 'geocent_time' ]

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


def load_asd(data_folder, detector_names, freqs_interp=None, is_asd=True):
    asd_dict = {}
    for detname in detector_names:
        asd_dict[detname] = []
        filenames = glob.glob(f"{data_folder}/{detname}/*.txt")
        for filename in filenames:
            asd = np.loadtxt(filename)
            if freqs_interp is not None:
                asd_interp = np.interp(freqs_interp, asd[:,0], asd[:,1])
            else:
                asd_interp = asd[:,1]
            if not is_asd:
                asd_interp = asd_interp**0.5
            asd_dict[detname].append(asd_interp)
    return asd_dict

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
    dl = (high - low) * np.random.power(a=power, size=Nsample) + low
    return dl

def generate_random_extrinsic_angles(Nsample):
    cos_theta_jn = np.random.uniform(-1, 1, Nsample)
    theta_jn = np.arccos(cos_theta_jn)

    ra = np.random.uniform(0, 2*np.pi, Nsample)
    sindec = np.random.uniform(-1, 1, Nsample)
    dec = np.arcsin(sindec)
    psi = np.random.uniform(0, np.pi, Nsample)
    phi = np.random.uniform(0, 2*np.pi, Nsample)
    
    
    return theta_jn, ra, dec, psi, phi 


def assemble_pBNS_parameters(mass_1, mass_2, chirp_mass, mass_ratio, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, lambda_1, lambda_2, lambda_tilde, delta_lambda_tilde,
    theta_jn, luminosity_distance, ra, dec, psi, phase, geocent_time):
    injection_parameters = {}
    injection_parameters['mass_1'] = mass_1
    injection_parameters['mass_2'] = mass_2
    injection_parameters['chirp_mass'] = chirp_mass
    injection_parameters['mass_ratio'] = mass_ratio
    injection_parameters['a_1'] = a_1
    injection_parameters['a_2'] = a_2
    injection_parameters['tilt_1'] = tilt_1
    injection_parameters['tilt_2'] = tilt_2
    injection_parameters['phi_12'] = phi_12
    injection_parameters['phi_jl'] = phi_jl
    injection_parameters['lambda_1'] = lambda_1
    injection_parameters['lambda_2'] = lambda_2
    injection_parameters['lambda_tilde'] = lambda_tilde
    injection_parameters['delta_lambda_tilde'] = delta_lambda_tilde
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
        a_max,
        d_min,
        d_max,
        d_power,
        tc_min=-0.1,
        tc_max=0.1,
        lambda_min = 0,
        lambda_max = 5000,
        intrinsic_only=False,
        **kwargs):
    mass_1, mass_2 = generate_random_component_mass(Nsample, 1.1, 3)
    chirp_mass = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)
    mass_ratio = mass_2/mass_1
    a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl = generate_random_twospins_bilbystyle(Nsample, a_max=a_max)
    luminosity_distance = generate_random_distance(Nsample, low=d_min, high=d_max, power=d_power)
    theta_jn, ra, dec, psi, phase = generate_random_extrinsic_angles(Nsample)
    geocent_time = np.random.uniform(tc_min, tc_max, Nsample)
    #lambda_tilde = np.random.uniform(0, 1000, Nsample)
    #delta_lambda_tilde = np.random.uniform(-5000, 5000, Nsample)
    #lambda_1, lambda_2 = bilby.gw.conversion.lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(lambda_tilde, delta_lambda_tilde, mass_1, mass_2)
    lambda_1 = np.random.uniform(lambda_min, lambda_max, Nsample)
    lambda_2 = np.random.uniform(lambda_min, lambda_max, Nsample)
    lambda_tilde = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
    delta_lambda_tilde = bilby.gw.conversion.lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)

    if intrinsic_only:
        luminosity_distance = np.ones(Nsample)
        geocent_time = np.zeros(Nsample)
        ra = np.zeros(Nsample)
        dec = np.zeros(Nsample)
        psi = np.zeros(Nsample)
        
    injection_parameters_all = assemble_pBNS_parameters(mass_1, mass_2, chirp_mass, mass_ratio, 
                                                         a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, 
                                                         lambda_1, lambda_2, lambda_tilde, delta_lambda_tilde, 
                                                         theta_jn, luminosity_distance, ra, dec, psi, phase, geocent_time)

    return injection_parameters_all

def generate_2dBNS_injection_parameters(
        Nsample,
        a_max,
        d_min,
        d_max,
        d_power,
        tc_min=-0.1,
        tc_max=0.1,
        lambda_min = 0,
        lambda_max = 5000,
        **kwargs):
    mass_1, mass_2 = generate_random_component_mass(Nsample, 1.1, 3)
    chirp_mass = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1, mass_2)
    mass_ratio = mass_2/mass_1
    a_1 = np.zeros(Nsample)
    a_2 = np.zeros(Nsample)
    tilt_1  = np.zeros(Nsample) 
    tilt_2 = np.zeros(Nsample)
    phi_12 = np.zeros(Nsample)
    phi_jl = np.zeros(Nsample)

    luminosity_distance  = np.zeros(Nsample) + 40
    theta_jn = np.zeros(Nsample)
    ra = np.zeros(Nsample)
    dec = np.zeros(Nsample)
    psi = np.zeros(Nsample)
    phase  = np.zeros(Nsample)

    geocent_time = np.zeros(Nsample)
    #lambda_tilde = np.random.uniform(0, 1000, Nsample)
    #delta_lambda_tilde = np.random.uniform(-5000, 5000, Nsample)
    #lambda_1, lambda_2 = bilby.gw.conversion.lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(lambda_tilde, delta_lambda_tilde, mass_1, mass_2)
    lambda_1 = np.zeros(Nsample) + 435
    lambda_2 = np.zeros(Nsample) + 425
    lambda_tilde = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)
    delta_lambda_tilde = bilby.gw.conversion.lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2)

    injection_parameters_all = assemble_pBNS_parameters(mass_1, mass_2, chirp_mass, mass_ratio, 
                                                         a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, 
                                                         lambda_1, lambda_2, lambda_tilde, delta_lambda_tilde, 
                                                         theta_jn, luminosity_distance, ra, dec, psi, phase, geocent_time)

    return injection_parameters_all

def get_precalwf_list(folder, nbatch, file_per_batch, filename_prefix, **kwargs):
    file_list = []
    for ibatch in range(nbatch):
        for ifile in range(file_per_batch):
            filename = f"{folder}/batch{ibatch}/{filename_prefix}_{ifile}.h5"
            wf_dict = load_dict_from_hdf5(filename)
            file_list.append(wf_dict)
    return file_list