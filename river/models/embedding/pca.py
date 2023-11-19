from sklearn.decomposition import IncrementalPCA
import numpy as np
import torch

def project_strain_data_FDAPhi(strain, psd, detector_names, ipca_gen, project=True, downsample_rate=1):
    '''
    strain: DatasetStrainFD in batches, e.g. DatasetStrainFD[0:10]
    psd: strain-like
    detector_names: DatasetStrainFD.detector_names
    ipca_gen: IPCAGenerator
    '''
    strain_amp = np.abs(strain)
    strain_phi = np.unwrap(np.angle(strain) , axis=-1)
    n_components = ipca_gen.n_components

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
            output_psd.append(psd.numpy()[:,i,:][:,::downsample_rate])

    output_amp = torch.from_numpy(np.array(output_amp))
    output_phi = torch.from_numpy(np.array(output_phi))
    output_psd = torch.from_numpy(np.array(output_psd))

    return torch.cat((output_amp, output_phi, output_psd)).movedim(0,1).float()


class IPCAGenerator():
    def __init__(self, strain_template_dict, n_components, detector_names, decomposition='exp_unwrap'):
        self.pca_dict = {}
        self.strain_template_dict = {}
        self.n_components = n_components
        self.detector_names = detector_names
        self.decomposition = decomposition
        for detname in self.detector_names:
            print(f"Generating PCA for {detname}")
            self.pca_dict[detname] = {}

            if decomposition=='exp_unwrap':
                self.pca_dict[detname]['amplitude'] = IncrementalPCA(n_components)
                self.pca_dict[detname]['amplitude'].fit(np.abs(strain_template_dict[detname]))

                self.pca_dict[detname]['phase'] = IncrementalPCA(n_components)
                self.pca_dict[detname]['phase'].fit(np.unwrap(np.angle(strain_template_dict[detname])))
            elif decomposition=='exp_wrap':
                self.pca_dict[detname]['amplitude'] = IncrementalPCA(n_components)
                self.pca_dict[detname]['amplitude'].fit(np.abs(strain_template_dict[detname]))

                self.pca_dict[detname]['phase'] = IncrementalPCA(n_components)
                self.pca_dict[detname]['phase'].fit(np.angle(strain_template_dict[detname]))
            elif decomposition=='realimag':
                self.pca_dict[detname]['real'] = IncrementalPCA(n_components)
                self.pca_dict[detname]['real'].fit(np.real(strain_template_dict[detname]))

                self.pca_dict[detname]['imag'] = IncrementalPCA(n_components)
                self.pca_dict[detname]['imag'].fit(np.imag(strain_template_dict[detname]))

    def project(self, strain, detname, part):
        pca = self.pca_dict[detname][part]
        proj = pca.transform(strain)

        return proj 

