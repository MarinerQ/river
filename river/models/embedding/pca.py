from sklearn.decomposition import IncrementalPCA
import numpy as np
import torch



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

