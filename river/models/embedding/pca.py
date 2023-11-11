from sklearn.decomposition import IncrementalPCA
import numpy as np

class IPCAGenerator():
    def __init__(self, strain_template_dict, n_components, detector_names):
        self.pca_dict = {}
        self.strain_template_dict = {}
        self.n_components = n_components
        self.detector_names = detector_names
        for detname in self.detector_names:
            print(f"Generating PCA for {detname}")
            self.pca_dict[detname] = {}

            self.pca_dict[detname]['amplitude'] = IncrementalPCA(n_components)
            self.pca_dict[detname]['amplitude'].fit(np.abs(strain_template_dict[detname]))

            self.pca_dict[detname]['phase'] = IncrementalPCA(n_components)
            self.pca_dict[detname]['phase'].fit(np.angle(strain_template_dict[detname]))

    def project(self, strain, detname, part):
        pca = self.pca_dict[detname][part]
        proj = pca.transform(strain)

        return proj 

