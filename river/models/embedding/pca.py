from sklearn.decomposition import IncrementalPCA
import numpy as np
import torch



class IPCAGenerator():
    def __init__(self, n_components, detector_names, decomposition='exp_unwrap', whiten=False):
        self.pca_dict = {}
        self.n_components = n_components
        self.detector_names = detector_names
        self.decomposition = decomposition
        self.whiten = whiten
        for detname in self.detector_names:
            self.pca_dict[detname] = {}

            if decomposition=='exp_unwrap':
                self.pca_dict[detname]['amplitude'] = IncrementalPCA(n_components, whiten=whiten)
                self.pca_dict[detname]['phase'] = IncrementalPCA(n_components, whiten=whiten)
            elif decomposition=='exp_wrap':
                self.pca_dict[detname]['amplitude'] = IncrementalPCA(n_components, whiten=whiten)
                self.pca_dict[detname]['phase'] = IncrementalPCA(n_components, whiten=whiten)
            elif decomposition=='realimag':
                #self.pca_dict[detname]['real'] = IncrementalPCA(n_components, whiten=whiten)
                #self.pca_dict[detname]['imag'] = IncrementalPCA(n_components, whiten=whiten)
                self.pca_dict[detname]['realimag'] = IncrementalPCA(n_components, whiten=whiten)

    def fit(self, strain_template_dict):
        for detname in self.detector_names:
            print(f"Generating PCA for {detname}")
            if self.decomposition=='exp_unwrap':
                self.pca_dict[detname]['amplitude'].partial_fit(np.abs(strain_template_dict[detname]))
                self.pca_dict[detname]['phase'].partial_fit(np.unwrap(np.angle(strain_template_dict[detname])))
            elif self.decomposition=='exp_wrap':
                self.pca_dict[detname]['amplitude'].partial_fit(np.abs(strain_template_dict[detname]))
                self.pca_dict[detname]['phase'].partial_fit(np.angle(strain_template_dict[detname]))
            elif self.decomposition=='realimag':
                #self.pca_dict[detname]['real'].partial_fit(np.real(strain_template_dict[detname]))
                #self.pca_dict[detname]['imag'].partial_fit(np.imag(strain_template_dict[detname]))
                self.pca_dict[detname]['realimag'].partial_fit(np.real(strain_template_dict[detname]))
                self.pca_dict[detname]['realimag'].partial_fit(np.imag(strain_template_dict[detname]))

    def project(self, strain, detname, part):
        pca = self.pca_dict[detname][part]
        #proj = pca.transform(strain)
        proj = np.dot(strain, pca.components_.T)
        return proj 

class IPCAGeneratorFDWFRL():
    '''
    IPCA for FD waveform, real and imag parts. Assume real(h+), imag(h+), real(hx), imag(hx) can be compressed by one IPCA
    '''
    def __init__(self, n_components, whiten=False):
        self.pca_dict = {}
        self.n_components = n_components
        self.whiten = whiten
        self.ipca = IncrementalPCA(n_components, whiten=whiten)


    def fit(self, strain_template_dict):

        for modename in ['plus', 'cross']:
            print(f"Training IPCA for {modename}")
            #print(type(np.real(strain_template_dict[modename])))
            #print(np.real(strain_template_dict[modename]))
            #print(strain_template_dict[modename]['amplitude'])
            h = np.array(strain_template_dict[modename]['amplitude']) * np.exp(1j* np.array(strain_template_dict[modename]['phase']))
            self.ipca.partial_fit(np.real(h))
            self.ipca.partial_fit(np.imag(h))

    def project(self, strain):
        #pca = self.pca_dict[modename][part]
        #proj = pca.transform(strain)
        proj = np.dot(strain, self.ipca.components_.T)
        return proj 
