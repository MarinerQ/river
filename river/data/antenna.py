import numpy as np
import bilby 
import sealgw
import lal



class GWAntennaOnCPU():
    def __init__(self, detector_names):
        self.ifos = sealgw.simulation.sealinterferometers.SealInterferometerList(detector_names)

    def getgha(self, gpstime, ra):
        # Greenwich hour angle of source (radians).
        gha = np.zeros_like(gpstime) - ra
        for i,gpst in enumerate(gpstime):
            gha[i] += bilby.gw.utils.greenwich_mean_sidereal_time(gpst)
        return gha
    
    def response(self, ra, dec, psi, gpstime):
        bs = ra.shape[0]
        
        X = np.zeros((bs, 3))
        Y = np.zeros((bs, 3))

        
        gha = self.getgha(gpstime, ra)

        cosgha = np.cos(gha)
        singha = np.sin(gha)
        cosdec = np.cos(dec)
        sindec = np.sin(dec)
        cospsi = np.cos(psi)
        sinpsi = np.sin(psi)
    
        X[:,0] = -cospsi * singha - sinpsi * cosgha * sindec
        X[:,1] = -cospsi * cosgha + sinpsi * singha * sindec
        X[:,2] =  sinpsi * cosdec

        Y[:,0] =  sinpsi * singha - cospsi * cosgha * sindec
        Y[:,1] =  sinpsi * cosgha + cospsi * singha * sindec
        Y[:,2] =  cospsi * cosdec
        
        resp_dict = {}
        for det in self.ifos:
            D = det.detector_tensor
            fp = np.einsum('ij,jk,ik->i', X, D, X) - np.einsum('ij,jk,ik->i', Y, D, Y)
            fc = np.einsum('ij,jk,ik->i', X, D, Y) + np.einsum('ij,jk,ik->i', Y, D, X)
            resp_dict[det.name] = (fp, fc)

        return resp_dict
    
    def time_delay_from_geocenter(self, ra, dec, gpstime):
        bs = ra.shape[0]
        gha = self.getgha(gpstime, ra)
        
        cosgha = np.cos(gha)
        singha = np.sin(gha)
        cosdec = np.cos(dec)
        sindec = np.sin(dec)
        
        wavevector = np.zeros((bs, 3))
        wavevector[:,0],wavevector[:,1],wavevector[:,2]  = \
            -cosgha*cosdec, cosdec*singha, -sindec
        
        dt_dict = {}
        for det in self.ifos:
            loc = det.vertex
            dt = np.einsum('ij,j->i', wavevector, loc) / 299792458
            dt_dict[det.name] = dt
        
        return dt_dict

    def resp_and_dt(self, ra, dec, gpstime, psi):
        bs = ra.shape[0]        
        X = np.zeros((bs, 3))
        Y = np.zeros((bs, 3))

        gha = self.getgha(gpstime, ra)

        cosgha = np.cos(gha)
        singha = np.sin(gha)
        cosdec = np.cos(dec)
        sindec = np.sin(dec)
        cospsi = np.cos(psi)
        sinpsi = np.sin(psi)
        
        X[:,0] = -cospsi * singha - sinpsi * cosgha * sindec
        X[:,1] = -cospsi * cosgha + sinpsi * singha * sindec
        X[:,2] =  sinpsi * cosdec

        Y[:,0] =  sinpsi * singha - cospsi * cosgha * sindec
        Y[:,1] =  sinpsi * cosgha + cospsi * singha * sindec
        Y[:,2] =  cospsi * cosdec

        
        
        wavevector = np.zeros((bs, 3))
        wavevector[:,0],wavevector[:,1],wavevector[:,2]  = \
            -cosgha*cosdec, cosdec*singha, -sindec
        
        
        resp_and_dt_dict = {}
        for det in self.ifos:
            loc = det.vertex
            D = det.detector_tensor
            
            fp = np.einsum('ij,jk,ik->i', X, D, X) - np.einsum('ij,jk,ik->i', Y, D, Y)
            fc = np.einsum('ij,jk,ik->i', X, D, Y) + np.einsum('ij,jk,ik->i', Y, D, X)
            dt = np.einsum('ij,j->i', wavevector, loc) / 299792458
            resp_and_dt_dict[det.name] = [fp,fc,dt]
        
        return resp_and_dt_dict