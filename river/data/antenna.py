import numpy as np
import bilby 
import sealgw
import lal
import torch


class GWAntennaOnCPU():
    def __init__(self, detector_names, gmst_fit = False, gps_start=None, gps_end=None):
        self.ifos = sealgw.simulation.sealinterferometers.SealInterferometerList(detector_names)
        self.gmst_fit = gmst_fit 
        if gmst_fit:
            self.fit_linear_approx_gmst(gps_start, gps_end)

    def fit_linear_approx_gmst(self, gps_start, gps_end):
        t = np.linspace(gps_start, gps_end, 100000)
        gmst=[]
        for tt in t:
            gmst.append(bilby.gw.utils.greenwich_mean_sidereal_time(tt))
        gmst = np.array(gmst)
        self.k, self.b = np.polyfit(t, gmst, 1)
        self.gmst_fit = True

        # validate the accuracy of linear fit 
        #t_test = np.linspace(gps_start, gps_end, 13417)
        t_test = t
        gmst_fit = self.k * t_test + self.b
        #gmst_true = []
        #for tt in t_test:
        #    gmst_true.append(bilby.gw.utils.greenwich_mean_sidereal_time(tt))
        #gmst_true = np.array(gmst_true)
        gmst_true = gmst 
        err = np.max(np.abs(gmst_true - gmst_fit))
        print(f"Max error in linear fit of gmst: {err}")
        

    def getgha(self, gpstime, ra):
        # Greenwich hour angle of source (radians).
        gha = np.zeros_like(gpstime) - ra

        if self.gmst_fit:
            gha += (self.k * gpstime + self.b)
        else:
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



'''
class GWAntennaOnGPU():
    def __init__(self, detector_names, gmst_fit=False, gps_start=None, gps_end=None, device='cuda', dtype=torch.float64):
        self.ifos = sealgw.simulation.sealinterferometers.SealInterferometerList(detector_names)
        self.gmst_fit = gmst_fit
        self.device = device
        self.dtype = dtype
        if gmst_fit:
            self.fit_linear_approx_gmst(gps_start, gps_end)

    def fit_linear_approx_gmst(self, gps_start, gps_end):
        t = np.linspace(gps_start, gps_end, 100000).astype(np.float64)
        gmst = []
        for tt in t:
            gmst.append(bilby.gw.utils.greenwich_mean_sidereal_time(tt))
        gmst = np.array(gmst)
        self.k, self.b = np.polyfit(t, gmst, 1)
        self.gmst_fit = True

        # validate the accuracy of linear fit
        t_test = t
        gmst_fit = self.k * t_test + self.b
        gmst_true = gmst
        err = np.max(np.abs(gmst_true - gmst_fit))
        print(f"Max error in linear fit of gmst: {err}")
        
        self.k = torch.tensor(self.k).to(self.device).type(self.dtype)
        self.b = torch.tensor(self.b).to(self.device).type(self.dtype)

    def getgha(self, gpstime, ra):
        # Greenwich hour angle of source (radians).
        gha = torch.zeros_like(gpstime, device=self.device, dtype=self.dtype) - ra

        if self.gmst_fit:
            gha += (self.k * gpstime + self.b)
        else:
            for i, gpst in enumerate(gpstime):
                gha[i] += bilby.gw.utils.greenwich_mean_sidereal_time(gpst)
        return gha

    def response(self, ra, dec, psi, gpstime):
        bs = ra.shape[0]

        X = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)
        Y = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)

        gha = self.getgha(gpstime, ra)

        cosgha = torch.cos(gha)
        singha = torch.sin(gha)
        cosdec = torch.cos(dec)
        sindec = torch.sin(dec)
        cospsi = torch.cos(psi)
        sinpsi = torch.sin(psi)

        X[:, 0] = -cospsi * singha - sinpsi * cosgha * sindec
        X[:, 1] = -cospsi * cosgha + sinpsi * singha * sindec
        X[:, 2] = sinpsi * cosdec

        Y[:, 0] = sinpsi * singha - cospsi * cosgha * sindec
        Y[:, 1] = sinpsi * cosgha + cospsi * singha * sindec
        Y[:, 2] = cospsi * cosdec

        resp_dict = {}
        for det in self.ifos:
            D = torch.from_numpy(det.detector_tensor).to(self.device).type(self.dtype)
            fp = torch.einsum('ij,jk,ik->i', X, D, X) - torch.einsum('ij,jk,ik->i', Y, D, Y)
            fc = torch.einsum('ij,jk,ik->i', X, D, Y) + torch.einsum('ij,jk,ik->i', Y, D, X)
            resp_dict[det.name] = (fp, fc)

        return resp_dict

    def time_delay_from_geocenter(self, ra, dec, gpstime):
        bs = ra.shape[0]
        gha = self.getgha(gpstime, ra)

        cosgha = torch.cos(gha)
        singha = torch.sin(gha)
        cosdec = torch.cos(dec)
        sindec = torch.sin(dec)

        wavevector = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)
        wavevector[:, 0], wavevector[:, 1], wavevector[:, 2] = \
            -cosgha * cosdec, cosdec * singha, -sindec

        dt_dict = {}
        for det in self.ifos:
            loc = torch.from_numpy(det.vertex).to(self.device).type(self.dtype)
            dt = torch.einsum('ij,j->i', wavevector, loc) / 299792458
            dt_dict[det.name] = dt

        return dt_dict

    def resp_and_dt(self, ra, dec, gpstime, psi):
        bs = ra.shape[0]
        X = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)
        Y = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)

        gha = self.getgha(gpstime, ra)

        cosgha = torch.cos(gha)
        singha = torch.sin(gha)
        cosdec = torch.cos(dec)
        sindec = torch.sin(dec)
        cospsi = torch.cos(psi)
        sinpsi = torch.sin(psi)

        X[:, 0] = -cospsi * singha - sinpsi * cosgha * sindec
        X[:, 1] = -cospsi * cosgha + sinpsi * singha * sindec
        X[:, 2] = sinpsi * cosdec

        Y[:, 0] = sinpsi * singha - cospsi * cosgha * sindec
        Y[:, 1] = sinpsi * cosgha + cospsi * singha * sindec
        Y[:, 2] = cospsi * cosdec

        wavevector = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)
        wavevector[:, 0], wavevector[:, 1], wavevector[:, 2] = \
            -cosgha * cosdec, cosdec * singha, -sindec


        wavevector = wavevector.detach()
        resp_and_dt_dict = {}
        for det in self.ifos:
            loc = torch.from_numpy(det.vertex).to(self.device).type(self.dtype)
            D = torch.from_numpy(det.detector_tensor).to(self.device).type(self.dtype)
            
            fp = torch.einsum('ij,jk,ik->i', X, D, X) - torch.einsum('ij,jk,ik->i', Y, D, Y)
            fc = torch.einsum('ij,jk,ik->i', X, D, Y) + torch.einsum('ij,jk,ik->i', Y, D, X)
            dt = torch.einsum('ij,j->i', wavevector, loc) / 299792458.0
            resp_and_dt_dict[det.name] = [fp, fc, dt]

        return resp_and_dt_dict
'''

class GWAntennaOnGPU():
    def __init__(self, detector_names, gmst_fit=False, gps_start=None, gps_end=None, device='cuda', dtype=torch.float64):
        self.ifos = sealgw.simulation.sealinterferometers.SealInterferometerList(detector_names)
        self.gmst_fit = gmst_fit
        self.device = device
        self.dtype = dtype
        if gmst_fit:
            self.fit_linear_approx_gmst(gps_start, gps_end)
        self.locs = torch.stack([torch.from_numpy(det.vertex).to(self.device).type(self.dtype) for det in self.ifos])
        self.Ds = torch.stack([torch.from_numpy(det.detector_tensor).to(self.device).type(self.dtype) for det in self.ifos])

    def fit_linear_approx_gmst(self, gps_start, gps_end):
        t = np.linspace(gps_start, gps_end, 100000).astype(np.float64)
        gmst = []
        for tt in t:
            gmst.append(bilby.gw.utils.greenwich_mean_sidereal_time(tt))
        gmst = np.array(gmst)
        self.k, self.b = np.polyfit(t, gmst, 1)
        self.gmst_fit = True

        # validate the accuracy of linear fit
        t_test = t
        gmst_fit = self.k * t_test + self.b
        gmst_true = gmst
        err = np.max(np.abs(gmst_true - gmst_fit))
        print(f"Max error in linear fit of gmst: {err}")
        
        self.k = torch.tensor(self.k).to(self.device).type(self.dtype)
        self.b = torch.tensor(self.b).to(self.device).type(self.dtype)

    def getgha(self, gpstime, ra):
        # Greenwich hour angle of source (radians).
        # this gha is measured clockwise, while alpha Eq.6 in arXiv:gr-qc/9804014 is measured counter-clockwise.
        # also, phi in Eq.B4 in arXiv:gr-qc/0008066 is also counter-clockwise.
        # therefore, there is an opposite sign when it comes to sin(gha), but cos(gha) is the same.
        gha = torch.zeros_like(gpstime, device=self.device, dtype=self.dtype) - ra

        if self.gmst_fit:
            gha += (self.k * gpstime + self.b)
        else:
            for i, gpst in enumerate(gpstime):
                gha[i] += bilby.gw.utils.greenwich_mean_sidereal_time(gpst)
        return gha

    def response(self, ra, dec, gpstime, psi):
        bs = ra.shape[0]
        X = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)
        Y = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)

        gha = self.getgha(gpstime, ra)

        cosgha = torch.cos(gha)
        singha = torch.sin(gha)
        cosdec = torch.cos(dec)
        sindec = torch.sin(dec)
        cospsi = torch.cos(psi)
        sinpsi = torch.sin(psi)

        X[:, 0] = -cospsi * singha - sinpsi * cosgha * sindec
        X[:, 1] = -cospsi * cosgha + sinpsi * singha * sindec
        X[:, 2] = sinpsi * cosdec

        Y[:, 0] = sinpsi * singha - cospsi * cosgha * sindec
        Y[:, 1] = sinpsi * cosgha + cospsi * singha * sindec
        Y[:, 2] = cospsi * cosdec

        fps = torch.einsum('ij,ajk,ik->ia', X, self.Ds, X) - torch.einsum('ij,ajk,ik->ia', Y, self.Ds, Y)
        fcs = torch.einsum('ij,ajk,ik->ia', X, self.Ds, Y) + torch.einsum('ij,ajk,ik->ia', Y, self.Ds, X)

        response_dict = {det.name: [fp, fc] for det, fp, fc in zip(self.ifos, fps.T, fcs.T)}

        return response_dict

    def time_delay_from_geocenter(self, ra, dec, gpstime):
        bs = ra.shape[0]
        gha = self.getgha(gpstime, ra)

        cosgha = torch.cos(gha)
        singha = torch.sin(gha)
        cosdec = torch.cos(dec)
        sindec = torch.sin(dec)

        wavevector = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)
        wavevector[:, 0], wavevector[:, 1], wavevector[:, 2] = -cosgha * cosdec, cosdec * singha, -sindec

        dts = torch.einsum('ij,aj->ia', wavevector, self.locs) / 299792458.0

        time_delay_dict = {det.name: dt for det, dt in zip(self.ifos, dts.T)}

        return time_delay_dict

    def resp_and_dt(self, ra, dec, gpstime, psi):
        bs = ra.shape[0]
        X = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)
        Y = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)

        gha = self.getgha(gpstime, ra)

        cosgha = torch.cos(gha)
        singha = torch.sin(gha)
        cosdec = torch.cos(dec)
        sindec = torch.sin(dec)
        cospsi = torch.cos(psi)
        sinpsi = torch.sin(psi)

        X[:, 0] = -cospsi * singha - sinpsi * cosgha * sindec
        X[:, 1] = -cospsi * cosgha + sinpsi * singha * sindec
        X[:, 2] = sinpsi * cosdec

        Y[:, 0] = sinpsi * singha - cospsi * cosgha * sindec
        Y[:, 1] = sinpsi * cosgha + cospsi * singha * sindec
        Y[:, 2] = cospsi * cosdec

        wavevector = torch.zeros((bs, 3), device=self.device, dtype=self.dtype)
        wavevector[:, 0], wavevector[:, 1], wavevector[:, 2] = \
            -cosgha * cosdec, cosdec * singha, -sindec

        wavevector = wavevector.detach()

        fps = torch.einsum('ij,ajk,ik->ia', X, self.Ds, X) - torch.einsum('ij,ajk,ik->ia', Y, self.Ds, Y)
        fcs = torch.einsum('ij,ajk,ik->ia', X, self.Ds, Y) + torch.einsum('ij,ajk,ik->ia', Y, self.Ds, X)
        dts = torch.einsum('ij,aj->ia', wavevector, self.locs) / 299792458.0

        resp_and_dt_dict = {det.name: [fp, fc, dt] for det, fp, fc, dt in zip(self.ifos, fps.T, fcs.T, dts.T)}

        return resp_and_dt_dict
