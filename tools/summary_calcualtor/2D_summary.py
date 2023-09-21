import numpy as np
import torch
import torch.fft


# the ST coefs is calulated using kymatio:
# from kymatio.numpy import Scattering2D
# S = Scattering2D(J=9,L=4, shape=(512, 512),out_type='list')
# st= S.scattering(data)
# we use this function to further compress the output(average over different angles)
def st_to_compact_torch(st_raw,J,L):

    S0=torch.tensor(torch.mean(st_raw[0]['coef'],dim=(1,2)))[:,None]
    N=len(S0)
    S1=torch.zeros((N,J,L))
    for j in range(1,J*L+1):
        j1,l1=(*st_raw[j]['j'],*st_raw[j]['theta'])
        S1[:,j1,l1]=torch.mean(st_raw[j]['coef'],dim=(1,2))

    #angle average
    S1=S1.mean(dim=2)

        
    S2=torch.zeros((N,J,J,L,L))
    for j in range(J*L+1,len(st_raw)):
        j1,j2,l1,l2=(*st_raw[j]['j'],*st_raw[j]['theta'])
        S2[:,j1,j2,l1,l2]=torch.mean(st_raw[j]['coef'],dim=(1,2))

    S2=torch.mean(S2,dim=(3,4))

    tridx=torch.triu_indices(J,J,offset=1)

    S2=S2[:,tridx[0],tridx[1]]

    return torch.hstack((S0,S1,S2))

def st_to_compact(st_raw,J,L):


    S0=np.array(np.mean(st_raw[0]['coef']))

    S1=np.zeros((J,L))
    for j in range(1,J*L+1):
        entry=(*st_raw[j]['j'],*st_raw[j]['theta'])
        S1[entry]=np.mean(st_raw[j]['coef'])

    #angle average
    S1=S1.mean(axis=1)

        
    S2=np.zeros((J,J,L,L))
    for j in range(J*L+1,len(st_raw)):
        entry=(*st_raw[j]['j'],*st_raw[j]['theta'])
        S2[entry]=np.mean(st_raw[j]['coef'])

    S2=np.mean(S2,axis=(2,3))

    tridx=np.triu_indices(J,k=1)
    S2=S2[tridx]
    
    return np.hstack((S0,S1,S2))



# the 2D ps and bs codes are modifed from 
# https://github.com/SihaoCheng/scattering_transform/blob/master/scattering/polyspectra_calculators.py



# power spectrum computer
def get_power_spectrum(image, k_range=None, bins=None, bin_type='log', device='gpu'):
    '''
    get the power spectrum of a given image
    '''
    if type(image) == np.ndarray:
        image = torch.from_numpy(image)    
    if type(k_range) == np.ndarray:
        k_range = torch.from_numpy(k_range) 
    if not torch.cuda.is_available(): device='cpu'
    if device=='gpu':
        image = image.cuda()
            
    M, N = image.shape[-2:]
    modulus = torch.fft.fftn(image, dim=(-2,-1), norm='ortho').abs()
    
    modulus = torch.cat(
        ( torch.cat(( modulus[..., M//2:, N//2:], modulus[..., :M//2, N//2:] ), -2),
          torch.cat(( modulus[..., M//2:, :N//2], modulus[..., :M//2, :N//2] ), -2)
        ),-1)
    
    X = torch.arange(M)[:,None]
    Y = torch.arange(N)[None,:]
    Xgrid = X+Y*0
    Ygrid = X*0+Y
    k = ((Xgrid - M/2)**2 + (Ygrid - N/2)**2)**0.5
    
    if k_range is None:
        if bin_type=='linear':
            k_range = torch.linspace(1, M/2*1.415, bins+1) # linear binning
        if bin_type=='log':
            k_range = torch.logspace(0, np.log10(M/2*1.415), bins+1) # log binning

    power_spectrum = torch.zeros(len(image), len(k_range)-1, dtype=image.dtype)
    if device=='gpu':
        k = k.cuda()
        k_range = k_range.cuda()
        power_spectrum = power_spectrum.cuda()

    for i in range(len(k_range)-1):
        select = (k > k_range[i]) * (k <= k_range[i+1])
        power_spectrum[:,i] = ((modulus**2*select[None,...]).sum((-2,-1))/select.sum()).log()
    return power_spectrum, k_range


class Bispectrum_Calculator(object):
    def __init__(self, M, N, k_range=None, bins=None, bin_type='log', device='gpu', edge=0):
        if not torch.cuda.is_available(): device='cpu'
        # k_range in unit of pixel in Fourier space
        self.device = device
        if k_range is None:
            if bin_type=='linear':
                k_range = np.linspace(0, M/2*1.415, bins+1) # linear binning
            if bin_type=='log':
                k_range = np.logspace(0, np.log10(M/2*1.415), bins+1) # log binning
#         k_range = np.concatenate((np.array([0]), k_range), axis=0)
        self.k_range = k_range
        self.M = M
        self.N = N
        self.bin_type = bin_type
        X = torch.arange(M)[:,None]
        Y = torch.arange(N)[None,:]
        Xgrid = X+Y*0
        Ygrid = X*0+Y
        d = ((X-M//2)**2+(Y-N//2)**2)**0.5
        
        self.k_filters = torch.zeros((len(k_range)-1, M, N), dtype=bool)
        for i in range(len(k_range)-1):
            # k-selection function 
            self.k_filters[i,:,:] = torch.fft.ifftshift((d<=k_range[i+1]) * (d>k_range[i]))
        #fourier transform k-selection function to real space
        self.k_filters_if = torch.fft.ifftn(self.k_filters, dim=(-2,-1), norm='ortho')
        
        self.select = torch.zeros(
            (len(self.k_range)-1, len(self.k_range)-1, len(self.k_range)-1), 
            dtype=bool
        )
        self.B_ref_array = torch.zeros(
            (len(self.k_range)-1, len(self.k_range)-1, len(self.k_range)-1),
            dtype=torch.float32
        )
        self.mask_xy = (Xgrid >= edge) * (Xgrid <= M-edge-1) * (Ygrid >= edge) * (Ygrid <= N-edge-1)
        for i1 in range(len(self.k_range)-1):
            for i2 in range(i1,len(self.k_range)-1):
                for i3 in range(i2,len(self.k_range)-1):
                    # if self.k_range[i1+1] + self.k_range[i2+1] > self.k_range[i3] + 0.5:
                    #change to select isosceles triangle
                    if (self.k_range[i1+1] + self.k_range[i2+1] > self.k_range[i3] + 0.5) and (self.k_range[i1+1] == self.k_range[i2+1]):
                        self.select[i1, i2, i3] = True
                        self.B_ref_array[i1, i2, i3] = (
                            self.k_filters_if[i1] * self.k_filters_if[i2] * self.k_filters_if[i3]
                        ).sum().real
        if device=='gpu':
            self.k_filters = self.k_filters.cuda()
            self.k_filters_if = self.k_filters_if.cuda()
            self.select = self.select.cuda()
            self.B_ref_array = self.B_ref_array.cuda()
            self.mask_xy = self.mask_xy.cuda()
    
    def forward(self, image, normalization='both'):
        '''
        normalization is one of 'image', 'dirac', or 'both'
        '''
        if type(image) == np.ndarray:
            image = torch.from_numpy(image)

        B_array = torch.zeros(
            (len(image), len(self.k_range)-1, len(self.k_range)-1, len(self.k_range)-1), 
            dtype=image.dtype
        )
        
        if self.device=='gpu':
            image   = image.cuda()
            B_array = B_array.cuda()
        
        image_f = torch.fft.fftn(image, dim=(-2,-1), norm='ortho')
        conv = torch.fft.ifftn(image_f[None,...] * self.k_filters[:,None,...], dim=(-2,-1), norm='ortho')
        P_bin = (conv.abs()**2 * self.mask_xy[None,...]).sum((-2,-1)) / (self.k_filters_if[:,None,...].abs()**2).sum((-2,-1)) \
            / self.mask_xy.sum() * self.M * self.N
        for i1 in range(len(self.k_range)-1):
            for i2 in range(i1,len(self.k_range)-1):
                for i3 in range(i2,len(self.k_range)-1):
                    # if self.k_range[i1+1] + self.k_range[i2+1] > self.k_range[i3] + 0.5:
                    #change to select isosceles triangle
                    if (self.k_range[i1+1] + self.k_range[i2+1] > self.k_range[i3] + 0.5) and (self.k_range[i1+1] == self.k_range[i2+1]):
                        B = (
                            conv[i1] * conv[i2] * conv[i3] * self.mask_xy[None,...]).sum((-2,-1)
                        ).real / self.mask_xy.sum() * self.M * self.N
                        # B = (conv[i1] * conv[i2] * conv[i3]).sum((-2,-1)).real
                        if normalization=='image':
                            B_array[:, i1, i2, i3] = B / (P_bin[i1] * P_bin[i2] * P_bin[i3])**0.5
                        elif normalization=='dirac':
                            B_array[:, i1, i2, i3] = B / self.B_ref_array[i1, i2, i3]
                        elif normalization=='both':
                            B_array[:, i1, i2, i3] = B / (P_bin[i1] * P_bin[i2] * P_bin[i3])**0.5 / self.B_ref_array[i1, i2, i3]
        return B_array.reshape(len(image), (len(self.k_range)-1)**3)[:,self.select.flatten()]