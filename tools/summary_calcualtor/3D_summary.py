from tabnanny import verbose
import numpy as np

import tools21cm as t2c
from scipy.signal import fftconvolve
from skimage import morphology
import Pk_library as PKL
import contextlib

def PowerSpectrum_fromLC_norm(dT,kbins,box_dims):
    '''
    calculate power spectrum using tools21cm
    '''
    
    ps, ks = t2c.power_spectrum_1d(dT, kbins=kbins, box_dims=box_dims)

    return  np.log10(ps*ks**3/2/np.pi**2+1e-7)


def BiSpec_FromCube_ico(cube,box_dims,threads,kbins=None,thetas=None,selection=None):
    '''
    calculate bispectrum usuing Pylians
    '''
    if kbins is None:
        kF=2*np.pi/box_dims
        #N=(np.arange(11)+1)*2 # for pure signal
        N=(np.arange(8)+1) #for signal with observational effects only use small ks
        kbins=kF*N

    if thetas is None:
        tri_angle_pi=np.array([0.05, 0.1, 0.2, 0.33, 0.4, 0.5, 0.6, 0.7, 0.85, 0.95])*np.pi     
        thetas=np.pi-tri_angle_pi

    cube=np.array(cube,dtype=np.float32)
    bs_cube=np.zeros((len(kbins),len(thetas)))
    for idx, k_to_cal in enumerate(kbins):
        k1      = k_to_cal    
        k2      = k_to_cal  

        # compute bispectrum
        try:
            with contextlib.redirect_stdout(None):
                BBk = PKL.Bk(cube,box_dims, k1, k2, thetas, None, threads)
            
            bs_cube[idx]=normalized_BS(BBk)

        except ZeroDivisionError:
            bs_cube[idx]=0
    #use selection array to choose triangle that satisfy our constrain
    return bs_cube.flat[selection]


def get_filter(dim=100,kbins=None,thetas=None):
    '''
    get k pairs satisfying kF<k<kmax
    '''

    re=[]
    kF=2*np.pi/dim
    for k in kbins:
        for theta in thetas:
            # calcuate the length of the third side
            k3=np.sqrt((k*np.sin(theta))**2 + (k*np.cos(theta)+k)**2)
            
            if kF<k3<kbins[-1]:
                re.append(k3)
            else:
                re.append(0)
    
    return np.array(re)!=0





def normalized_BS(BBk):
    '''
    normalize bs following Watkinson et al 2019
    '''
    bs=BBk.B
    ps=BBk.Pk
    ks=BBk.k
    
    k1,Pk1=ks[0],ps[0]
    k2,Pk2=ks[1],ps[1]
    k3s,Pk3s=ks[2:],ps[2:]
    
    normal_fac=np.sqrt((Pk1*Pk2*Pk3s)/(k1*k2*k3s))
    bs_norm=bs/normal_fac
    
    return bs_norm