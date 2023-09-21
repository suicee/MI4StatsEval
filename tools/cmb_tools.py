import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from functools import partial
import os
#modules for deal with CMB data generate CMB datasets
def is_2n(x):
    return x!=0 and ((x&(x-1))==0)


def get_matrix_order(patch_size):
    '''
    return matrix that contain orders of values in nested order
    '''
    lst=[]
    order=np.zeros((patch_size,patch_size),dtype=np.int_)
    lst.append(order)
    idx=iter((np.arange(patch_size**2)))
    while len(lst)!=0:
        todo_a=lst.pop()
        
        size=todo_a.shape[0]
        if size==1:
            todo_a[0,0]=int(next(idx))
        else:
            lst.append(todo_a[:size//2,size//2:])
            lst.append(todo_a[size//2:,size//2:])
            lst.append(todo_a[:size//2,:size//2])
            lst.append(todo_a[size//2:,:size//2])

        # print(len(lst))
    return order

def get_patches(map,patch_size=256,nside=512):

    n_pix=hp.nside2npix(nside)
    assert patch_size<=nside and is_2n(patch_size) and is_2n(nside)
    n_patch=int(n_pix/(patch_size**2))  

    map=hp.reorder(map,r2n=True)

    patches=np.zeros((n_patch,patch_size,patch_size))
    fll_order=get_matrix_order(patch_size)

    for loc_idx in range(n_patch):
        patches[loc_idx]=map[loc_idx*patch_size**2:(loc_idx+1)*patch_size**2][fll_order]
    
    return patches

def visualize_patches(map,patches_idx,patch_size=256,nside=512):
    assert patch_size<=nside and is_2n(patch_size) and is_2n(nside)
    map=hp.reorder(map,r2n=True)
    map_vis=map.copy()
    for loc_idx in (patches_idx):
        map_vis[loc_idx*patch_size**2:(loc_idx+1)*patch_size**2]=0
    
    return map_vis



def removeCLNorm(normalzied_CL):
    '''
    from normalized cl(camb output) to unnomralized cl(healpy input)
    '''
    index=np.arange(normalzied_CL.shape[0])
    cl=(normalzied_CL/(index*(index+1)/(np.pi*2))[:,None])
    cl=cl/1e12

    cl[np.isnan(cl)]=0
    return cl