from bisect import bisect_right
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from glob import glob
from natsort import natsorted
import os
import matplotlib
def locate_idx(file=None):

    with open(file) as f:
        lines = f.readlines()

    
    missing_idx=[int(idx) for idx in re.findall(r'(\d+)\.npy', ' '.join(lines))]

    return np.array(missing_idx)

def gridplot(data,true_para:list=None,para_mins:list=None,para_maxs:list=None,para_names:list=None,figsize:tuple=(10,10)):
    # plt.rcParams.update({'font.size': figsize[0]*1.5})
    # plt.rcParams["font.family"] = "Times New Roman"
    # plt.rcParams.update({'label.fontsize': figsize[0]*1.5})
    # plt.rcParams.update({'xtick.labelsize': figsize[0]*1})
    # plt.rcParams.update({'ytick.labelsize': figsize[0]*1})

    SMALL_SIZE = figsize[0]//2
    MEDIUM_SIZE = figsize[0]
    BIGGER_SIZE = figsize[0]*1.5

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=figsize)
    v_line_fac=1.1
    para_space_fac=0.5
    tick_number=5

    N_dim=data.shape[1]
    plt.subplots_adjust(wspace=0.0,hspace=0.0)

    for i in range(N_dim):
        for j in range(i+1):

            if para_mins is not None and para_maxs is not None:
                x_min=para_mins[j]
                x_max=para_maxs[j]
            else:
                x_min=np.min(data[:,j])
                x_max=np.max(data[:,j])
                x_min, x_max = x_min-(x_max-x_min)*para_space_fac,x_max+(x_max-x_min)*para_space_fac
            
            plt.subplot(N_dim,N_dim,i*N_dim+j+1)
            if j==i:
                ax = sns.kdeplot(data[:,i])
                height = np.max(ax.lines[0].get_ydata())
                
                if true_para is not None:
                    plt.vlines(true_para[i], ymin=0, ymax=height*v_line_fac,colors='r',linestyles='dashed')

                plt.xlim(x_min,x_max)
                plt.ylim(0,height*v_line_fac)
                plt.yticks([], [])
                plt.ylabel("")
            else:
                ax = sns.kdeplot(x=data[:,j],y=data[:,i],levels=[0.05,0.32,1],shade=True,grid_size=50)
                if para_mins is not None and para_maxs is not None:
                    y_min=para_mins[i]
                    y_max=para_maxs[i]
                else:
                    y_min=np.min(data[:,i])
                    y_max=np.max(data[:,i])
                    y_min, y_max = y_min-(y_max-y_min)*para_space_fac,y_max+(y_max-y_min)*para_space_fac
                if true_para is not None:
                    plt.vlines(true_para[j], ymin=y_min, ymax=y_max,colors='r',linestyles='dashed')
                    plt.hlines(true_para[i], xmin=x_min, xmax=x_max,colors='r',linestyles='dashed')
                    plt.scatter(true_para[j],true_para[i],s=figsize[0],c='r',marker='s')
                plt.xlim(x_min,x_max)
                plt.ylim(y_min,y_max)

                if not (j==0):
                    plt.yticks([], [])
                else:

                    plt.yticks(np.linspace(y_min,y_max,tick_number)[1:-1])

                    if para_names is None:
                        plt.ylabel(f"$param_{i}$")
                    else:
                        plt.ylabel(f"${para_names[i]}$")

            if not (i==N_dim-1):
                plt.xticks([], [])
            else:
                plt.xticks(np.linspace(x_min,x_max,tick_number)[1:-1])
                if para_names is None:
                    plt.xlabel(f"$param_{j}$")
                else:
                    plt.xlabel(f"${para_names[j]}$")


def Corner_Plot(flat_samples,true_para,mode='obs'):
    plt.figure(figsize=(10,10))
    assert mode in ['obs','bright','faint'], 'mode not supported'
    ##boundary for bright
    if mode=='bright':
        para0_min=5.35
        para0_max=5.6

        para1_min=2.20
        para1_max=2.40

    #boundary for faint
    if mode=='faint':
        para0_min=4.6
        para0_max=4.8

        para1_min=1.4
        para1_max=1.55

    ##boundary for obs
    if mode=='obs':
        para0_min=4.2
        para0_max=5.7

        para1_min=1.2
        para1_max=2.4

    v_line_fac=1.1

    plt.subplots_adjust(
                        wspace=0.0,
                        hspace=0.0)

    plt.subplot(221)
    ax =sns.kdeplot(flat_samples[:,0])
    height = np.max(ax.lines[0].get_ydata())
    plt.vlines(true_para[0], ymin=0, ymax=height*v_line_fac,colors='r',linestyles='dashed')
    plt.xlim(para0_min,para0_max)
    plt.ylim(0,height*v_line_fac)
    plt.yticks([], [])
    plt.xticks([], [])

    plt.subplot(223)
    plt.vlines(true_para[0], ymin=para1_min, ymax=para1_max,colors='r',linestyles='dashed')
    plt.hlines(true_para[1], xmin=para0_min, xmax=para0_max,colors='r',linestyles='dashed')
    plt.plot(true_para[0],true_para[1],'rs',ms=1)
    # sns.kdeplot(flat_samples[:,0],flat_samples[:,1],shade_lowest=False,shade=True,levels=2)
    sns.kdeplot(x=flat_samples[:,0],y=flat_samples[:,1],levels=[0.05,0.32,1],shade=True)
    plt.xlim(para0_min,para0_max)
    plt.ylim(para1_min,para1_max)

    plt.subplot(224)
    ax2 =sns.kdeplot(flat_samples[:,1])
    height = np.max(ax2.lines[0].get_ydata())
    plt.vlines(true_para[1], ymin=0, ymax=height*v_line_fac,colors='r',linestyles='dashed')
    plt.xlim(para1_min,para1_max)
    plt.ylim(0,height*v_line_fac)
    plt.yticks([], [])

def get_xs_data(data_type='noise',para_type='faint',summary_type='st'):
    assert data_type in ['noise','noise&foreground','pure'],'wrong data type'
    assert summary_type in ['ps','st'],'wrong summary type'
    assert para_type in ['faint','bright'],'wrong param type'


    if data_type=='pure' and para_type=='faint':
        #pure ST faint

        f = open("/scratch/zxs/scripts/st/job/pos_more2230/po_0.pkl", 'rb') #l6 5678*2 9000*4
        d = pickle.load(f)
        # c_total=np.unique(d[0], axis=0, return_counts=False)
        c_total=d[0]

    if data_type=='pure' and para_type=='bright':
        #pure ST bright
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9811.pkl", 'rb') #l6 5678*2 9000*4
        d = pickle.load(f)
        # c_total=np.unique(d[0], axis=0, return_counts=False)
        c_total=d[0]

   
    #################################################### PS
    #noised PS faint 
    if data_type=='noise' and summary_type=='ps' and para_type=='faint':
        f = open("/scratch/zxs/scripts/st/orion_data/1746_10/po_9026.pkl", 'rb')
        d = pickle.load(f)
        c1=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/scripts/st/orion_data/1746_10/po_9027.pkl", 'rb')
        d = pickle.load(f)
        c2=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/scripts/st/orion_data/1746_10/po_9028.pkl", 'rb')
        d = pickle.load(f)
        c3=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/scripts/st/orion_data/1746_10/po_9029.pkl", 'rb') #not good
        d = pickle.load(f)
        c4=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/scripts/st/orion_data/1746_10/po_9030.pkl", 'rb')
        d = pickle.load(f)
        c5=np.unique(d[0], axis=0, return_counts=False)
        c_total=np.concatenate((c1,c2,c3,c4,c5),axis=0)

    #noised PS bright
    if data_type=='noise' and summary_type=='ps' and para_type=='bright':
        f = open("/scratch/zxs/scripts/st/orion_data/1746_10/po_9031.pkl", 'rb')
        d = pickle.load(f)
        c1=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/scripts/st/orion_data/1746_10/po_9032.pkl", 'rb')
        d = pickle.load(f)
        c2=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/scripts/st/orion_data/1746_10/po_9033.pkl", 'rb')
        d = pickle.load(f)
        c3=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/scripts/st/orion_data/1746_10/po_9034.pkl", 'rb')
        d = pickle.load(f)
        c4=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/scripts/st/orion_data/1746_10/po_9035.pkl", 'rb')
        d = pickle.load(f)
        c5=np.unique(d[0], axis=0, return_counts=False)
        c_total=np.concatenate((c1,c2,c3,c4,c5),axis=0)

    #################################################### ST
    #noised ST faint
    if data_type=='noise' and summary_type=='st' and para_type=='faint':
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9909.pkl", 'rb')
        d = pickle.load(f)
        c1=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9910.pkl", 'rb')
        d = pickle.load(f)
        c2=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9911.pkl", 'rb')
        d = pickle.load(f)
        c3=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9912.pkl", 'rb')
        d = pickle.load(f)
        c4=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9913.pkl", 'rb')
        d = pickle.load(f)
        c5=np.unique(d[0], axis=0, return_counts=False)
        c_total=np.concatenate((c1,c2,c3,c4,c5),axis=0)

    #noised ST bright
    if data_type=='noise' and summary_type=='st' and para_type=='bright':
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9914.pkl", 'rb')
        d = pickle.load(f)
        c1=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9915.pkl", 'rb')
        d = pickle.load(f)
        c2=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9916.pkl", 'rb')
        d = pickle.load(f)
        c3=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9917.pkl", 'rb') #not good
        d = pickle.load(f)
        c4=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9918.pkl", 'rb')
        d = pickle.load(f)
        c5=np.unique(d[0], axis=0, return_counts=False)
        c_total=np.concatenate((c1,c2,c3,c4,c5),axis=0)




    #################################################### PS
    #residual PS faint
    if data_type=='noise&foreground' and summary_type=='ps' and para_type=='faint':
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_4/po_7269.pkl", 'rb')
        d = pickle.load(f)
        c1=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_4/po_7270.pkl", 'rb')
        d = pickle.load(f)
        c2=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_4/po_7271.pkl", 'rb')
        d = pickle.load(f)
        c3=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_4/po_7272.pkl", 'rb')
        d = pickle.load(f)
        c4=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_4/po_7273.pkl", 'rb') #not good
        d = pickle.load(f)
        c5=np.unique(d[0], axis=0, return_counts=False)
        c_total=np.concatenate((c1,c2,c3,c4,c5),axis=0)

    #residual PS bright
    if data_type=='noise&foreground' and summary_type=='ps' and para_type=='bright':
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_4/po_7274.pkl", 'rb')
        d = pickle.load(f)
        c1=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_4/po_7275.pkl", 'rb')
        d = pickle.load(f)
        c2=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_4/po_7276.pkl", 'rb')
        d = pickle.load(f)
        c3=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_4/po_7277.pkl", 'rb')
        d = pickle.load(f)
        c4=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_4/po_7278.pkl", 'rb')
        d = pickle.load(f)
        c5=np.unique(d[0], axis=0, return_counts=False)
        c_total=np.concatenate((c1,c2,c3,c4,c5),axis=0)

    #################################################### ST
    #residual ST faint
    if data_type=='noise&foreground' and summary_type=='st' and para_type=='faint':
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9832.pkl", 'rb') #not good
        d = pickle.load(f)
        c1=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9833.pkl", 'rb')
        d = pickle.load(f)
        c2=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9834.pkl", 'rb')
        d = pickle.load(f)
        c3=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9835.pkl", 'rb')
        d = pickle.load(f)
        c4=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9836.pkl", 'rb')
        d = pickle.load(f)
        c5=np.unique(d[0], axis=0, return_counts=False)
        c_total=np.concatenate((c1,c2,c3,c4,c5),axis=0)

    #residual ST bright
    if data_type=='noise&foreground' and summary_type=='st' and para_type=='bright':
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9837.pkl", 'rb')
        d = pickle.load(f)
        c1=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9838.pkl", 'rb')
        d = pickle.load(f)
        c2=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9839.pkl", 'rb')
        d = pickle.load(f)
        c3=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9840.pkl", 'rb')
        d = pickle.load(f)
        c4=np.unique(d[0], axis=0, return_counts=False)
        f = open("/scratch/zxs/delfi_fast/pos_more3st2_3/po_9841.pkl", 'rb')
        d = pickle.load(f)
        c5=np.unique(d[0], axis=0, return_counts=False)
        c_total=np.concatenate((c1,c2,c3,c4,c5),axis=0)
    return c_total




def get_data(data_dirs, label_dirs):


    assert len(data_dirs)==len(label_dirs),'length of data dirs and labels dirs doesnt match '

    total_data=[]
    total_para=[]

    for dir_idx in range(len(data_dirs)):
        files=natsorted(glob(data_dirs[dir_idx] + '*.npy'))
        idxs=[re.findall('\d+',s)[-1] for s in files]
        for i in range(len(files)):
            data=np.load(files[i])
            idx=int(idxs[i])
            para=np.load(os.path.join(label_dirs[dir_idx],f'Idlt-{idx}-y.npy'))
            total_data.append(data)
            total_para.append(para)

    return np.array(total_data),np.array(total_para)
