# basic data treatment libraries
import numpy as np, pandas as pd, scipy as sp, re, os, pathlib, glob
from datetime import datetime, timedelta
from time import mktime
import h5py
import itertools
from decimal import Decimal

# scipy signal treatment
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks
from scipy import stats

# others
from collections import OrderedDict

# basic plotting libraries
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import proj3d
from matplotlib.collections import PolyCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib import gridspec
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator

# user libraries
from .aux.aux_plot import *

# --------------------------------------------------- SVD analysis and plots ---------------------------------------------------
def get_intensity_arr(df,intensity_col):
    if len(intensity_col)==0:
        try:
            intensity_col=get_intensity_col(df)
        except:
            raise ValueError("Please define a proper column for the intensity.")
    elif intensity_col not in df.columns:
        raise ValueError("Please define a proper column for the intensity.")
    int_arr = np.array(df[intensity_col].tolist())
    return int_arr

def display_all(df,intensity_col='',colormap='viridis'):
    energy=df['shifted energy'][0]
    arr=get_svd_arr(df,intensity_col)
    fig=plt.figure(figsize=(8,5))
    cm=plt.get_cmap(colormap)
    colors = cm(np.linspace(0, 1, arr.shape[0]))
    for i,line in enumerate(arr):
        plt.plot(energy,line, color=colors[-i])
    return fig

def plot_variance(df,n_pc=20,intensity_col=''):
    arr=get_intensity_arr(df,intensity_col)
    svd_arr = sp.linalg.svd(arr)
    fig, axs = plt.subplots(2,figsize=(8,7))
    axs[0].plot(np.arange(1,n_pc+1),100*svd_arr[1][:n_pc]**2/sum(svd_arr[1]**2),marker="o",fillstyle='none')
    axs[0].set_yscale('log')
    axs[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[0].tick_params(axis='both', labelsize=13)
    axs[0].set_xlabel("Component number",fontsize=13)
    axs[0].set_title("Variance plot in log scale",fontweight='bold',fontsize=14)
    axs[1].plot(np.arange(1,n_pc+1),100*svd_arr[1][:n_pc]**2/sum(svd_arr[1]**2),marker="o",fillstyle='none')
    axs[1].xaxis.set_major_locator(ticker.MultipleLocator(2))
    axs[1].tick_params(axis='both', labelsize=13)
    axs[1].set_xlabel("Component number",fontsize=13)
    axs[1].set_title("Variance plot",fontweight='bold',fontsize=14)
    plt.subplots_adjust(hspace=0.5)
    return fig

def plot_scree(df,n_pc=20,intensity_col=''):
    arr=get_intensity_arr(df,intensity_col)
    svd_arr = sp.linalg.svd(arr)
    fig=plt.figure(figsize=(8,5))
    plt.plot(np.arange(1,n_pc+1),svd_arr[1][:n_pc],marker="D",fillstyle='none')
    plt.yscale('log')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.tick_params(axis='both', labelsize=13)
    plt.xlabel("Component number",fontsize=13)
    plt.title("Scree plot in log scale",fontweight='bold',fontsize=14)
    return fig

def plot_PCA_eigs(df,n_pc=20,intensity_col=''):
    arr=get_intensity_arr(df,intensity_col)
    svd_arr = sp.linalg.svd(arr)
    if n_pc>6:
        n_pc=6
    gs = gridspec.GridSpec(2, 1)
    fig1,ax1 = plt.subplots(figsize=(15,5))
    for i, (eigenvalue, PC_col) in enumerate(zip(svd_arr[1][:n_pc],np.transpose(svd_arr[0])[:n_pc])):
        ax1.plot(np.arange(1,arr.shape[0]+1), PC_col, color="rbgkmcy"[i], label='PCA%1d eig. =%6.4f' %(i+1,eigenvalue))

    ax1.legend(bbox_to_anchor=(0.9, 0.7))
    ax1.set_xlabel("Spectrum number",fontsize=13)
    ax1.set_title("u as a function of spectra number",fontweight='bold',fontsize=14)

    fig2, axs = plt.subplots(nrows=2, ncols=3)
    fig2.set_size_inches(20, 10)
    fig2.subplots_adjust(wspace=0.2)
    fig2.subplots_adjust(hspace=0.5)

    for i, (eigenvalue, PC_col, ax) in enumerate(zip(svd_arr[1][:n_pc], np.transpose(svd_arr[0])[:n_pc], axs.flatten())):
        ax.plot(np.arange(1,arr.shape[0]+1), PC_col, color="rbgkmcy"[i], label='PCA%1d eig. =%6.4f' %(i+1,eigenvalue))
        ax.set_xlabel("Spectrum number",fontsize=13)
        ax.set_title("PCA%1d Evolution" %(i+1),fontweight='bold',fontsize=14)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend(fontsize=13)
    return fig1, fig2

def plot_PCA_vars(df,n_pc=20,intensity_col=''):
    arr=get_intensity_arr(df,intensity_col)
    svd_arr = sp.linalg.svd(arr)
    if n_pc>6:
        n_pc=6
    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.set_size_inches(20, 10)
    fig.subplots_adjust(wspace=0.2)
    fig.subplots_adjust(hspace=0.5)
    sum_eigenvalues=sum(svd_arr[1]**2)

    for i, (eigenvalue, PC_row, ax) in enumerate(zip(svd_arr[1][:n_pc], svd_arr[2][:n_pc], axs.flatten())):
        ax.plot(np.arange(1,arr.shape[1]+1), PC_row)
        ax.set_title("PCA %1d var. = %.6g" %(i+1,100*eigenvalue**2/sum_eigenvalues),fontweight='bold',fontsize=14)
        ax.tick_params(axis='both', labelsize=13)
    return fig

def plot_scores_2D(df,n_pc=4,intensity_col=''):
    arr=get_intensity_arr(df,intensity_col)
    svd_arr = sp.linalg.svd(arr)
    if n_pc==4:
        fig, axs = plt.subplots(nrows=2, ncols=3)
        fig.set_size_inches(11, 7)
        fig.subplots_adjust(wspace=0.3)
        fig.subplots_adjust(hspace=0.3)
        flat = axs.flatten()
        k=0
        for i, (eigenvalue, PC_col) in enumerate(zip(svd_arr[1][:n_pc], np.transpose(svd_arr[0])[:n_pc])):
            for j in range(i+1,n_pc):
                flat[k].scatter(PC_col*eigenvalue, np.transpose(svd_arr[0])[j]*svd_arr[1][j], color="r", s=10)
                flat[k].set_title("PCA"+str(i+1)+" vs PCA"+str(j+1),fontweight='bold',fontsize=14)
                flat[k].tick_params(axis='both', labelsize=11)
                k+=1
    elif n_pc==5:
        fig, axs = plt.subplots(nrows=3, ncols=4)
        fig.set_size_inches(15, 10)
        fig.subplots_adjust(wspace=0.3)
        fig.subplots_adjust(hspace=0.3)
        flat = axs.flatten()
        k=0
        for i, (eigenvalue, PC_col) in enumerate(zip(svd_arr[1][:n_pc], np.transpose(svd_arr[0])[:n_pc])):
            for j in range(i+1,n_pc):
                if (k==7) or (k==10):
                    flat[k].axis('off')
                    k+=1
                flat[k].scatter(PC_col*eigenvalue, np.transpose(svd_arr[0])[j]*svd_arr[1][j], color="r", s=10)
                flat[k].set_title("PCA"+str(i+1)+" vs PCA"+str(j+1),fontweight='bold',fontsize=14)
                flat[k].tick_params(axis='both', labelsize=11)
                k+=1
    return fig
# ---------------------------------------------------------- MCR ALS ----------------------------------------------------------
# initial estimation
# SIMPLISMA Algorithm, source : https://github.com/ClarkAH/SIMPLISMA
def simplisma(d, nr, error):

    def wmat(c,imp,irank,jvar):
        dm=np.zeros((irank+1, irank+1))
        dm[0,0]=c[jvar,jvar]
        
        for k in range(irank):
            kvar=np.int(imp[k])
            
            dm[0,k+1]=c[jvar,kvar]
            dm[k+1,0]=c[kvar,jvar]
            
            for kk in range(irank):
                kkvar=np.int(imp[kk])
                dm[k+1,kk+1]=c[kvar,kkvar]
                
        return dm

    nrow,ncol=d.shape
    
    dl = np.zeros((nrow, ncol))
    imp = np.zeros(nr)
    mp = np.zeros(nr)
    
    w = np.zeros((nr, ncol))
    p = np.zeros((nr, ncol))
    s = np.zeros((nr, ncol))
    
    error=error/100
    mean=np.mean(d, axis=0)
    error=np.max(mean)*error
    
    s[0,:]=np.std(d, axis=0)
    w[0,:]=(s[0,:]**2)+(mean**2)
    p[0,:]=s[0,:]/(mean+error)

    imp[0] = np.int(np.argmax(p[0,:]))
    mp[0] = p[0,:][np.int(imp[0])]
    
    l=np.sqrt((s[0,:]**2)+((mean+error)**2))

    for j in range(ncol):
        dl[:,j]=d[:,j]/l[j]
        
    c=np.dot(dl.T,dl)/nrow
    
    w[0,:]=w[0,:]/(l**2)
    p[0,:]=w[0,:]*p[0,:]
    s[0,:]=w[0,:]*s[0,:]
    
    print('purest variable 1: ', np.int(imp[0]+1), mp[0])

    for i in range(nr-1):
        for j in range(ncol):
            dm=wmat(c,imp,i+1,j)
            w[i+1,j]=np.linalg.det(dm)
            p[i+1,j]=w[i+1,j]*p[0,j]
            s[i+1,j]=w[i+1,j]*s[0,j]
            
        imp[i+1] = np.int(np.argmax(p[i+1,:]))
        mp[i+1] = p[i+1,np.int(imp[i+1])]
        
        print('purest variable '+str(i+2)+': ', np.int(imp[i+1]+1), mp[i+1])
        
    sp_arr=np.zeros((nrow, nr))
    
    for i in range(nr):
        sp_arr[0:nrow,i]=d[0:nrow,np.int(imp[i])]
        
    plt.subplot(3, 1, 2)
    plt.plot(sp_arr)
    plt.title('Estimate Components')
    
    concs = np.dot(np.linalg.pinv(sp_arr), d)
    
    plt.subplot(3, 1, 3)
    for i in range(nr):
        plt.plot(concs[i])
    plt.title('Concentrations')
    plt.show()
    
    return sp_arr, concs
