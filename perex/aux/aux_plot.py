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


#### for EC data only
def check_cycles(df):
    '''
    Function to check if an electro dataframe has columns of the 'cycle number' and 'half cycle'. If not, it creates them based on the changes in the potential.
    
    :df: Input dataframe created from an EC Lab file source.
    :return: Returns a merged dataframe with both informations.
    '''
    if 'cycle_number' not in df.columns and 'cycle_number_EC' not in df.columns:
        print('No cycle column')
        V_col=get_potential_col(df)
        oxred_col=get_oxred_col(df)
        change_idx=df[np.sign(df[V_col].diff(2)).diff().ne(0)].index.to_list()
        change_idx=np.array(change_idx)
        rows_mask=np.diff(change_idx)!=1
        rows_mask=np.append(True,rows_mask)
        change_idx=change_idx[rows_mask]
        if change_idx[-1]!=df.index[-1]:
            change_idx=np.append(change_idx,df.index[-1])
        half_cycle=[]
        cycle_number=[]
        start=0
        i=0
        last_oxred=df.iloc[start:change_idx[1]][oxred_col].mode()[0]
        for idx in change_idx[1:]:
            if df.iloc[start:idx][oxred_col].mode()[0]!=last_oxred:
                i+=1
            half_cycle.extend((i)*np.ones(idx-start))
            cycle_number.extend(((i+2)//2)*np.ones(idx-start))
            if idx==change_idx[-1]:
                half_cycle.append(i)
                cycle_number.append((i+2)//2)
            last_oxred=df.iloc[start:idx][oxred_col].mode()[0]
            start=idx
        df['cycle_number']=cycle_number
    if 'half_cycle' not in df.columns and 'half_cycle_EC' not in df.columns:
        print('No half cycle column')
        df['half_cycle']=half_cycle
    return

def get_potential_col(df):
    '''
    Function to get the name of the potential column.
    
    :df: Pandas dataframe with the merged or EC data.
    :return: Name of the column
    '''
    for col in df.columns:
        if 'Ewe' in col or '<Ewe>' in col:
            potential_col=col
            break
    return potential_col

def get_current_col(df):
    '''
    Function to get the name of the current column.
    
    :df: Pandas dataframe with the merged or EC data.
    :return: Name of the column
    '''
    for col in df.columns:
        if '<I>/mA' in col or 'I/mA' in col:
            current_col=col
            break
    return current_col

def get_capa_col(df):
    '''
    Function to get the name of the capacity column.
    
    :df: Pandas dataframe with the merged or EC data.
    :return: Name of the column
    '''
    for col in df.columns:
        if 'capacity' in col.lower():
            capa_col=col
            break
    return capa_col

def get_dq_col(df):
    '''
    Function to get the name of the dq column.
    
    :df: Pandas dataframe with the merged or EC data.
    :return: Name of the column
    '''
    for col in df.columns:
        if 'dq/mA.h' in col:
            dq_col=col
            break
    return dq_col

def get_oxred_col(df):
    '''
    Function to get the name of the ox/red column.
    
    :df: Pandas dataframe with the merged or EC data.
    :return: Name of the column
    '''
    for col in df.columns:
        if 'ox/red' in col.lower():
            oxred_col=col
            break
    return oxred_col

def get_half_cycle_col(df):
    '''
    Function to get the name of the half cycle column.
    
    :df: Pandas dataframe with the merged or EC data.
    :return: Name of the column
    '''
    for col in df.columns:
        if 'half_cycle' in col:
            half_cycle_col=col
            break
    return half_cycle_col

def get_cycle_number_col(df):
    '''
    Function to get the name of the cycle number column.
    
    :df: Pandas dataframe with the merged or XAS data.
    :return: Name of the column
    '''
    for col in df.columns:
        if 'cycle_number' in col:
            cycle_number_col=col
            break
    return cycle_number_col

def convert_nb_cycle(df,nb_cycle):
    cycle_col=get_cycle_number_col(df)
    if nb_cycle=='all':
        nb_cycle=df[cycle_col].unique()
    else:
        try:
            nb_cycle=int(nb_cycle)
            if nb_cycle not in df[cycle_col].unique():
                raise ValueError("Not a valid nb_cycle.")
            else:
                nb_cycle=[nb_cycle]
        except ValueError: raise ValueError('Not a valid nb_cycle.')
        except: pass
        
    if not all(elem in df[cycle_col].unique() for elem in nb_cycle):
        raise ValueError('The chosen cycle numbers are not in the sequence.')
    return nb_cycle

### for XAS or XAS merged with EC data
def get_energy_col(df):
    '''
    Function to get the name of the energy column.
    
    :df: Pandas dataframe with the merged or XAS data.
    :return: Name of the column
    '''
    np_arr_list=(df.applymap(type) == np.ndarray).all().index[(df.applymap(type) == np.ndarray).all()].tolist()
    for col in df.columns:
        if 'energy' in col.lower() and col in np_arr_list:
            energy_col=col
    return energy_col

def get_intensity_col(df):
    '''
    Function to get the name of the intensity column.
    
    :df: Pandas dataframe with the merged or XAS data.
    :return: Name of the column
    '''
    np_arr_list=(df.applymap(type) == np.ndarray).all().index[(df.applymap(type) == np.ndarray).all()].tolist()
    for col in df.columns:
        if ('norm' in col.lower()) and (col in np_arr_list) and ('ref' not in col.lower()) or ('mux' in col.lower()):
            intensity_col=col
            break
    return intensity_col

def get_abstime_col(df):
    '''
    Function to get the name of the absolute time column.
    
    :df: Pandas dataframe with the merged or XAS data.
    :return: Name of the column
    '''
    for col in df.columns:
        if 'absolute_time' in col.lower().replace(' ','_'):
            abstime_col=col
            break
    return abstime_col

def get_edge(df,intensity_val='inflection',intensity_col=''):
    '''
    Function to get the interpolated energy value at a certain intesity.
    
    :df: Pandas dataframe with the XAS spectra data.
    :intensity: Intensity value to get the edge energy value. Default is the inflection point.
    :return: List of energy values.
    '''
    indexes=[]
    edge_energy=[]
    energy_col=get_energy_col(df)
    if len(intensity_col)==0:
        try:
            intensity_col=get_intensity_col(df)
        except:
            raise ValueError("Please define a proper column for the intensity.")
    elif intensity_col not in df.columns:
        raise ValueError("Please define a proper column for the intensity.")
    for index, row in df.iterrows():
        E_fine = np.linspace(min(row[energy_col]),max(row[energy_col]), 100000)
        interpol = interp1d(row[energy_col], row[intensity_col])
        mu_norm_int=interpol(E_fine)
        deriv=pd.Series(mu_norm_int).diff()/pd.Series(E_fine).diff()
        if type(intensity_val)==float or type(intensity_val)==int:
            edge_energy.append(pd.Series(E_fine)[pd.Series(mu_norm_int).sub(intensity_val).abs().idxmin()])
            indexes.append(index)
        elif intensity_val=='inflection':
            edge_energy.append(pd.Series(E_fine)[pd.Series(deriv).idxmax()])
            indexes.append(index)
        else:
            raise ValueError('Not a valid intensity value.')
        edge_df=pd.Series(data=edge_energy,index=indexes)
    return edge_df