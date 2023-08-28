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

### plotting EC data only
def UI_vs_time(df,width=15,height=5):
    '''
    Function to plot a potential/current vs time graph.
    
    :df: Pandas dataframe with the data from the EC Lab file.
    :width: Width of the graph.
    :height: Height of the graph.
    :return: Plot.
    '''
    V_col=get_potential_col(df)
    I_col=get_current_col(df)
    fig, ax1 = plt.subplots(figsize=(width,height))
    # add an additional y axis for the current
    ax2 = ax1.twinx()
    # get time series
    try:
        timeCol=next(x for x in df.columns if 'absolute_time/s' in x.lower())
    except:
        timeCol=next(x for x in df.columns if 'time/s' in x.lower())
    #timeCol=[col for col in df.columns if 'time/s' in col.lower()][0]
    try:
        timeCol=next(x for x in df.columns if 'acquisition_datetime' in x.lower())
        timeSeries=(df[timeCol]-df[timeCol][0]).dt.total_seconds()
    except:
        timeSeries=df[timeCol]
    #elif df[df[timeCol]==0].shape[0]>1:
    #    timeSeries=list(df[timeCol][:df[df[timeCol]==0].index[1]])
    #    for i in df[df[timeCol]==0].index[1:-1]:
    #        timeSeries.extend(list(df[timeCol][i:i+1]+df[timeCol][i-1]))
    #else:
    #    timeSeries=df[timeCol]
    # plot V vs. time
    ax1.scatter(timeSeries/3600,df[V_col],color="blue",s=0.2)
    # plot I vs. time
    ax2.scatter(timeSeries/3600,df[I_col],color="green",s=0.2)
    # set the labels and the colors of the curves
    ax1.set_xlabel("Time (h)",fontsize=14)
    ax1.set_ylabel("Potential vs. Li/Li$^+$ (V)",color="blue",fontsize=14)
    ax2.set_ylabel("Current (mA)",color="green",fontsize=14)
    # color the axes
    ax1.tick_params(axis='y',colors="blue")
    ax2.tick_params(axis='y',colors="green")
    ax2.spines['left'].set_color("blue")
    ax2.spines['right'].set_color("green")
    # axes ticks inwards
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax2.tick_params(axis='both', labelsize=13, direction='in')
    return fig

def U_vs_capacity(df,nb_cycle='all',width=8,height=5,mass=1000):
    '''
    Function to plot a capacity vs potential graph.
    
    :df: Pandas dataframe with the data from the EC Lab file.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :width: Width of the graph.
    :height: Height of the graph.
    :mass: Mass of active material in mg.
    :return: Plot.
    '''
    mass=mass/1000 # mg to g
    df_copy=df.copy(deep=True)
    check_cycles(df_copy)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    #dq_col=get_dq_col(df_copy)
    total_cycles=len(df_copy[cycle_col].unique())
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    if 'Capacity/mA.h' not in df_copy.columns and 'Capacity/mA.h_EC' not in df_copy.columns:
        try:
            capa=[]
            for half_cycle in df_copy[half_cycle_col].unique():
                condition=df_copy[half_cycle_col]==half_cycle
                dq_col=get_dq_col(df_copy)
                capa.extend(df_copy[dq_col][condition].abs().cumsum())
            capa=np.array(capa)
            capa_col='Capacity/mA.h'
            df_copy[capa_col]=capa
        except:
            raise ValueError("Column for 'Capacity/mA.h' or 'dq/mA.h' not found.")
    else:
        capa_col=get_capa_col(df_copy)
    color_discharge = plt.get_cmap('seismic')(np.linspace(0, 0.5, total_cycles+2))[1:-1]
    color_charge = plt.get_cmap('seismic')(np.linspace(1, 0.5, total_cycles+2))[1:-1]
    # build figure
    fig, ax = plt.subplots(figsize=(width,height))
    for i,half_cycle in enumerate(df_copy[half_cycle_col].unique()):
        subdf=df_copy[df_copy[half_cycle_col]==half_cycle]
        cycle_number=int(subdf[cycle_col].mode())
        #cycle_number=int(subdf[cycle_col].mean())
        if cycle_number in nb_cycle:
            #if subdf[oxred_col].mode()[0]==1:
            if np.average(np.diff(subdf[V_col]))>0:
                ax.scatter(subdf[capa_col]/mass, subdf[V_col],s=0.2, color=color_charge[int(np.floor(i/2))])
                ax.annotate(str(cycle_number), (subdf[capa_col].iloc[-1]/mass, subdf[V_col].iloc[-1]+0.01), 
                            color=color_charge[int(np.floor(i/2))], va='bottom')
            else:
                ax.scatter(subdf[capa_col]/mass,subdf[V_col],s=0.2,
                           color=color_discharge[int(np.floor(i/2))])
                ax.annotate(str(cycle_number), (subdf[capa_col].iloc[-1]/mass, subdf[V_col].iloc[-1]-0.01), 
                            color=color_discharge[int(np.floor(i/2))], va='top')
    
    # set axes labels and limits
    ax.set_ylabel("Potential vs. Li/Li$^+$ (V)",fontsize=14)
    if mass==1:
        ax.set_xlabel("Capacity (mAh)",fontsize=14)
    else:
        ax.set_xlabel("Capacity (mAh$\cdot$g$^{-1}$)",fontsize=14)
    margin=0.05
    xmin=df_copy[capa_col].min()-(df_copy[capa_col].max()-df_copy[capa_col].min())*margin/(1-margin*2)
    xmax=df_copy[capa_col].max()+(df_copy[capa_col].max()-df_copy[capa_col].min())*margin/(1-margin*2)
    ymin=df_copy[V_col].min()-(df_copy[V_col].max()-df_copy[V_col].min())*margin/(1-margin*2)
    ymax=df_copy[V_col].max()+(df_copy[V_col].max()-df_copy[V_col].min())*margin/(1-margin*2)
    ax.set_xlim(xmin/mass,xmax/mass)
    ax.set_ylim(ymin,ymax)
    ax.tick_params(axis='both', labelsize=13, direction='in')
    return fig

def dQdU_vs_U(df,nb_cycle='all',reduce_by=1,boxcar=1,savgol=(1,0),colormap='plasma',
              width=10,height=6,dotsize=10,alpha=1, mass=1000):
    '''
    Function to plot a capacity vs potential graph.
    
    :df: Pandas dataframe with the data from the EC Lab file.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :reduce_by: Factor by which you want to reduce the number of points on your dataframe.
    :boxcar: Factor indicating the size of the moving window of a moving average filter
    :savgol: Tuple (x,y) with the parameters of a Savitzky-Golay filter.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot. Defaul set to 10.
    :alpha: Opacity of the points. Default set to 1.
    :mass: Mass of active material in mg.
    :return: Plot.
    '''
    mass=mass/1000 # mg to g
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    #dq_col=get_dq_col(df_copy)
    check_cycles(df_copy)
    total_cycles=len(df_copy[cycle_col].unique())
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)

    if 'Capacity/mA.h' not in df_copy.columns and 'Capacity/mA.h_EC' not in df_copy.columns:
        try:
            dq_col=get_dq_col(df_copy)
            df_copy['dQdV']=(df_copy[dq_col].abs()/mass)/df_copy[V_col].diff()
        except:
            raise ValueError("Column for 'Capacity/mA.h' or 'dq/mA.h' not found.")
    else:
        capa_col=get_capa_col(df_copy)
        df_copy['dQdV']=(df_copy[capa_col].diff()/mass)/df_copy[V_col].diff()
    # apply filters/smoothing
    df_copy=df_copy.iloc[::reduce_by]
    df_copy['dQdV']=df_copy['dQdV'].rolling(boxcar).mean()
    df_copy['dQdV']=savgol_filter(df_copy['dQdV'],savgol[0],savgol[1])

    cm=plt.get_cmap(colormap)
    # color map from the total number of cycles in the DF, not from the length of the input nb_cycles
    color_dqdv=cm(np.linspace(0, 1, total_cycles))
    
    # build figure
    fig, ax = plt.subplots(figsize=(width,height))

    for i, cycle in enumerate(df_copy[cycle_col].unique()):
        if cycle in nb_cycle:
            subdf=df_copy[df_copy[cycle_col]==cycle]
            ax.scatter(subdf[V_col],subdf['dQdV'],s=dotsize,color=color_dqdv[i],label='cycle '+str(int(cycle)),alpha=alpha)

    # set axes labels and limits
    ax.tick_params(axis='both', labelsize=13, direction='in')
    # x
    margin=0.05
    xmin=df_copy[V_col].min()-(df_copy[V_col].max()-df_copy[V_col].min())*margin/(1-margin*2)
    xmax=df_copy[V_col].max()+(df_copy[V_col].max()-df_copy[V_col].min())*margin/(1-margin*2)
    ax.set_xlim(xmin,xmax)
    ax.set_xlabel("Potential vs. Li/Li$^+$ (V)",fontsize=14)
    # y
    ylim=(abs(df_copy['dQdV'].quantile(0.95))+abs(df_copy['dQdV'].quantile(0.05)))/2
    ax.set_ylim(-ylim,ylim)
    if mass==1:
        ax.set_ylabel("dQ/dV (mAh$\cdot$V$^{-1}$)",fontsize=14)
    else:
        ax.set_ylabel("dQ/dV (mAh$\cdot$g$^{-1}\cdot$V$^{-1}$)",fontsize=14)

    # put colorbar legend
    norm = Normalize(vmin=int(df_copy[cycle_col].min()), vmax=int(df_copy[cycle_col].max()))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    if len(nb_cycle)>5:
        cbar = fig.colorbar(sm)
        cbar.set_label('Cycle', rotation=270, labelpad=10, fontsize=14)
    else:
        leg = ax.legend(loc='upper left',prop={'size': 14},markerscale=2)
    return fig

### plotting XAS data only
def all_XAS(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='viridis',pre=20, post=40,width=7,height=4):
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    energy_col=get_energy_col(df)
    rang=[edge.mean()-pre,edge.mean()+post]
    if len(intensity_col)==0:
        try:
            intensity_col=get_intensity_col(df)
        except:
            raise ValueError("Please define a proper column for the intensity.")
    elif intensity_col not in df.columns:
        raise ValueError("Please define a proper column for the intensity.")
    # colors
    cm=plt.get_cmap(colormap)
    norm = Normalize(vmin=df.index[0], vmax=df.index[-1])
    # bulid the plot
    fig, ax = plt.subplots(figsize=(width,height))
    for index, row in df.iterrows():
        ax.plot(row[energy_col],row[intensity_col],color=cm(norm(index)))
    #ax.set_xlim(rang[0],rang[1])
    ax.tick_params(axis='both', labelsize=13, direction='in')
    ax.set_xlabel('Energy (eV)',fontsize=14)
    ax.set_ylabel('Normalized intensity',fontsize=14)
    # put a legend
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
    cbar = fig.colorbar(sm)
    cbar.set_label('Spectrum #', rotation=270, labelpad=15, fontsize=14)
    cbar.ax.tick_params(labelsize=13)
    return fig

def XAS_vs_t_2D(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',abstime_col='',
                colormap='turbo', width=7,height=6,plot_range=None,hlines=False):
    '''
    Function to plot a 2D intensity graph of all the XAS spectra over time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'tab20b'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
    # if no cycle number is selected then it just plots all of them
    if len(intensity_col)==0:
        try:
            intensity_col=get_intensity_col(df)
        except:
            raise ValueError("Please define a proper column for the intensity.")
    elif intensity_col not in df.columns:
        raise ValueError("Please define a proper column for the intensity.")
    if len(abstime_col)==0:
        try:
            abstime_col=get_abstime_col(df)
        except:
            raise ValueError("Please define a proper column for time.")
    elif abstime_col not in df.columns:
        raise ValueError("Please define a proper column for time.")
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    energy_col=get_energy_col(df)

    if plot_range:
        if ((len(plot_range)==2) & (type(plot_range)==list)) & (all([isinstance(item, (int,float)) for item in plot_range])):
            rang=plot_range
        else:
            raise ValueError("Not a valid plot_range.")
    else:
        rang=[edge.mean()-20,edge.mean()+40]

    #abstime=df[abstime_col]
    # get energy grid and normalized absorption profiles
    #energies=np.array(df[energy_col].to_list())
    all_spectra=np.array(df[intensity_col].to_list())

    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(all_spectra.min(), all_spectra.max())
    fig, ax = plt.subplots(figsize=(width,height))
    diff_time=df[abstime_col].diff()/3600
    cut_indexes=list(df[diff_time>diff_time.describe()['mean']+diff_time.describe()['std']*4].index)
    cut_indexes.append(None)

    begin=0
    for idx in cut_indexes:
        energies=np.array(df[begin:idx][energy_col].to_list())
        spectra=np.array(df[begin:idx][intensity_col].to_list())
        times=df[begin:idx][abstime_col]/3600
        im = ax.pcolormesh(energies, times, spectra, cmap=cmap, norm=norm, shading='gouraud')
        begin=idx

    ax.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
    ax.set_ylabel('Time (h)',fontsize=14,labelpad=10)
    ax.tick_params(axis='both', labelsize=13)
    ax.minorticks_on()
    #ax.set_xlim(rang[0], rang[1])
    ax.set_ylim(0,df[abstime_col].max()/3600)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized absorption',fontsize=14,labelpad=10)
    cbar.ax.tick_params(labelsize=13)
    
    return fig

### plotting combiined XAS + EC
def Eshift_vs_U(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',option=1,
                colormap='plasma',width=7,height=4,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs potential graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value. Default is the inflection point.
    :intensity_col: Name of the column with the intensity values.
    :option: Parameter to choose the style of the graph. Default set to 1, meaning the figure will have a simplified color code with charge subcycles in red, discharge subcycles in blue, and different markerstyles for each cycle. If set to 2, the figure will have a different color for each cycle (according to a chosen colormap). If set to 3, x axis will be extended (1 sub-figure for each cycle).
    :colormap: Chosen colormap if option set to 2. Default set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot (option 2).
    :alpha: Opacity of the points (option 2).
    :return: Plot.
    '''
    if option==1:
        fig=Eshift_vs_U_alpha(df,nb_cycle,edge_intensity,intensity_col,width,height,guideline)
    elif option==2:
        fig=Eshift_vs_U_beta(df,nb_cycle,edge_intensity,intensity_col,colormap,width,height,dotsize,alpha,guideline)
    else:
        fig=Eshift_vs_U_long(df,nb_cycle,edge_intensity,intensity_col,colormap,guideline=guideline)
    return fig

def Eshift_vs_U_alpha(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',width=10,height=6,
                      guideline=False):
    '''
    Function to plot edge shift vs potential graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :width: Width of the graph.
    :height: Height of the graph.
    :return: Plot.
    '''
    
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    fillstyle=['full','none']
    markers = itertools.cycle(('o','o','v','v','X','X','*','*','s','s','P','P')) 
    colors = ['r','b']


    fig, ax = plt.subplots(figsize=(width,height))
    # plot Edge shift
    condition = df_copy[cycle_col].isin(nb_cycle)
    counter=0
    if guideline:
        ax.plot(df_copy[condition][V_col],edge[condition],alpha=0.8,linewidth=1.5,color='lightgrey',zorder=counter)
        counter+=1
    for i, cycle in enumerate(df_copy[cycle_col].unique()):
        if cycle in nb_cycle:
            subdf=df_copy[df_copy[cycle_col]==cycle]
            marker=next(markers)
            for j, subcycle in enumerate(subdf[half_cycle_col].unique()):
                if subdf[subdf[half_cycle_col]==subcycle][oxred_col].mode()[0]==1:
                    subcycle_type='charge'
                    color=colors[0]
                else:
                    subcycle_type='discharge'
                    color=colors[1]
                ax.plot(subdf[subdf[half_cycle_col]==subcycle][V_col],edge[subdf[subdf[half_cycle_col]==subcycle].index],
                        marker=marker, fillstyle=fillstyle[i%2],color=color,lw=0,
                        label='cycle '+str(int(cycle))+' '+subcycle_type,zorder=counter)
                counter+=1
    # axes labels and limits
    margin=0.05
    ax.tick_params(axis='both', labelsize=13, direction='in')
    # x
    xmin=df_copy[V_col].min()-(df_copy[V_col].max()-df_copy[V_col].min())*margin/(1-margin*2)
    xmax=df_copy[V_col].max()+(df_copy[V_col].max()-df_copy[V_col].min())*margin/(1-margin*2)
    ax.set_xlim(xmin,xmax)
    ax.set_xlabel("Potential vs. Li/Li$^+$ (V)",fontsize=14)
    # y
    ymin=edge.min()-(edge.max()-edge.min())*margin/(1-margin*2)
    ymax=edge.max()+(edge.max()-edge.min())*margin/(1-margin*2)
    ax.set_ylim(ymin,ymax)
    label_y='Edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y,fontsize=14)

    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w', direction="in")
    
    # Put a legend to the right of the current axis
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    #plt.legend(by_label.values(), by_label.keys())
    leg = ax.legend(by_label.values(), by_label.keys(),loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
    return fig

def Eshift_vs_U_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                     width=7,height=4,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs potential graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)

    # normalize the colormap with respect to the total number of cycles in the df
    norm = Normalize(vmin=int(df_copy[cycle_col].min()), vmax=int(df_copy[cycle_col].max()))
    # build plot
    fig, ax = plt.subplots(figsize=(width,height))
    
    condition = df_copy[cycle_col].isin(nb_cycle)
    # plot Edge shift
    counter=0
    if guideline:
        ax.plot(df_copy[condition][V_col],edge[condition],alpha=0.8,linewidth=1.5,color='lightgrey',zorder=counter)
        counter+=1
    scatter = ax.scatter(df_copy[V_col][condition],edge[condition],s=dotsize,c=df_copy[cycle_col][condition], cmap=colormap, norm=norm,alpha=alpha,zorder=counter)
    
    # axes labels and limits
    margin=0.05
    ax.tick_params(axis='both', labelsize=13, direction='in')
    # x
    xmin=df_copy[V_col].min()-(df_copy[V_col].max()-df_copy[V_col].min())*margin/(1-margin*2)
    xmax=df_copy[V_col].max()+(df_copy[V_col].max()-df_copy[V_col].min())*margin/(1-margin*2)
    ax.set_xlim(xmin,xmax)
    ax.set_xlabel("Potential vs. Li/Li$^+$ (V)",fontsize=14)
    # y
    ymin=edge.min()-(edge.max()-edge.min())*margin/(1-margin*2)
    ymax=edge.max()+(edge.max()-edge.min())*margin/(1-margin*2)
    ax.set_ylim(ymin,ymax)
    label_y='Edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y,fontsize=14)

    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w', direction="in")
    
    # put a legend
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
    if len(nb_cycle)>5:
        cbar = fig.colorbar(sm)
        cbar.set_label('Cycle', rotation=270, labelpad=10, fontsize=14)
    else:
        #leg = ax.legend(loc='upper left',prop={'size': 13})
        leg = ax.legend(handles=scatter.legend_elements()[0], labels=['cycle '+str(int(i)) for i in nb_cycle], prop={'size': 12})
    return fig

def Eshift_vs_U_long(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                     colormap='plasma',width=15,height=4,linewidth=1.5,dotsize=18,alpha=0.5,top=0.04,guideline=True):
    '''
    Function to plot edge shift vs potential.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)

    #colormap according to total number of cycles
    cm=plt.get_cmap(colormap)
    color_dqdv=cm(np.linspace(0, 1, len(df_copy[cycle_col].unique())))
    
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(width, height)

    half_cycles = []
    maxs = []
    mins = []
    widths = []
    mode = []
    for index, row in df_copy.groupby(half_cycle_col).agg({V_col: ['min', 'max'], oxred_col:['mean']}).iterrows():
        width=row[1]-row[0]
        if width !=0:
            half_cycles.append(index)
            mins.append(row[0])
            maxs.append(row[1])
            widths.append(width)
            mode.append(round(row[2]))

    gs_top = fig.add_gridspec(nrows=1, ncols=len(half_cycles), width_ratios=widths,top=0.95)

    # PLOTTING THE EDGE EVOLUTION
    art_idx=0
    for col in range(len(half_cycles)):
        condition=df_copy[half_cycle_col]==half_cycles[col]
        index_cm=col//2
        ax0 = fig.add_subplot(gs_top[0,col])
        if guideline:
            ax0.plot(df_copy[condition][V_col],edge[art_idx:art_idx+len(df_copy[condition][V_col])],
                     alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
        ax0.scatter(df_copy[condition][V_col],edge[art_idx:art_idx+len(df_copy[condition][V_col])],
                    label=str(half_cycles[col]),color=color_dqdv[index_cm],s=dotsize,zorder=1)
        art_idx=art_idx+len(df_copy[condition][V_col])
        # x ticks locator
        xticks = ticker.MaxNLocator(round(widths[col]/0.25))
        ax0.xaxis.set_major_locator(xticks)
        ax0.tick_params(axis='both', labelsize=13)
        ax0.set_xlim(mins[col],maxs[col])
        ax0.set_ylim(edge.min(),edge.max())
        if mode[col]==0:
            ax0.invert_xaxis()
        if col!=0:
            ax0.get_yaxis().set_visible(False)
            ax0.spines['left'].set_linestyle("dashed")
            ax0.spines['left'].set_capstyle("butt")
        else:
            ax0.set_ylabel("Edge @ J="+str(edge_intensity)+" (eV)", fontsize=15, labelpad=10)

    gs_base = fig.add_gridspec(nrows=1, ncols=len(half_cycles), width_ratios=widths, top=top,bottom=top-0.01)
    # X AXIS LABEL
    ax2 = fig.add_subplot(gs_base[0,:])
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_color('white')
    ax2.set_xlabel("Potential vs. Li/Li$^+$ (V)", fontsize=15)
    plt.subplots_adjust(wspace=0)
    return fig

def Eshift_vs_x(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',option=1,
                colormap='plasma',width=7,height=4,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs x graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value. Default is the inflection point.
    :intensity_col: Name of the column with the intensity values.
    :option: Parameter to choose the style of the graph. Default set to 1, meaning the figure will have a simplified color code with charge subcycles in red, discharge subcycles in blue, and different markerstyles for each cycle. If set to 2, the figure will have a different color for each cycle (according to a chosen colormap). If set to 3, x axis will be extended (1 sub-figure for each cycle).
    :colormap: Chosen colormap if option set to 2. Default set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :return: Plot.
    '''
    if option==1:
        fig=Eshift_vs_x_alpha(df,nb_cycle,edge_intensity,intensity_col,width,height,guideline)
    elif option==2:
        fig=Eshift_vs_x_beta(df,nb_cycle,edge_intensity,intensity_col,colormap,width,height,dotsize,alpha,guideline)
    else:
        fig=Eshift_vs_x_long(df,nb_cycle,edge_intensity,intensity_col,colormap,guideline=guideline)
    return fig

def Eshift_vs_x_alpha(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',width=10,height=6,
                      guideline=False):
    '''
    Function to plot edge shift vs x graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :width: Width of the graph.
    :height: Height of the graph.
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    fillstyle=['full','none']
    markers = itertools.cycle(('o','o','v','v','X','X','*','*','s','s','P','P')) 
    colors = ['r','b']


    fig, ax = plt.subplots(figsize=(width,height))
    # plot Edge shift
    condition = df_copy[cycle_col].isin(nb_cycle)
    counter=0
    try:
        if 'calculated_x' in df_copy.columns:
            x_col='calculated_x'
        elif 'calculated_x_EC' in df_copy.columns:
            x_col='calculated_x_EC'
        elif 'x' in df_copy.columns:
            x_col='x'
        else:
            x_col='x_EC'
    except: raise ValueError('x not found in the dataframe columns.')
        
    if guideline:
        ax.plot(df_copy[condition][x_col],edge[condition],alpha=0.8,linewidth=1.5,color='lightgrey',zorder=counter)
        counter+=1
    for i, cycle in enumerate(df_copy[cycle_col].unique()):
        if cycle in nb_cycle:
            subdf=df_copy[df_copy[cycle_col]==cycle]
            marker=next(markers)
            for j, subcycle in enumerate(subdf[half_cycle_col].unique()):
                if subdf[subdf[half_cycle_col]==subcycle][oxred_col].mode()[0]==1:
                    subcycle_type='charge'
                    color=colors[0]
                else:
                    subcycle_type='discharge'
                    color=colors[1]
                if guideline:
                    ax.plot(subdf[subdf[half_cycle_col]==subcycle][x_col],edge[subdf[subdf[half_cycle_col]==subcycle].index],
                            alpha=0.3,linewidth=1.5,color=color,zorder=counter)
                    counter+=1
                ax.plot(subdf[subdf[half_cycle_col]==subcycle][x_col],edge[subdf[subdf[half_cycle_col]==subcycle].index],
                        marker=marker, fillstyle=fillstyle[i%2],color=color,lw=0,
                        label='cycle '+str(int(cycle))+' '+subcycle_type,zorder=counter)
                counter+=1
    # axes labels and limits
    margin=0.05
    ax.tick_params(axis='both', labelsize=13, direction='in')
    # x
    xmin=df_copy[x_col].min()-(df_copy[x_col].max()-df_copy[x_col].min())*margin/(1-margin*2)
    xmax=df_copy[x_col].max()+(df_copy[x_col].max()-df_copy[x_col].min())*margin/(1-margin*2)
    ax.set_xlim(xmin,xmax)
    ax.set_xlabel("x",fontsize=14)
    # y
    ymin=edge.min()-(edge.max()-edge.min())*margin/(1-margin*2)
    ymax=edge.max()+(edge.max()-edge.min())*margin/(1-margin*2)
    ax.set_ylim(ymin,ymax)
    label_y='Edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y,fontsize=14)

    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w', direction="in")
    
    # put a legend
    
    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
    return fig

def Eshift_vs_x_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                     width=10,height=6,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs x graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)


    # normalize the colormap with respect to the total number of cycles in the df
    norm = Normalize(vmin=int(df_copy[cycle_col].min()), vmax=int(df_copy[cycle_col].max()))
    # build plot
    fig, ax = plt.subplots(figsize=(width,height))


    condition = df_copy[cycle_col].isin(nb_cycle)
    # plot Edge shift
    counter=0
    try:
        if 'calculated_x' in df_copy.columns:
            x_col='calculated_x'
        elif 'calculated_x_EC' in df_copy.columns:
            x_col='calculated_x_EC'
        elif 'x' in df_copy.columns:
            x_col='x'
        else:
            x_col='x_EC'
    except: raise ValueError('x not found in the dataframe columns.')
    
    if guideline:
        ax.plot(df_copy[condition][x_col],edge[condition],alpha=0.8,linewidth=1.5,color='lightgrey',zorder=counter)
        counter+=1
    scatter = ax.scatter(df_copy[x_col][condition],edge[condition],s=dotsize,
                         c=df_copy[cycle_col][condition], cmap=colormap, norm=norm,alpha=alpha,zorder=counter)
    
    # axes labels and limits
    margin=0.05
    ax.tick_params(axis='both', labelsize=13, direction='in')
    # x
    xmin=df_copy[x_col].min()-(df_copy[x_col].max()-df_copy[x_col].min())*margin/(1-margin*2)
    xmax=df_copy[x_col].max()+(df_copy[x_col].max()-df_copy[x_col].min())*margin/(1-margin*2)
    ax.set_xlim(xmin,xmax)
    ax.set_xlabel("x",fontsize=14)
    # y
    ymin=edge.min()-(edge.max()-edge.min())*margin/(1-margin*2)
    ymax=edge.max()+(edge.max()-edge.min())*margin/(1-margin*2)
    ax.set_ylim(ymin,ymax)
    label_y='Edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y,fontsize=14)

    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w', direction="in")
    
    # put a legend
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
    if len(nb_cycle)>5:
        cbar = fig.colorbar(sm)
        cbar.set_label('Cycle', rotation=270, labelpad=10, fontsize=14)
    else:
        #leg = ax.legend(loc='upper left',prop={'size': 13})
        leg = ax.legend(handles=scatter.legend_elements()[0], labels=['cycle '+str(int(i)) for i in nb_cycle], prop={'size': 12})
    return fig

def Eshift_vs_x_long(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                     colormap='plasma',width=15,height=4,linewidth=1.5,dotsize=18,alpha=0.5,top=0.04,guideline=True):
    '''
    Function to plot edge shift vs x.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    try:
        if 'calculated_x' in df_copy.columns:
            x_col='calculated_x'
        elif 'calculated_x_EC' in df_copy.columns:
            x_col='calculated_x_EC'
        elif 'x' in df_copy.columns:
            x_col='x'
        else:
            x_col='x_EC'
    except: raise ValueError('x not found in the dataframe columns.')
        
    #colormap according to total number of cycles
    cm=plt.get_cmap(colormap)
    color_dqdv=cm(np.linspace(0, 1, len(df_copy[cycle_col].unique())))
    
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(width, height)

    half_cycles = []
    maxs = []
    mins = []
    widths = []
    mode = []
    for index, row in df_copy.groupby(half_cycle_col).agg({x_col: ['min', 'max'], oxred_col:['mean']}).iterrows():
        width=row[1]-row[0]
        if width !=0:
            half_cycles.append(index)
            mins.append(row[0])
            maxs.append(row[1])
            widths.append(width)
            mode.append(round(row[2]))

    gs_top = fig.add_gridspec(nrows=1, ncols=len(half_cycles), width_ratios=widths, top=0.95)

    # PLOTTING THE EDGE EVOLUTION
    art_idx=0
    for col in range(len(half_cycles)):
        condition=df_copy[half_cycle_col]==half_cycles[col]
        index_cm=col//2
        ax0 = fig.add_subplot(gs_top[0,col])
        if guideline:
            ax0.plot(df_copy[condition][x_col],edge[art_idx:art_idx+len(df_copy[condition][x_col])],
                     alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
        ax0.scatter(df_copy[condition][x_col],edge[art_idx:art_idx+len(df_copy[condition][x_col])],
                    label=str(half_cycles[col]),color=color_dqdv[index_cm],s=dotsize,zorder=1)
        art_idx=art_idx+len(df_copy[condition][x_col])
        # x tick locator
        if round(widths[col]/0.25)==0:
            xticks = ticker.MaxNLocator(1)
        else:
            xticks = ticker.MaxNLocator(round(widths[col]/0.25))
        ax0.xaxis.set_major_locator(xticks)
        ax0.tick_params(axis='both', labelsize=13)
        ax0.set_xlim(mins[col],maxs[col])
        ax0.set_ylim(edge.min(),edge.max())
        
        if mode[col]!=0:
            ax0.invert_xaxis()
                
        if col!=0:
            ax0.get_yaxis().set_visible(False)
            ax0.spines['left'].set_linestyle("dashed")
            ax0.spines['left'].set_capstyle("butt")
        else:
            ax0.set_ylabel("Edge @ J="+str(edge_intensity)+" (eV)", fontsize=15, labelpad=10)

    gs_base = fig.add_gridspec(nrows=1, ncols=len(half_cycles), width_ratios=widths, top=top,bottom=top-0.01)
    # X AXIS LABEL
    ax2 = fig.add_subplot(gs_base[0,:])
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_color('white')
    ax2.set_xlabel("x", fontsize=15)
    plt.subplots_adjust(wspace=0)
    return fig

def xfromEdge_vs_x(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',x_col='x',option=1,
                   colormap='plasma',width=7,height=4,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs x graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value. Default is the inflection point.
    :intensity_col: Name of the column with the intensity values.
    :option: Parameter to choose the style of the graph. Default set to 1, meaning the figure will have a simplified color code with charge subcycles in red, discharge subcycles in blue, and different markerstyles for each cycle. If set to 2, the figure will have a different color for each cycle (according to a chosen colormap). If set to 3, x axis will be extended (1 sub-figure for each cycle).
    :colormap: Chosen colormap if option set to 2. Default set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :return: Plot.
    '''
    if option==1:
        fig=x_vs_x_alpha(df,nb_cycle,edge_intensity,intensity_col,x_col,width,height,guideline)
    elif option==2:
        fig=x_vs_x_beta(df,nb_cycle,edge_intensity,intensity_col,x_col,colormap,width,height,dotsize,alpha,guideline)
    else:
        fig=x_vs_x_long(df,nb_cycle,edge_intensity,intensity_col,x_col,colormap,width,height,dotsize,alpha,guideline)
    return fig

def x_vs_x_alpha(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',x_col='x',width=10,height=6,guideline=False):
    '''
    Function to plot x caculated from edge shift vs x.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :x_col: Name of the column with x.
    :width: Width of the graph.
    :height: Height of the graph.
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    df_copy['x_from_edge']=1-(edge-edge.min())/(edge.max()-edge.min())
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    fillstyle=['full','none']
    markers = itertools.cycle(('o','o','v','v','X','X','*','*','s','s','P','P')) 
    colors = ['r','b']


    fig, ax = plt.subplots(figsize=(width,height))
    # plot Edge shift
    condition = df_copy[cycle_col].isin(nb_cycle)
    counter=0
    try:
        if 'calculated_x' in df_copy.columns:
            x_col='calculated_x'
        elif 'calculated_x_EC' in df_copy.columns:
            x_col='calculated_x_EC'
        elif 'x' in df_copy.columns:
            x_col='x'
        else:
            x_col='x_EC'
    except: raise ValueError('x not found in the dataframe columns.')
    
    ax.plot([0,1],[0,1],color='black')
    if guideline:
        ax.plot(df_copy[condition][x_col],df_copy[condition]['x_from_edge'],alpha=0.8,linewidth=1.5,color='lightgrey',zorder=counter)
        counter+=1
    for i, cycle in enumerate(df_copy[cycle_col].unique()):
        if cycle in nb_cycle:
            subdf=df_copy[df_copy[cycle_col]==cycle]
            marker=next(markers)
            for j, subcycle in enumerate(subdf[half_cycle_col].unique()):
                if subdf[subdf[half_cycle_col]==subcycle][oxred_col].mode()[0]==1:
                    subcycle_type='charge'
                    color=colors[0]
                else:
                    subcycle_type='discharge'
                    color=colors[1]
                if guideline:
                    ax.plot(subdf[subdf[half_cycle_col]==subcycle][x_col],subdf[subdf[half_cycle_col]==subcycle]['x_from_edge'],
                            alpha=0.3,linewidth=1.5,color=color,zorder=counter)
                    counter+=1
                ax.plot(subdf[subdf[half_cycle_col]==subcycle][x_col],subdf[subdf[half_cycle_col]==subcycle]['x_from_edge'],
                        marker=marker, fillstyle=fillstyle[i%2],color=color,lw=0,
                        label='cycle '+str(int(cycle))+' '+subcycle_type,zorder=counter)
                counter+=1
    # axes labels and limits
    margin=0.05
    ax.tick_params(axis='both', labelsize=13, direction='in')
    # x
    xmin=df_copy[x_col].min()-(df_copy[x_col].max()-df_copy[x_col].min())*margin/(1-margin*2)
    xmax=df_copy[x_col].max()+(df_copy[x_col].max()-df_copy[x_col].min())*margin/(1-margin*2)
    ax.set_xlim(xmin,xmax)
    ax.set_xlabel("x",fontsize=14)
    # y
    ymin=df_copy['x_from_edge'].min()-(df_copy['x_from_edge'].max()-df_copy['x_from_edge'].min())*margin/(1-margin*2)
    ymax=df_copy['x_from_edge'].max()+(df_copy['x_from_edge'].max()-df_copy['x_from_edge'].min())*margin/(1-margin*2)
    ax.set_ylim(ymin,ymax)
    label_y='x from edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y,fontsize=14)

    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w', direction="in")
    
    # put a legend
    
    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
    return fig

def x_vs_x_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',x_col='x',colormap='plasma',
                width=10,height=6,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs x graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :x_col: Name of the column with the x values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    df_copy['x_from_edge']=1-(edge-edge.min())/(edge.max()-edge.min())
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    # normalize the colormap with respect to the total number of cycles in the df
    norm = Normalize(vmin=int(df_copy[cycle_col].min()), vmax=int(df_copy[cycle_col].max()))
    # build plot
    fig, ax = plt.subplots(figsize=(width,height))
    
    condition = df_copy[cycle_col].isin(nb_cycle)
    # plot Edge shift
    counter=0
    try:
        if 'calculated_x' in df_copy.columns:
            x_col='calculated_x'
        elif 'calculated_x_EC' in df_copy.columns:
            x_col='calculated_x_EC'
        elif 'x' in df_copy.columns:
            x_col='x'
        else:
            x_col='x_EC'
    except: raise ValueError('x not found in the dataframe columns.')
    
    if guideline:
        ax.plot(df_copy[condition][x_col],df_copy[condition]['x_from_edge'],alpha=0.8,linewidth=1.5,color='lightgrey',zorder=counter)
        counter+=1
    scatter = ax.scatter(df_copy[x_col][condition],df_copy[condition]['x_from_edge'],s=dotsize,
                         c=df_copy[cycle_col][condition], cmap=colormap, norm=norm,alpha=alpha,zorder=counter)
    ax.plot([0,1],[0,1],color='black')
    
    # axes labels and limits
    margin=0.05
    ax.tick_params(axis='both', labelsize=13, direction='in')
    # x
    xmin=df_copy[x_col].min()-(df_copy[x_col].max()-df_copy[x_col].min())*margin/(1-margin*2)
    xmax=df_copy[x_col].max()+(df_copy[x_col].max()-df_copy[x_col].min())*margin/(1-margin*2)
    ax.set_xlim(xmin,xmax)
    ax.set_xlabel("x",fontsize=14)
    # y
    ymin=df_copy['x_from_edge'].min()-(df_copy['x_from_edge'].max()-df_copy['x_from_edge'].min())*margin/(1-margin*2)
    ymax=df_copy['x_from_edge'].max()+(df_copy['x_from_edge'].max()-df_copy['x_from_edge'].min())*margin/(1-margin*2)
    ax.set_ylim(ymin,ymax)
    label_y='x from edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y,fontsize=14)

    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w', direction="in")
    
    # put a legend
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
    if len(nb_cycle)>5:
        cbar = fig.colorbar(sm)
        cbar.set_label('Cycle', rotation=270, labelpad=10, fontsize=14)
    else:
        #leg = ax.legend(loc='upper left',prop={'size': 13})
        leg = ax.legend(handles=scatter.legend_elements()[0], labels=['cycle '+str(int(i)) for i in nb_cycle], prop={'size': 12})
    return fig

def x_vs_x_long(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',x_col='x',
                colormap='plasma',width=15,height=4,dotsize=18,alpha=0.5,top=0.04,guideline=True,linewidth=1.5):
    '''
    Function to plot edge shift vs x.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    df_copy['x_from_edge']=1-(edge-edge.min())/(edge.max()-edge.min())
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    try:
        if 'calculated_x' in df_copy.columns:
            x_col='calculated_x'
        elif 'calculated_x_EC' in df_copy.columns:
            x_col='calculated_x_EC'
        elif 'x' in df_copy.columns:
            x_col='x'
        else:
            x_col='x_EC'
    except: raise ValueError('x not found in the dataframe columns.')
        
    #colormap according to total number of cycles
    cm=plt.get_cmap(colormap)
    color_dqdv=cm(np.linspace(0, 1, len(df_copy[cycle_col].unique())))
    
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(width, height)

    half_cycles = []
    maxs = []
    mins = []
    widths = []
    mode = []
    for index, row in df_copy.groupby(half_cycle_col).agg({x_col: ['min', 'max'], oxred_col:['mean']}).iterrows():
        width=row[1]-row[0]
        if width !=0:
            half_cycles.append(index)
            mins.append(row[0])
            maxs.append(row[1])
            widths.append(width)
            mode.append(round(row[2]))

    gs_top = fig.add_gridspec(nrows=1, ncols=len(half_cycles), width_ratios=widths, top=0.95)

    # PLOTTING THE X FROM EDGE EVOLUTION
    for col in range(len(half_cycles)):
        condition=df_copy[half_cycle_col]==half_cycles[col]
        index_cm=col//2
        ax0 = fig.add_subplot(gs_top[0,col])
        if guideline:
            ax0.plot(df_copy[condition][x_col],df_copy[condition]['x_from_edge'],
                     alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
        ax0.scatter(df_copy[condition][x_col],df_copy[condition]['x_from_edge'],
                    label=str(half_cycles[col]),color=color_dqdv[index_cm],s=dotsize,zorder=1)
        ax0.plot([0,1],[0,1],color='black')
        # x tick locator
        xticks = ticker.MaxNLocator(round(widths[col]/0.25))
        ax0.xaxis.set_major_locator(xticks)
        ax0.tick_params(axis='both', labelsize=13)
        ax0.set_xlim(mins[col],maxs[col])
        ax0.set_ylim(df_copy[condition]['x_from_edge'].min(),df_copy[condition]['x_from_edge'].max())
        
        if mode[col]!=0:
            ax0.invert_xaxis()
                
        if col!=0:
            ax0.get_yaxis().set_visible(False)
            ax0.spines['left'].set_linestyle("dashed")
            ax0.spines['left'].set_capstyle("butt")
        else:
            ax0.set_ylabel("x from edge @ J="+str(edge_intensity)+" (eV)", fontsize=15, labelpad=10)

    gs_base = fig.add_gridspec(nrows=1, ncols=len(half_cycles), width_ratios=widths, top=top,bottom=top-0.01)
    # X AXIS LABEL
    ax2 = fig.add_subplot(gs_base[0,:])
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_color('white')
    ax2.set_xlabel("x", fontsize=15)
    plt.subplots_adjust(wspace=0)
    return fig

def Eshift_vs_t(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',width=10,height=4,
                colormap='plasma',option=1,dotsize=10,alpha=0.5,guideline=False,hspace=.0):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value. Default set to inflection point.
    :intensity_col: Name of the column with the intensity values.
    :width: Width of the graph.
    :height: Height of the graph.
    :option: Parameter to choose the style of the graph. Default set to 1, meaning the figure will have a simplified color code with charge subcycles in red, discharge subcycles in blue, and different markerstyles for each cycle. If set to 2, the figure will have a different color for each cycle (according to a chosen colormap).
    :colormap: Chosen colormap if option set to 2. Default set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    if option==1:
        fig=Eshift_vs_t_alpha(df,nb_cycle,edge_intensity,intensity_col,width,height,guideline)
    elif option==2:
        fig=Eshift_vs_t_beta(df,nb_cycle,edge_intensity,intensity_col,colormap,width,height,dotsize,alpha,guideline)
    elif option==3:
        fig=Eshift_vs_t_stacked_alpha(df,nb_cycle,edge_intensity,intensity_col,width,height,dotsize,guideline,hspace)
    elif option==4:
        fig=Eshift_vs_t_stacked_beta(df,nb_cycle,edge_intensity,intensity_col,colormap,width,height,dotsize,guideline,hspace)
    return fig
    
def Eshift_vs_t_alpha(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',width=10,height=6,
                      guideline=False):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :width: Width of the graph.
    :height: Height of the graph.
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    fillstyle=['full','none']
    markers = itertools.cycle(('o','o','v','v','X','X','*','*','s','s','P','P')) 
    colors = ['r','b']
    
    # build figure
    fig, ax = plt.subplots(figsize=(width,height))
    # plot Edge shift
    condition = df_copy[cycle_col].isin(nb_cycle)
    counter=1
    for i, cycle in enumerate(df_copy[cycle_col].unique()):
        if cycle in nb_cycle:
            subdf=df_copy[df_copy[cycle_col]==cycle]
            marker=next(markers)
            for j, subcycle in enumerate(subdf[half_cycle_col].unique()):
                if subdf[subdf[half_cycle_col]==subcycle][oxred_col].mode()[0]==1:
                    subcycle_type='charge'
                    color=colors[0]
                else:
                    subcycle_type='discharge'
                    color=colors[1]
                if guideline:
                    ax.plot(subdf[subdf[half_cycle_col]==subcycle]['absolute_time/s_XAS']/3600,
                            edge[subdf[subdf[half_cycle_col]==subcycle].index],
                            alpha=0.3,linewidth=1.5,color=color,zorder=counter)
                    counter+=1
                ax.plot(subdf[subdf[half_cycle_col]==subcycle]['absolute_time/s_XAS']/3600,
                        edge[subdf[subdf[half_cycle_col]==subcycle].index],
                        marker=marker, fillstyle=fillstyle[i%2],color=color,lw=0,
                        label='cycle '+str(int(cycle))+' '+subcycle_type,zorder=counter)
                counter+=1
    ax.spines['left'].set_color('red')
    ax.tick_params(axis='y', colors='b')
    ax.yaxis.label.set_color('r')
    
    # plot potential over time
    condition = df_copy[cycle_col].isin(nb_cycle)
    ax1 = ax.twinx()
    if guideline:
        ax.plot(df_copy[condition]['absolute_time/s_XAS']/3600,edge[condition],alpha=0.8,linewidth=1.5,color='lightgrey',zorder=0)
        ax1.plot(df_copy[condition]['absolute_time/s_XAS']/3600,df_copy[V_col][condition],
                 alpha=0.8,linewidth=1.5,color='lightgrey',zorder=0)
    ax1.scatter(df_copy[condition]['absolute_time/s_XAS']/3600,df_copy[V_col][condition], color='black',s=10, marker="_")
    # axes parameters
    ax.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w')
    # x
    ax.set_xlabel("Time (h)",fontsize=14,labelpad=10)
    margin = 0.02
    full_range_x = df_copy['absolute_time/s_XAS'].max()/3600-df_copy['absolute_time/s_XAS'].min()/3600
    x_min=df_copy['absolute_time/s_XAS'].min()/3600-(full_range_x)*margin/(1-margin*2)
    x_max=df_copy['absolute_time/s_XAS'].max()/3600+(full_range_x)*margin/(1-margin*2)
    #ax.set_xlim(x_min,x_max)
    ax.set_xlim(df_copy['absolute_time/s_XAS'].min()/3600,df_copy['absolute_time/s_XAS'].max()/3600)
    # y left
    label_y_right='Edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y_right,fontsize=14,labelpad=10)
    # y right
    margin = 0.1
    ax1.set_ylabel("Potential vs. Li/Li$^+$ (V)",fontsize=14,labelpad=10)
    full_range_y_right = df_copy[V_col].max()-df_copy[V_col].min()
    y_right_min=df_copy[V_col].min()-(full_range_y_right)*margin/(1-margin*2)
    y_right_max=df_copy[V_col].max()+(full_range_y_right)*2*margin/(1-margin*4)
    ax1.set_ylim(y_right_min,y_right_max)
    ax1.spines['left'].set_color('red')
    # Put a legend to the right of the current axis
    leg = ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), prop={'size': 12})
    return fig

def Eshift_vs_t_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                     width=10,height=6,dotsize=10,alpha=0.5,guideline=False):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    # normalize the colormap with respect to the total number of cycles in the df
    norm = Normalize(vmin=int(df_copy[cycle_col].min()), vmax=int(df_copy[cycle_col].max()))
    cm=plt.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    # build figure
    fig, ax = plt.subplots(figsize=(width,height))
    condition = df_copy[cycle_col].isin(nb_cycle)
    # plot Edge shift
    scatter = ax.scatter(df_copy[condition]['absolute_time/s_XAS']/3600, edge[condition], s=dotsize,
                         c=df_copy[cycle_col][condition], cmap=colormap, norm=norm)
    
    axis_color=sm.to_rgba(df_copy[cycle_col][condition].min())
    ax.spines['left'].set_color(axis_color)
    ax.tick_params(axis='y', colors=axis_color)
    ax.yaxis.label.set_color(axis_color)
    
    # plot potential over time
    ax1 = ax.twinx()
    if guideline:
        ax.plot(df_copy[condition]['absolute_time/s_XAS']/3600,edge[condition],alpha=0.8,linewidth=1.5,color='lightgrey',zorder=0)
        ax1.plot(df_copy[condition]['absolute_time/s_XAS']/3600,df_copy[V_col][condition],
                 alpha=0.8,linewidth=1.5,color='lightgrey',zorder=0)
    ax1.scatter(df_copy[condition]['absolute_time/s_XAS']/3600,df_copy[V_col][condition], color='black',s=6, marker="_")
    # axes parameters
    ax.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w')
    # x
    ax.set_xlabel("Time (h)",fontsize=14,labelpad=10)
    margin = 0.02
    full_range_x = df_copy['absolute_time/s_XAS'].max()/3600-df_copy['absolute_time/s_XAS'].min()/3600
    x_min=df_copy['absolute_time/s_XAS'].min()/3600-(full_range_x)*margin/(1-margin*2)
    x_max=df_copy['absolute_time/s_XAS'].max()/3600+(full_range_x)*margin/(1-margin*2)
    #ax.set_xlim(x_min,x_max)
    ax.set_xlim(df_copy['absolute_time/s_XAS'].min()/3600,df_copy['absolute_time/s_XAS'].max()/3600)
    # y left
    label_y_right='Edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y_right,fontsize=14,labelpad=10)
    # y right
    margin = 0.1
    ax1.set_ylabel("Potential vs. Li/Li$^+$ (V)",fontsize=14,labelpad=10)
    full_range_y_right = df_copy[V_col].max()-df_copy[V_col].min()
    y_right_min=df_copy[V_col].min()-(full_range_y_right)*margin/(1-margin*2)
    y_right_max=df_copy[V_col].max()+(full_range_y_right)*2*margin/(1-margin*4)
    ax1.set_ylim(y_right_min,y_right_max)
    #ax1.set_ylim(2.8,5) # check later
    # put a legend
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
    #if len(nb_cycle)>5:
        #cbar = fig.colorbar(sm)
        #cbar.set_label('Cycle', rotation=270, labelpad=10, fontsize=14)
    #else:
        #leg = ax.legend(loc='upper left',prop={'size': 13})
        #leg = ax.legend(handles=scatter.legend_elements()[0], labels=['cycle '+str(int(i)) for i in nb_cycle], prop={'size': 12})
    #ax.legend(handles=scatter.legend_elements()[0], title="Edge shift", ncol = 3, title_fontsize=12, labelspacing=0.05)
    return fig

def Eshift_vs_t_stacked_alpha(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',width=10,height=6,
                              dotsize=10,guideline=False,hspace=.0):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the merged data from the EC Lab file + the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    fillstyle=['full','none']
    markers = itertools.cycle(('o','o','v','v','X','X','*','*','s','s','P','P')) 
    colors = ['r','b']

    # build figure
    #fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(width,height))
    fig = plt.figure(figsize=(width,height))
    gs = gridspec.GridSpec(2, 1) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    condition = df_copy[cycle_col].isin(nb_cycle)
    counter=1
    # plot potential over time
    if guideline:
        ax0.plot(df_copy[condition]['absolute_time/s_XAS']/3600,df_copy[V_col][condition],
                 alpha=0.8,linewidth=1.5,color='lightgrey',zorder=0)
        ax1.plot(df_copy[condition]['absolute_time/s_XAS']/3600,edge[condition],
                 alpha=0.8,linewidth=1.5,color='lightgrey',zorder=0)
    ax0.scatter(df_copy[condition]['absolute_time/s_XAS']/3600,df_copy[V_col][condition], s=dotsize, c='black')
    # plot Edge shift
    for i, cycle in enumerate(df_copy[cycle_col].unique()):
        if cycle in nb_cycle:
            subdf=df_copy[df_copy[cycle_col]==cycle]
            marker=next(markers)
            for j, subcycle in enumerate(subdf[half_cycle_col].unique()):
                if subdf[subdf[half_cycle_col]==subcycle][oxred_col].mode()[0]==1:
                    subcycle_type='charge'
                    color=colors[0]
                else:
                    subcycle_type='discharge'
                    color=colors[1]
                if guideline:
                    ax1.plot(subdf[subdf[half_cycle_col]==subcycle]['absolute_time/s_XAS']/3600,
                             edge[subdf[subdf[half_cycle_col]==subcycle].index],
                            alpha=0.3,linewidth=1.5,color=color,zorder=counter)
                    counter+=1
                ax1.plot(subdf[subdf[half_cycle_col]==subcycle]['absolute_time/s_XAS']/3600,
                         edge[subdf[subdf[half_cycle_col]==subcycle].index],
                        marker=marker, fillstyle=fillstyle[i%2],color=color,lw=0,
                        label='cycle '+str(int(cycle))+' '+subcycle_type,zorder=counter)
                counter+=1

    # axes parameters
    ax0.minorticks_on()
    ax1.minorticks_on()
    ax0.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax0.tick_params(which="minor", axis="both", direction="in")
    ax1.tick_params(which="minor", axis="both", direction="in")
    # x
    ax1.set_xlabel("Time (h)",fontsize=14,labelpad=10)
    margin = 0.02
    full_range_x = df_copy['absolute_time/s_XAS'].max()/3600-df_copy['absolute_time/s_XAS'].min()/3600
    x_min=df_copy['absolute_time/s_XAS'].min()/3600-(full_range_x)*margin/(1-margin*2)
    x_max=df_copy['absolute_time/s_XAS'].max()/3600+(full_range_x)*margin/(1-margin*2)
    #ax0.set_xlim(x_min,x_max)
    #ax1.set_xlim(x_min,x_max)
    ax0.set_xlim(df_copy['absolute_time/s_XAS'].min()/3600,df_copy['absolute_time/s_XAS'].max()/3600)
    ax1.set_xlim(df_copy['absolute_time/s_XAS'].min()/3600,df_copy['absolute_time/s_XAS'].max()/3600)
    # y left
    label_y_right='Edge @ J='+str(edge_intensity)+' (eV)'
    ax1.set_ylabel(label_y_right,fontsize=14,labelpad=10)
    # y right
    margin = 0.1
    ax0.set_ylabel("Potential vs. Li/Li$^+$ (V)",fontsize=14,labelpad=10)
    full_range_y_right = df_copy[V_col].max()-df_copy[V_col].min()
    y_right_min=df_copy[V_col].min()-(full_range_y_right)*margin/(1-margin*2)
    y_right_max=df_copy[V_col].max()+(full_range_y_right)*2*margin/(1-margin*4)
    ax0.set_ylim(y_right_min,y_right_max)
    # adjust ticks
    ax0.xaxis.set_ticklabels([])
    # remove last tick label and first tick label for the necessary subplots
    yticks = ax1.yaxis.get_major_ticks()
    #yticks[-1].label1.set_visible(False)
    #plt.subplots_adjust(wspace=.0) # no vertical space between plots
    plt.subplots_adjust(hspace=hspace) # no horizontal space between plots
    # Put a legend to the right of the current axis
    leg = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.2), prop={'size': 12})
    return fig

def Eshift_vs_t_stacked_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                             width=10,height=6,dotsize=10,alpha=0.5,hspace=.0):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    # normalize the colormap with respect to the total number of cycles in the df
    norm = Normalize(vmin=int(df_copy[cycle_col].min()), vmax=int(df_copy[cycle_col].max()))

    # build figure
    #fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(width,height))
    fig = plt.figure(figsize=(width,height))
    gs = gridspec.GridSpec(2, 1) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    condition = df_copy[cycle_col].isin(nb_cycle)
    # plot potential over time
    ax0.plot(df_copy[condition]['absolute_time/s_XAS']/3600,df_copy[V_col][condition], color='black')
    # plot Edge shift
    ax1.scatter(df_copy[condition]['absolute_time/s_XAS']/3600, edge[condition], s=dotsize,
                c=df_copy[cycle_col][condition], cmap=colormap, norm=norm)

    # axes parameters
    ax0.minorticks_on()
    ax1.minorticks_on()
    ax0.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax0.tick_params(which="minor", axis="both", direction="in")
    ax1.tick_params(which="minor", axis="both", direction="in")
    # x
    ax1.set_xlabel("Time (h)",fontsize=14,labelpad=10)
    margin = 0.02
    full_range_x = df_copy['absolute_time/s_XAS'].max()/3600-df_copy['absolute_time/s_XAS'].min()/3600
    x_min=df_copy['absolute_time/s_XAS'].min()/3600-(full_range_x)*margin/(1-margin*2)
    x_max=df_copy['absolute_time/s_XAS'].max()/3600+(full_range_x)*margin/(1-margin*2)
    #ax0.set_xlim(x_min,x_max)
    #ax1.set_xlim(x_min,x_max)
    ax0.set_xlim(df_copy['absolute_time/s_XAS'].min()/3600,df_copy['absolute_time/s_XAS'].max()/3600)
    ax1.set_xlim(df_copy['absolute_time/s_XAS'].min()/3600,df_copy['absolute_time/s_XAS'].max()/3600)
    # y left
    label_y_right='Edge @ J='+str(edge_intensity)+' (eV)'
    ax1.set_ylabel(label_y_right,fontsize=14,labelpad=10)
    # y right
    margin = 0.1
    ax0.set_ylabel("Potential vs. Li/Li$^+$ (V)",fontsize=14,labelpad=10)
    full_range_y_right = df_copy[V_col].max()-df_copy[V_col].min()
    y_right_min=df_copy[V_col].min()-(full_range_y_right)*margin/(1-margin*2)
    y_right_max=df_copy[V_col].max()+(full_range_y_right)*2*margin/(1-margin*4)
    ax0.set_ylim(y_right_min,y_right_max)
    # adjust ticks
    ax0.xaxis.set_ticklabels([])
    # remove last tick label and first tick label for the necessary subplots
    yticks = ax1.yaxis.get_major_ticks()
    #yticks[-1].label1.set_visible(False)
    #plt.subplots_adjust(wspace=.0) # no vertical space between plots
    plt.subplots_adjust(hspace=hspace) # no horizontal space between plots
    return fig

def dEshiftdt_vs_t(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                   width=10,height=6,dotsize=10,alpha=0.5):
    '''
    Function to plot edge shift derivative vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    # normalize the colormap with respect to the total number of cycles in the df
    norm = Normalize(vmin=int(df_copy[cycle_col].min()), vmax=int(df_copy[cycle_col].max()))
    cm=plt.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    # build figure
    fig, ax = plt.subplots(figsize=(width,height))
    condition = df_copy[cycle_col].isin(nb_cycle)
    # plot Edge shift
    scatter = ax.plot(df_copy[condition]['absolute_time/s_XAS']/3600,
                      abs(edge[condition].diff()/df_copy[condition]['absolute_time/s_XAS'].diff()/3600), color='blue')
    
    axis_color=sm.to_rgba(df_copy[cycle_col][condition].min())
    ax.spines['left'].set_color(axis_color)
    ax.tick_params(axis='y', colors=axis_color)
    ax.yaxis.label.set_color(axis_color)
    
    # plot potential over time
    ax1 = ax.twinx()

    ax1.plot(df_copy[condition]['absolute_time/s_XAS']/3600,df_copy[V_col][condition], color='black')
    # axes parameters
    ax.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w')
    # x
    ax.set_xlabel("Time (h)",fontsize=14,labelpad=10)
    margin = 0.02
    full_range_x = df_copy['absolute_time/s_XAS'].max()/3600-df_copy['absolute_time/s_XAS'].min()/3600
    x_min=df_copy['absolute_time/s_XAS'].min()/3600-(full_range_x)*margin/(1-margin*2)
    x_max=df_copy['absolute_time/s_XAS'].max()/3600+(full_range_x)*margin/(1-margin*2)
    #ax.set_xlim(x_min,x_max)
    ax.set_xlim(df_copy['absolute_time/s_XAS'].min()/3600,df_copy['absolute_time/s_XAS'].max()/3600)
    # y left
    label_y_right='dEdge @ J='+str(edge_intensity)+'/dt'
    ax.set_ylabel(label_y_right,fontsize=14,labelpad=10)
    # y right
    margin = 0.1
    ax1.set_ylabel("Potential vs. Li/Li$^+$ (V)",fontsize=14,labelpad=10)
    full_range_y_right = df_copy[V_col].max()-df_copy[V_col].min()
    y_right_min=df_copy[V_col].min()-(full_range_y_right)*margin/(1-margin*2)
    y_right_max=df_copy[V_col].max()+(full_range_y_right)*2*margin/(1-margin*4)
    ax1.set_ylim(y_right_min,y_right_max)
    #ax1.set_ylim(2.8,5) # check later
    # put a legend
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
    #if len(nb_cycle)>5:
        #cbar = fig.colorbar(sm)
        #cbar.set_label('Cycle', rotation=270, labelpad=10, fontsize=14)
    #else:
        #leg = ax.legend(loc='upper left',prop={'size': 13})
        #leg = ax.legend(handles=scatter.legend_elements()[0], labels=['cycle '+str(int(i)) for i in nb_cycle], prop={'size': 12})
    #ax.legend(handles=scatter.legend_elements()[0], title="Edge shift", ncol = 3, title_fontsize=12, labelspacing=0.05)
    return fig



###### TO CORRECT FROM HERE

def XANES_vs_t_3D(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',arrows_cycles=True,
                  colormap_potential='viridis',colormap_cycle='plasma',width=12,height=10,alpha=0.5,
                  plot_range=None):
    '''
    Function to plot a 3D graph of all the XAS spectra over time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :arrows_cycles: Puts an arrow on the graph to mark the beginning of each cycle.
    :colormap_potential: Name of the colormap you want to use for the plot (according to voltage). Default is set to 'viridis'.
    :colormap_cycle: Name of the colormap you want to use for the arrows pointing the cycles. Default is set to 'plasma'.  Default is set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :alpha: Opacity of the line collections.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
    # if no cycle number is selected then it just plots all of them
    if len(intensity_col)==0:
        try:
            intensity_col=get_intensity_col(df)
        except:
            raise ValueError("Please define a proper column for the intensity.")
    elif intensity_col not in df.columns:
        raise ValueError("Please define a proper column for the intensity.")
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    energy_col=get_energy_col(df)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    if plot_range:
        if ((len(plot_range)==2) & (type(plot_range)==list)) & (all([isinstance(item, (int,float)) for item in plot_range])):
            rang=plot_range
        else:
            raise ValueError("Not a valid plot_range.")
    else:
        rang=[edge.mean()-20,edge.mean()+40]

    # colormap from Ewe/V
    cm=plt.get_cmap(colormap_potential)
    norm = Normalize(vmin=df_copy[V_col].min(), vmax=df_copy[V_col].max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

    absorption=[]
    energy=[]
    zs = list(df_copy['absolute_time/s_XAS']/3600)

    for i in range(df_copy.shape[0]):
        x_data=list(pd.Series(df_copy[energy_col][i])[pd.Series(df_copy[energy_col][i]).between(rang[0],rang[1])])
        y_data=list(pd.Series(df_copy[intensity_col][i])[pd.Series(df_copy[energy_col][i]).between(rang[0],rang[1])])
        energy.append(x_data)
        absorption.append(y_data)

    profiles = np.array(absorption)
    energies = np.array(energy)
    times = np.array(zs)

    # build plot
    fig = plt.figure(figsize=(width, height), dpi= 80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, projection='3d',proj_type='ortho')

    for i, (p, e, t) in enumerate(zip(profiles,energies,times)):
        ax.plot(e, p, zs=t, zdir='y', 
                zorder=(len(profiles) - i), color=sm.to_rgba(df_copy[V_col])[i], alpha=0.5)
    

    ax.minorticks_on()
    ax.set_xlabel('Energy (eV)', labelpad=10, fontsize=14)
    ax.set_xlim3d(rang[0]-5, rang[1]+5)

    ax.set_ylabel('Time (h)', labelpad=10, fontsize=14)
    ax.tick_params(labelsize=13)
    ax.tick_params(axis='z', pad=6)

    ax.set_ylim3d(0, 50)
    ax.set_zlabel('Normalized absorption', labelpad=13, fontsize=14)
    ax.set_zlim3d(0, 2)
    #ax.view_init(elev=20,azim=260)
    ax.view_init(20,290)
    #ax.set_title('Surface plot')
    cbar = fig.colorbar(sm, shrink=0.5)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Potential vs. Li/Li$^+$ (V)', rotation=90, labelpad=10, fontsize=14)

    df_copy[df_copy[cycle_col]==1].first_valid_index()
    #FIRST ANNOTATION
    cm_cycle=plt.get_cmap(colormap_cycle)
    # color map from the total number of cycles in the DF, not from the lenght of the input nb_cycles
    color_cycle=cm_cycle(np.linspace(0, 1, len(df_copy[cycle_col].unique())))
    for i,cycle in enumerate(df_copy[cycle_col].unique()):
        where=df_copy[df_copy[cycle_col]==cycle].first_valid_index()
        maximum=max(df_copy[intensity_col][where])
        index_max=max(range(len(df_copy[intensity_col][where])), key=(df_copy[intensity_col][where]).__getitem__)
        maximum_energy=df_copy[energy_col][where][index_max]
        when=(df_copy['absolute_time/s_XAS'][where])/3600
        x2, y2, _ = proj3d.proj_transform(maximum_energy,when,maximum, ax.get_proj())
        if arrows_cycles:
            ax.annotate("cycle "+str(int(cycle)), xy = (x2,y2), xytext = (-50, 30), fontsize=12, textcoords = 'offset points',
                        ha = 'center', va = 'bottom', arrowprops = dict(width=0.1,headwidth=7,headlength=8,color=color_cycle[i]))

    def update_position(e):
        for label, x, y, z in labels_and_points:
            x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
            label.xy = x2,y2
            label.update_positions(fig.canvas.renderer)
        fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', update_position)
    return fig

def XANES_vs_t_2D(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                  colormap='tab20b', width=8,height=6,plot_range=None,hlines=False):
    '''
    Function to plot a 2D intensity graph of all the XAS spectra over time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'tab20b'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
    # if no cycle number is selected then it just plots all of them
    if len(intensity_col)==0:
        try:
            intensity_col=get_intensity_col(df)
        except:
            raise ValueError("Please define a proper column for the intensity.")
    elif intensity_col not in df.columns:
        raise ValueError("Please define a proper column for the intensity.")
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    energy_col=get_energy_col(df)
    df_copy=df.copy(deep=True)
    cycle_col=get_cycle_number_col(df_copy)
    half_cycle_col=get_half_cycle_col(df_copy)
    V_col=get_potential_col(df_copy)
    oxred_col=get_oxred_col(df_copy)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)

    if plot_range:
        if ((len(plot_range)==2) & (type(plot_range)==list)) & (all([isinstance(item, (int,float)) for item in plot_range])):
            rang=plot_range
        else:
            raise ValueError("Not a valid plot_range.")
    else:
        rang=[edge.mean()-20,edge.mean()+40]
    
    #sub_df=df[df['cycle number'].isin(nb_cycle)]
    all_absorption=[]
    for i in df_copy.index:
        #x_data=list(pd.Series(df['shifted energy'][i])[pd.Series(df['shifted energy'][i]).between(rang[0],rang[1])])
        y_data=list(pd.Series(df_copy[intensity_col][i])[pd.Series(df_copy[energy_col][i]).between(rang[0],rang[1])])
        #energy.append(x_data)
        #all_absorption.append(y_data)
        all_absorption.append(y_data)
    all_profiles = np.array(all_absorption)
    #energy=[]
    #times=[]
    #zs = list(sub_df['absolute time/h'])
    cmap = plt.get_cmap(colormap)
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    norm = plt.Normalize(all_profiles.min(), all_profiles.max())
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(width,height),gridspec_kw={'width_ratios': [1, 1.4]})
    sub_df=df_copy[df_copy[cycle_col].isin(nb_cycle)].reset_index(drop=True)
    diff_time=sub_df['absolute_time/s_XAS'].diff()/3600
    cut_indexes=list(sub_df[diff_time>diff_time.describe()['mean']+diff_time.describe()['std']*4].index)
    cut_indexes.append(None)
    #if sub_df.index[-1] not in cut_indexes:
    #    cut_indexes.append(sub_df.index[-1])
    begin=0
    for idx in cut_indexes:
        absorption=[]
        energy=[]
        times=[]
        zs = list(sub_df[begin:idx]['absolute_time/s_XAS']/3600)
        for i in sub_df[begin:idx].index:
            x_data=list(pd.Series(sub_df[energy_col][i])[pd.Series(sub_df[energy_col][i]).between(rang[0],rang[1])])
            y_data=list(pd.Series(sub_df[intensity_col][i])[pd.Series(sub_df[energy_col][i]).between(rang[0],rang[1])])
            energy.append(x_data)
            absorption.append(y_data)
            all_absorption.append(y_data)


        profiles = np.array(absorption)
        energies = np.array(energy)
        #times = np.array(zs)
        for time in zs:
            times.append(time*np.ones(profiles.shape[1]))
        times=np.array(times)
        ax0.scatter(sub_df[V_col],sub_df['absolute_time/s_XAS']/3600,color='black',s=2)
        if hlines:
            peaks, _ = find_peaks(sub_df[V_col], prominence=0.1)
            for idx2 in peaks:
                ax0.axhline(sub_df.iloc[idx2]['absolute_time/s_XAS']/3600, color='b', ls='dashed', linewidth=0.7)
                ax1.axhline(sub_df.iloc[idx2]['absolute_time/s_XAS']/3600, color='b', ls='dashed', linewidth=0.7)
        
        im = ax1.pcolormesh(energies, times, profiles, cmap=cmap, norm=norm)
        begin=idx
    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    #cmap = plt.get_cmap(colormap)
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    #norm = plt.Normalize(profiles.min(), profiles.max())

    #fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(width,height),gridspec_kw={'width_ratios': [1, 1.4]})

    #ax0.scatter(sub_df['Ewe/V'],sub_df['absolute time/h'],color='black',s=2)
    ax0.set_xlabel('Potential vs. Li/Li$^+$ (V)',fontsize=14,labelpad=10)
    ax0.set_ylabel('Time (h)',fontsize=14,labelpad=10)
    ax0.tick_params(axis='both', labelsize=13)
    ax0.minorticks_on()

    #ax0.set_xlim(2.9,4.8)
    ax0.set_ylim(0,df_copy['absolute_time/s_XAS'].max()/3600)
    #im = ax1.pcolormesh(energies, times, profiles, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax1,shrink=0.8)
    cbar.set_label('Normalized absorption',fontsize=14,labelpad=10)
    cbar.ax.tick_params(labelsize=13)

    plt.subplots_adjust(wspace=.0)

    ax1.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
    ax1.tick_params(axis='both', labelsize=13)
    ax1.minorticks_on()
    #ax1.grid(visible=True, which='both', axis='both')
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set_xlim(rang[0], rang[1])
    ax1.set_ylim(0,df_copy['absolute_time/s_XAS'].max()/3600)
    
    # add horizontal lines at potential maxima
    #peaks, _ = find_peaks(sub_df['Ewe/V'], prominence=0.1)
    #for idx in peaks:
        #ax0.axhline(df['absolute time/h'][idx], color='b', ls='dashed', linewidth=0.7)
        #ax1.axhline(df['absolute time/h'][idx], color='b', ls='dashed', linewidth=0.7)
    return fig

def XANES_vs_t_2D_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                       colormap='tab20b', width=8,height=6,plot_range=None,hlines=False):
    '''
    Function to plot a 2D intensity graph of all the XAS spectra over time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'tab20b'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
    # if no cycle number is selected then it just plots all of them
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    energy_col=get_energy_col(df)
    df_copy=df.copy(deep=True)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)

    if plot_range:
        if ((len(plot_range)==2) & (type(plot_range)==list)) & (all([isinstance(item, (int,float)) for item in plot_range])):
            rang=plot_range
        else:
            raise ValueError("Not a valid plot_range.")
    else:
        rang=[edge.mean()-20,edge.mean()+40]
    
    #sub_df=df[df['cycle number'].isin(nb_cycle)]
    all_absorption=[]
    for i in df_copy.index:
        #x_data=list(pd.Series(df['shifted energy'][i])[pd.Series(df['shifted energy'][i]).between(rang[0],rang[1])])
        y_data=list(pd.Series(df_copy[intensity_col][i])[pd.Series(df_copy[energy_col][i]).between(rang[0],rang[1])])
        #energy.append(x_data)
        #all_absorption.append(y_data)
        all_absorption.append(y_data)
    all_profiles = np.array(all_absorption)
    #energy=[]
    #times=[]
    #zs = list(sub_df['absolute time/h'])
    cmap = plt.get_cmap(colormap)
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    norm = plt.Normalize(all_profiles.min(), all_profiles.max())
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(width,height),gridspec_kw={'width_ratios': [1, 1.4]})
    for cycle in nb_cycle:
        sub_df=df_copy[df_copy['cycle number']==cycle]
        absorption=[]
        energy=[]
        times=[]
        zs = list(sub_df['absolute time/h'])
        for i in sub_df.index:
            x_data=list(pd.Series(sub_df[energy_col][i])[pd.Series(sub_df[energy_col][i]).between(rang[0],rang[1])])
            y_data=list(pd.Series(sub_df[intensity_col][i])[pd.Series(sub_df[energy_col][i]).between(rang[0],rang[1])])
            energy.append(x_data)
            absorption.append(y_data)
            all_absorption.append(y_data)


        profiles = np.array(absorption)
        energies = np.array(energy)
        #times = np.array(zs)
        for time in zs:
            times.append(time*np.ones(profiles.shape[1]))
        times=np.array(times)
        ax0.scatter(sub_df['Ewe/V'],sub_df['absolute time/h'],color='black',s=2)
        if hlines:
            peaks, _ = find_peaks(sub_df['Ewe/V'], prominence=0.1)
            for idx in peaks:
                ax0.axhline(sub_df['absolute time/h'].iloc[idx], color='b', ls='dashed', linewidth=0.7)
                ax1.axhline(sub_df['absolute time/h'].iloc[idx], color='b', ls='dashed', linewidth=0.7)
        
        im = ax1.pcolormesh(energies, times, profiles, cmap=cmap, norm=norm)
    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    #cmap = plt.get_cmap(colormap)
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    #norm = plt.Normalize(profiles.min(), profiles.max())

    #fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(width,height),gridspec_kw={'width_ratios': [1, 1.4]})

    #ax0.scatter(sub_df['Ewe/V'],sub_df['absolute time/h'],color='black',s=2)
    ax0.set_xlabel('Potential vs. Li/Li$^+$ (V)',fontsize=14,labelpad=10)
    ax0.set_ylabel('Time (h)',fontsize=14,labelpad=10)
    ax0.tick_params(axis='both', labelsize=13)
    ax0.minorticks_on()

    ax0.set_xlim(2.9,4.8)
    ax0.set_ylim(0,df_copy['absolute time/h'].max())
    #im = ax1.pcolormesh(energies, times, profiles, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax1,shrink=0.8)
    cbar.set_label('Normalized absorption',fontsize=14,labelpad=10)
    cbar.ax.tick_params(labelsize=13)

    plt.subplots_adjust(wspace=.0)

    ax1.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
    ax1.tick_params(axis='both', labelsize=13)
    ax1.minorticks_on()
    #ax1.grid(visible=True, which='both', axis='both')
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set_xlim(rang[0], rang[1])
    ax1.set_ylim(0,df_copy['absolute time/h'].max())
    
    # add horizontal lines at potential maxima
    #peaks, _ = find_peaks(sub_df['Ewe/V'], prominence=0.1)
    #for idx in peaks:
        #ax0.axhline(df['absolute time/h'][idx], color='b', ls='dashed', linewidth=0.7)
        #ax1.axhline(df['absolute time/h'][idx], color='b', ls='dashed', linewidth=0.7)
    return fig

##### to do after performing MRC ALS
def Eshift_conc_vs_U_stacked(df,nb_cycle='all',conc_array=[],edge_intensity='inflection',intensity_col='',
                             colormap='plasma',width=15,height=6,linewidth=1.5,dotsize=18,alpha=0.5,wspace=0.0,
                             hspace=0.0,top=0.07):
    '''
    Function to plot edge shift and concentration profile (determined by MCR-ALS toolbox matlab) vs potential.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :conc_array: Numpy array with the concentration profiles.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)

    #colormap according to total number of cycles
    cm=plt.get_cmap(colormap)
    color_dqdv=cm(np.linspace(0, 1, len(df_copy['cycle number'].unique())))
    
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(width, height)

    half_cycles = []
    maxs = []
    mins = []
    widths = []
    mode = []
    for index, row in df_copy.groupby('half cycle').agg({'Ewe/V': ['min', 'max'], 'ox/red':['mean']}).iterrows():
        width=row[1]-row[0]
        if width !=0:
            half_cycles.append(index)
            mins.append(row[0])
            maxs.append(row[1])
            widths.append(width)
            mode.append(round(row[2]))

    gs_top = fig.add_gridspec(nrows=2, ncols=len(half_cycles), width_ratios=widths, top=0.95)

    # PLOTTING THE EDGE EVOLUTION
    art_idx=0
    for col in range(len(half_cycles)):
        condition=df_copy['half cycle']==half_cycles[col]
        index_cm=col//2
        ax0 = fig.add_subplot(gs_top[0,col])
        ax0.plot(df_copy[condition]['Ewe/V'],edge[art_idx:art_idx+len(df_copy[condition]['Ewe/V'])],
                 alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
        ax0.scatter(df_copy[condition]['Ewe/V'],edge[art_idx:art_idx+len(df_copy[condition]['Ewe/V'])],
                    label=str(half_cycles[col]),color=color_dqdv[index_cm],s=dotsize,zorder=1)
        art_idx=art_idx+len(df_copy[condition]['Ewe/V'])
        # remove last tick label and first tick label for the necessary subplots
        #ax.yaxis.set_ticklabels([])
        ax0.get_xaxis().set_ticks([])
        ax0.tick_params(axis='both', labelsize=13)
        ax0.set_xlim(mins[col],maxs[col])
        ax0.set_ylim(edge.min(),edge.max())
        if mode[col]==0:
            ax0.invert_xaxis()
        if col!=0:
            ax0.get_yaxis().set_visible(False)
            ax0.spines['left'].set_linestyle("dashed")
            ax0.spines['left'].set_capstyle("butt")
        else:
            ax0.set_ylabel("Edge @ J="+str(edge_intensity)+" (eV)", fontsize=15, labelpad=10)
            #xticks[0].label1.set_visible(False)

    art_idx=0
    # PLOTTING CONCENTRATION EVOLUTION
    for col in range(len(half_cycles)):
        condition=df_copy['half cycle']==half_cycles[col]
        ax1 = fig.add_subplot(gs_top[1,col])
        for conc in conc_array:
            ax1.plot(df_copy[condition]['Ewe/V'],conc[art_idx:art_idx+len(df_copy[condition]['Ewe/V'])],
                     alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
            ax1.scatter(df_copy[condition]['Ewe/V'],conc[art_idx:art_idx+len(df_copy[condition]['Ewe/V'])],s=dotsize*5/6,zorder=1)
            #ax1.scatter(output_cell4_df[condition]['Ewe/V'],conc[art_idx:art_idx+len(output_cell4_df[condition]['Ewe/V'])],s=15)
        art_idx=art_idx+len(df_copy[condition]['Ewe/V'])

        xticks = ticker.MaxNLocator(round(widths[col]/0.25))

        ax1.xaxis.set_major_locator(xticks)
        ax1.tick_params(axis='both', labelsize=13)
        ax1.set_xlim(mins[col],maxs[col])
        ax1.set_ylim(0,1)
        if mode[col]==0:
            ax1.invert_xaxis()
        if col!=0:
            ax1.get_yaxis().set_visible(False)
            ax1.spines['left'].set_linestyle("dashed")
            ax1.spines['left'].set_capstyle("butt")
        else:
            ax1.set_ylabel("Relative concentration", fontsize=15, labelpad=10)

    gs_base = fig.add_gridspec(nrows=1, ncols=len(half_cycles), width_ratios=widths, top=top,bottom=top-0.01)
    # X AXIS LABEL
    ax2 = fig.add_subplot(gs_base[0,:])
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_color('white')
    ax2.set_xlabel("Potential vs. Li/Li$^+$ (V)", fontsize=15)
    plt.subplots_adjust(wspace=wspace) # no vertical space between plots0.
    plt.subplots_adjust(hspace=hspace)
    return fig

def Eshift_conc_vs_x_stacked(df,nb_cycle='all',conc_array=[],edge_intensity='inflection',intensity_col='',
                             x_col='x',colormap='plasma',width=15,height=6,linewidth=1.5,dotsize=18,alpha=0.5,wspace=0.0,
                             hspace=0.0,top=0.07):
    '''
    Function to plot edge shift and concentration profile (determined by MCR-ALS toolbox matlab) vs potential.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :conc_array: Numpy array with the concentration profiles.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)
    
    if (x_col=='x') & ('calculated x' in df_copy.columns):
        x_col='calculated x'
    elif x_col in df_copy.columns:
        x_col=x_col
    else: raise ValueError('x not found in the dataframe columns.')
    
    #colormap according to total number of cycles
    cm=plt.get_cmap(colormap)
    color_dqdv=cm(np.linspace(0, 1, len(df_copy['cycle number'].unique())))
    
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(width, height)

    half_cycles = []
    maxs = []
    mins = []
    widths = []
    mode = []
    for index, row in df_copy.groupby('half cycle').agg({x_col: ['min', 'max'], 'ox/red':['mean']}).iterrows():
        width=row[1]-row[0]
        if width !=0:
            half_cycles.append(index)
            mins.append(row[0])
            maxs.append(row[1])
            widths.append(width)
            mode.append(round(row[2]))
    print(mode)
    gs_top = fig.add_gridspec(nrows=2, ncols=len(half_cycles), width_ratios=widths, top=0.95)

    # PLOTTING THE EDGE EVOLUTION
    art_idx=0
    for col in range(len(half_cycles)):
        condition=df_copy['half cycle']==half_cycles[col]
        index_cm=col//2
        ax0 = fig.add_subplot(gs_top[0,col])
        ax0.plot(df_copy[condition][x_col],edge[art_idx:art_idx+len(df_copy[condition][x_col])],
                 alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
        ax0.scatter(df_copy[condition][x_col],edge[art_idx:art_idx+len(df_copy[condition][x_col])],
                    label=str(half_cycles[col]),color=color_dqdv[index_cm],s=dotsize,zorder=1)
        art_idx=art_idx+len(df_copy[condition][x_col])
        ax0.get_xaxis().set_ticks([])
        ax0.tick_params(axis='both', labelsize=13)
        ax0.set_xlim(mins[col],maxs[col])
        ax0.set_ylim(edge.min(),edge.max())
        if mode[col]!=0:
            ax0.invert_xaxis()
        if col!=0:
            ax0.get_yaxis().set_visible(False)
            ax0.spines['left'].set_linestyle("dashed")
            ax0.spines['left'].set_capstyle("butt")
        else:
            ax0.set_ylabel("Edge @ J="+str(edge_intensity)+" (eV)", fontsize=15, labelpad=10)
            #xticks[0].label1.set_visible(False)

    art_idx=0
    # PLOTTING CONCENTRATION EVOLUTION
    for col in range(len(half_cycles)):
        condition=df_copy['half cycle']==half_cycles[col]
        ax1 = fig.add_subplot(gs_top[1,col])
        for conc in conc_array:
            ax1.plot(df_copy[condition][x_col],conc[art_idx:art_idx+len(df_copy[condition][x_col])],
                     alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
            ax1.scatter(df_copy[condition][x_col],conc[art_idx:art_idx+len(df_copy[condition][x_col])],s=dotsize*5/6,zorder=1)
        art_idx=art_idx+len(df_copy[condition][x_col])
        xticks = ticker.MaxNLocator(round(widths[col]/0.25))
        ax1.xaxis.set_major_locator(xticks)
        ax1.tick_params(axis='both', labelsize=13)
        ax1.set_xlim(mins[col],maxs[col])
        ax1.set_ylim(0,1)
        if mode[col]!=0:
            ax1.invert_xaxis()
        if col!=0:
            ax1.get_yaxis().set_visible(False)
            ax1.spines['left'].set_linestyle("dashed")
            ax1.spines['left'].set_capstyle("butt")
        else:
            ax1.set_ylabel("Relative concentration", fontsize=15, labelpad=10)

    gs_base = fig.add_gridspec(nrows=1, ncols=len(half_cycles), width_ratios=widths, top=top,bottom=top-0.01)
    # X AXIS LABEL
    ax2 = fig.add_subplot(gs_base[0,:])
    ax2.get_xaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_color('white')
    ax2.set_xlabel("x", fontsize=15)
    plt.subplots_adjust(wspace=wspace) # no vertical space between plots0.
    plt.subplots_adjust(hspace=hspace)
    return fig

def U_Eshift_conc_vs_t_stacked(df,nb_cycle='all',conc_array=[],edge_intensity='inflection',intensity_col='',
                               colormap='plasma',width=15,height=10,greyline=False,linewidth=1.5,dotsize=18,alpha=0.5,hspace=0.0):
    '''
    Function to plot edge shift and concentration profile (determined by MCR-ALS toolbox matlab) vs potential.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :conc_array: Numpy array with the concentration profiles.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)

    #colormap according to total number of cycles
    #cm=plt.get_cmap(colormap)
    #color_dqdv=cm(np.linspace(0, 1, len(df['cycle number'].unique())))
    norm = Normalize(vmin=int(df_copy['cycle number'].min()), vmax=int(df_copy['cycle number'].max()))
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(width, height)


    gs = gridspec.GridSpec(3, 1) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    condition = df_copy['cycle number'].isin(nb_cycle)
    if greyline:
        ax0.plot(df_copy['absolute time/h'][condition],df_copy['Ewe/V'][condition],
                 alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
        ax1.plot(df_copy['absolute time/h'][condition], edge[condition],
                 alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
    # PLOTTING THE POTENTIAL
    ax0.scatter(df_copy['absolute time/h'][condition],df_copy['Ewe/V'][condition], color='black',s=dotsize*3/5, marker="_",
                zorder=1)
    # PLOTTING THE EDGE SHIFT
    ax1.scatter(df_copy['absolute time/h'][condition], edge[condition], s=dotsize, c=df_copy['cycle number'][condition], 
                cmap=colormap, norm=norm, zorder=1)
    # PLOTTING THE CONCENTRATION PROFILE
    for conc in conc_array:
        if greyline:
            ax2.plot(df_copy[condition]['absolute time/h'],conc,alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
        ax2.scatter(df_copy[condition]['absolute time/h'],conc,s=dotsize*5/6,zorder=1)
            
    # axes parameters
    ax0.minorticks_on()
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax0.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax2.tick_params(axis='both', labelsize=13, direction='in')
    ax0.tick_params(which="minor", axis="both", direction="in")
    ax1.tick_params(which="minor", axis="both", direction="in")
    ax2.tick_params(which="minor", axis="both", direction="in")
    ax0.xaxis.set_ticklabels([])
    ax1.xaxis.set_ticklabels([])
    # x
    ax2.set_xlabel("Time (h)",fontsize=14,labelpad=10)
    margin = 0.02
    full_range_x = df_copy['absolute time/h'].max()-df_copy['absolute time/h'].min()
    x_min=df_copy['absolute time/h'].min()-(full_range_x)*margin/(1-margin*2)
    x_max=df_copy['absolute time/h'].max()+(full_range_x)*margin/(1-margin*2)
    #ax0.set_xlim(x_min,x_max)
    #ax1.set_xlim(x_min,x_max)
    #ax2.set_xlim(x_min,x_max)
    ax0.set_xlim(df_copy['absolute time/h'].min(),df_copy['absolute time/h'].max())
    ax1.set_xlim(df_copy['absolute time/h'].min(),df_copy['absolute time/h'].max())
    ax2.set_xlim(df_copy['absolute time/h'].min(),df_copy['absolute time/h'].max())
    # y label
    ax0.set_ylabel('Potential vs. Li/Li$^+$ (V)',fontsize=14,labelpad=10)
    ax1.set_ylabel('Edge @ J='+str(edge_intensity)+' (eV)',fontsize=14,labelpad=10)
    ax2.set_ylabel('Relative concentration',fontsize=14,labelpad=10)
    ax2.set_ylim(0,1)

    plt.subplots_adjust(hspace=hspace) # no horizontal space between plots
    return fig

def U_x_Eshift_conc_vs_t_stacked(df,nb_cycle='all',conc_array=[],edge_intensity='inflection',intensity_col='',
                                 x_col='x',colormap='plasma',width=15,height=15,greyline=False,linewidth=1.5,dotsize=18,
                                 alpha=0.5,hspace=0.0):
    '''
    Function to plot edge shift and concentration profile (determined by MCR-ALS toolbox matlab) vs potential.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :conc_array: Numpy array with the concentration profiles.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Opacity of the points.
    :return: Plot.
    '''
    edge=get_edge(df,intensity_val=edge_intensity,intensity_col=intensity_col)
    df_copy=df.copy(deep=True)
    check_cycles(df_copy)
    nb_cycle=convert_nb_cycle(df_copy,nb_cycle)

    #colormap according to total number of cycles
    #cm=plt.get_cmap(colormap)
    #color_dqdv=cm(np.linspace(0, 1, len(df['cycle number'].unique())))
    norm = Normalize(vmin=int(df_copy['cycle number'].min()), vmax=int(df_copy['cycle number'].max()))
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(width, height)


    gs = gridspec.GridSpec(4, 1) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    condition = df_copy['cycle number'].isin(nb_cycle)
    if greyline:
        ax0.plot(df_copy['absolute time/h'][condition],df_copy['Ewe/V'][condition],
                 alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
        ax1.plot(df_copy['absolute time/h'][condition], edge[condition],
                 alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
    # PLOTTING THE POTENTIAL
    ax0.scatter(df_copy['absolute time/h'][condition],df_copy['Ewe/V'][condition], color='black',s=dotsize*3/5, marker="_",
                zorder=1)
    # PLOTTING X
    ax1.scatter(df_copy['absolute time/h'][condition],df_copy[x_col][condition], color='black',s=dotsize*3/5, marker="_",
                zorder=1)
    # PLOTTING THE EDGE SHIFT
    ax2.scatter(df_copy['absolute time/h'][condition], edge[condition], s=dotsize, c=df_copy['cycle number'][condition], 
                cmap=colormap, norm=norm, zorder=1)
    # PLOTTING THE CONCENTRATION PROFILE
    for conc in conc_array:
        if greyline:
            ax3.plot(df_copy[condition]['absolute time/h'],conc,alpha=0.8,linewidth=linewidth,color='lightgrey',zorder=0)
        ax3.scatter(df_copy[condition]['absolute time/h'],conc,s=dotsize*5/6,zorder=1)
            
    # axes parameters
    ax0.minorticks_on()
    ax1.minorticks_on()
    ax2.minorticks_on()
    ax3.minorticks_on()
    ax0.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax2.tick_params(axis='both', labelsize=13, direction='in')
    ax3.tick_params(axis='both', labelsize=13, direction='in')
    ax0.tick_params(which="minor", axis="both", direction="in")
    ax1.tick_params(which="minor", axis="both", direction="in")
    ax2.tick_params(which="minor", axis="both", direction="in")
    ax3.tick_params(which="minor", axis="both", direction="in")
    ax0.xaxis.set_ticklabels([])
    ax1.xaxis.set_ticklabels([])
    ax2.xaxis.set_ticklabels([])
    # x
    ax3.set_xlabel("Time (h)",fontsize=14,labelpad=10)
    margin = 0.02
    full_range_x = df_copy['absolute time/h'].max()-df_copy['absolute time/h'].min()
    x_min=df_copy['absolute time/h'].min()-(full_range_x)*margin/(1-margin*2)
    x_max=df_copy['absolute time/h'].max()+(full_range_x)*margin/(1-margin*2)
    ax0.set_xlim(df_copy['absolute time/h'].min(),df_copy['absolute time/h'].max())
    ax1.set_xlim(df_copy['absolute time/h'].min(),df_copy['absolute time/h'].max())
    ax2.set_xlim(df_copy['absolute time/h'].min(),df_copy['absolute time/h'].max())
    ax3.set_xlim(df_copy['absolute time/h'].min(),df_copy['absolute time/h'].max())
    # y label
    ax0.set_ylabel('Potential vs. Li/Li$^+$ (V)',fontsize=14,labelpad=10)
    ax1.set_ylabel('x',fontsize=14,labelpad=10)
    ax2.set_ylabel('Edge @ J='+str(edge_intensity)+' (eV)',fontsize=14,labelpad=10)
    ax3.set_ylabel('Relative concentration',fontsize=14,labelpad=10)
    ax3.set_ylim(0,1)

    plt.subplots_adjust(hspace=hspace) # no horizontal space between plots
    return fig