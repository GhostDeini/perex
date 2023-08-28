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



def checkDelim(raw):
    '''
    Function to identify the delimiter in a txt file (, or .).
    '''
    if raw.count('.')>raw.count(','):
        delim='.'
    else:
        delim=','
    return delim

def getSingleXASdict_simplified(filename):
    '''
    Function to get a python dictionary with the information of a XAS file (simplified).
    
    :filename: Complete path of the file.
    :return: Returns a python dictionary with the XAS file information (only certain columns which contain information from the header).
    '''
    f = open(filename, "r")
    linesXAS = f.readlines()
    rawXAS = "".join(linesXAS)
    f.close()
    try:
        norm=next(line for line in linesXAS if '# Normalization' in line)
    except: norm=None
    # check delimiter
    delim=checkDelim(rawXAS)
    # get variables from header
    try:
        startTime=datetime.strptime(next(line for line in linesXAS if '#Time at start=' in line).strip('#Time at start=').strip(), "%Y-%m-%d %H:%M:%S.%f")
    except:
        startTime=datetime.strptime(next(line for line in linesXAS if '#Time at start=' in line).strip('#Time at start=').strip(), "%Y-%b-%d %H:%M:%S.%f")
    elapsedTime=float(next(line for line in linesXAS if '#Time from start (seconds)=' in line).strip('#Time from start (seconds)=').strip())
    sampleT=float(next(line for line in linesXAS if '#Sample temperature (C)=' in line).strip('#Sample temperature (C)=').strip())
    varNames=['filename','start time','elapsed time/s','sample T (C)']
    varVals=[os.path.basename(filename),startTime,elapsedTime,sampleT]
    headerDataDict = {varNames[i]: varVals[i] for i in range(len(varNames))}
    return headerDataDict

def getSingleXASdict(filename):
    '''
    Function to get a python dictionary with the information of a XAS file.
    
    :filename: Complete path of the file.
    :return: Returns a python dictionary with the XAS file information (all columns).
    '''
    f = open(filename, "r")
    linesXAS = f.readlines()
    rawXAS = "".join(linesXAS)
    f.close()
    try:
        norm=next(line for line in linesXAS if '# Normalization' in line)
    except: norm=None
    # check delimiter
    delim=checkDelim(rawXAS)
    # get variables from header
    try:
        startTime=datetime.strptime(next(line for line in linesXAS if '#Time at start=' in line).strip('#Time at start=').strip(), "%Y-%m-%d %H:%M:%S.%f")
    except:
        startTime=datetime.strptime(next(line for line in linesXAS if '#Time at start=' in line).strip('#Time at start=').strip(), "%Y-%b-%d %H:%M:%S.%f")
    elapsedTime=float(next(line for line in linesXAS if '#Time from start (seconds)=' in line).strip('#Time from start (seconds)=').strip())
    varNames=['filename','start time','elapsed time/s']
    varVals=[os.path.basename(filename),startTime,elapsedTime]
    try:
        sampleT=float(next(line for line in linesXAS if '#Sample temperature (C)=' in line).strip('#Sample temperature (C)=').strip())
        varNames.append('sample T (C)')
        varVals.append(sampleT)
    except:
        pass
    # get additional vars if normalised file
    if norm:
        # get number of header lines
        headerIdx=next(i for i,line in enumerate(linesXAS) if '# shifted energy' in line)
        # get variables info from header
        IjumpE0pivot=float(next(line for line in linesXAS if '# I jump at E0_pivot =' in line).strip('# I jump at E0_pivot =').strip())
        E0pivot=float(next(line for line in linesXAS if '# E0_pivot =' in line).strip('# E0_pivot =').strip())
        E0andEshift=next(line for line in linesXAS if '# E0=' in line)
        E0=float(E0andEshift[E0andEshift.find('# E0=')+len('# E0='):E0andEshift.find(' ',E0andEshift.find('# E0=')+len('# E0='))])
        Eshift=float(E0andEshift[E0andEshift.find('Eshift=')+len('Eshift='):E0andEshift.find('\n')])
        varNames=varNames+['E0','E_shift','E0_pivot','I jump at E0_pivot']
        varVals=varVals+[E0,Eshift,E0pivot,IjumpE0pivot]
        # get df for spectrum data
        df=pd.read_csv(filename, header=headerIdx, skip_blank_lines=False, decimal=delim)
        new_columns=df.columns[0].strip('#').strip().split('\t')
        df=df.iloc[:, 0].str.strip().str.split(expand=True)
        df.columns=new_columns
        df=df.astype(float)
    else:
        headerIdx=next(i for i,line in enumerate(linesXAS) if '#Energy' in line)
        df=pd.read_csv(filename, header=headerIdx, sep="\t", skip_blank_lines=False, decimal=delim)
        df.rename(columns={"#Energy":'Energy'},inplace=True)
        if df['Time'].isna().sum()==df.shape[0]:
            df=df.drop(columns='Time')
            df.rename(columns={"iFluo1": "Time"},inplace=True)
    df=df.astype(float)
    # get number of header lines
    spectrumDataDict = df.to_dict('list')
    headerDataDict = {varNames[i]: varVals[i] for i in range(len(varNames))}
    outputDict=headerDataDict|spectrumDataDict
    return outputDict