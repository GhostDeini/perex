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

def getXASfilesdf_samba_txt(XAS_folder,filters=[],recursive=False):
    '''
    Function to get a pandas dataframe with a list of the XAS files in a folder (SAMBA beamline format, files must have a .txt extension).
    
    :XAS_folder: String with the complete path of the folder where the XAS files are.
    :filters: A list of strings containing filter words to select the XAS filenames.
    :recursive: default False. Set to True if you want to look in the folders recursively.
    :return: Returns a pandas dataframe with the XAS filelist information.
    '''
    files = []
    if type(filters)==str:
        filters=[filters]
    if recursive:
        for (dir_path, dir_names, file_names) in os.walk(XAS_folder):
            files.extend(os.path.join(dir_path, f) for f in file_names if f.endswith(".txt") and all(word in f for word in filters))
    else:
        path = XAS_folder+'/*.txt'
        files = glob.glob(path)
        files = [f for f in files if all(word in f for word in filters)]
    if len(files)==0:
        raise ValueError("XAS folder empty. Choose another folder.")
    else:
        XAS_dictlist = []
        for f in sorted(files):
            with open(f, 'r') as openFile:
                lines = openFile.readlines()
                check=True if '#' in lines[0] else False
            if check:
                XAS_dictlist.append(getSingleXASdict_samba(f))
                #XAS_dictlist.append(getSingleXASdict_simplified(f))
        XAS_df = pd.DataFrame(XAS_dictlist)
        #XAS_df=XAS_df.sort_values(by='average time', ignore_index=True)
        a=(XAS_df.applymap(type) == list).all()
        for col in a.index[a]:
            XAS_df[col]=XAS_df[col].apply(lambda x: np.array(x))
        for col in XAS_df.columns:
            if (XAS_df[col].map(type) == np.ndarray).all() or (XAS_df[col].map(type) == list).all():
                if XAS_df[col].map(lambda x: x.size==0).all():
                    XAS_df.drop(columns=col,inplace=True)
    XAS_df.columns = XAS_df.columns.str.replace(' ', '_')
    print('XAS dataframe ready.')
    return XAS_df

def getXASfilesdf_samba_hdf(XAS_folder,filters=[],recursive=False):
    '''
    Function to get a pandas dataframe with a list of the XAS files in a folder (SAMBA beamline format, files must have an .hdf extension).
    
    :XAS_folder: String with the complete path of the folder where the XAS files are.
    :filters: A list of strings containing filter words to select the XAS filenames.
    :recursive: default False. Set to True if you want to look in the folders recursively.
    :return: Returns a pandas dataframe with the XAS filelist information.
    '''
    files = []
    if type(filters)==str:
        filters=[filters]
    if recursive:
        for (dir_path, dir_names, file_names) in os.walk(XAS_folder):
            files.extend(os.path.join(dir_path, f) for f in file_names if f.endswith(".hdf") and all(word in f for word in filters))
    else:
        path = XAS_folder+'/*.hdf'
        files = glob.glob(path)
        files = [f for f in files if all(word in f for word in filters)]
    if len(files)==0:
        raise ValueError("XAS folder empty. Choose another folder.")
    else:
        XAS_df = pd.DataFrame(columns=['filename', 'start time', 'elapsed time/s', 'stop time', 'average time'])
        for i,f in enumerate(sorted(files)):
            fh5 = h5py.File(f, 'r')
            time_at_start = datetime.fromtimestamp(float(fh5['context']['time_at_start'][1].decode('UTF-8')))
            time_at_stop = datetime.fromtimestamp(float(fh5['context']['time_at_stop'][1].decode('UTF-8')))
            XAS_df.loc[i] = [os.path.splitext(os.path.basename(f))[0],time_at_start,(time_at_stop-time_at_start).total_seconds(),time_at_stop,time_at_start+(time_at_stop-time_at_start)/2]
        #XAS_df=XAS_df[cols1+['average time XAS']+cols2]
    XAS_df=XAS_df.sort_values(by='average time', ignore_index=True)
    XAS_df.columns = XAS_df.columns.str.replace(' ', '_')
    print('XAS dataframe ready.')
    return XAS_df

def getSingleXASdict_samba(filename):
    '''
    Function to get a python dictionary with the information of a XAS file (SAMBA beamline format).
    
    :filename: Complete path of the file.
    :return: Returns a python dictionary with the XAS file information (all columns).
    '''
    f = open(filename, "r")
    linesXAS = f.readlines()
    rawXAS = "".join(linesXAS)
    f.close()
    # check delimiter
    delim=checkDelim(rawXAS)
    #
    varNames=['filename', 'header']
    varVals=[os.path.splitext(os.path.basename(filename))[0]]
    # get additional vars - not normalised file
    try:
        headerIdx=next(i for i,line in enumerate(linesXAS) if '# Energy' in line)
        varVals.append(linesXAS[0:headerIdx])
        cols=linesXAS[headerIdx].split(',')
    except:
        headerIdx=next(i for i,line in enumerate(linesXAS) if '# energy' in line)
        varVals.append(linesXAS[0:headerIdx])
        cols=linesXAS[headerIdx].split(';')
    if len(cols)==1:
        cols=linesXAS[headerIdx].split('\t')
    cols_stripped=[s.strip() for s in cols]
    #df=pd.read_csv(filename, header=headerIdx, index_col=False, names=cols_stripped, sep=" ", skip_blank_lines=False, decimal=delim)
    df=pd.read_csv(filename, header=headerIdx, index_col=False, names=cols_stripped, delim_whitespace=True, skip_blank_lines=False, decimal=delim)
    df.rename(columns={'# Energy':'Energy','# energy(eV)':'energy(eV)'},inplace=True)
    df=df.astype(float)
    # get number of header lines
    spectrumDataDict = df.to_dict('list')
    headerDataDict = {varNames[i]: varVals[i] for i in range(len(varNames))}
    outputDict=headerDataDict|spectrumDataDict
    return outputDict

def mergeXASdfs_samba(hdf_df,txt_df):
    r = '({})'.format('|'.join(hdf_df['filename']))
    first_merge = txt_df['filename'].str.extract(r, expand=False).fillna(txt_df['filename'])
    merged_df = txt_df.merge(hdf_df.rename(columns={"filename":'filename_hdf'}), left_on=first_merge,
                             right_on='filename_hdf', how='outer').drop(columns=['filename_hdf'])
    merged_df=merged_df.sort_values(by='average_time', ignore_index=True)
    a=(merged_df.applymap(type) == list).all()
    for col in a.index[a]:
        merged_df[col]=merged_df[col].apply(lambda x: np.array(x))
    return merged_df