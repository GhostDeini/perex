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

# user libraries

from . import plot, pca
from .aux.aux_EC import *
from .aux.aux_XAS import *
from .aux.aux_XAS_samba import *
from .aux.aux_plot import *


# ------------------------------------- getting a panda dataframe from BioLogic mpt files -------------------------------------

def getECdf(*args, with_settings=0, acTime=1):
    '''
    Function to get a pandas dataframe from one or several EC Lab files (must have an .mpt extension).
    
    :*args: String(s) containing the complete path(s) of the file(s). Must be separated by commas.
    :with_settings: Default set to 0. Set to 1 if you want an additional dataframe with the settings that are written in the file header.
    :acTime: Usually, the EC Lab file has a line indicating when the acquisition started. If not, you must set it as a string with the format 'm/d/Y H:M:S.f'
    :return: Returns a pandas dataframe with the data from EC Lab file(s).
    '''
    # get an additional dataframe with the settings or not
    if with_settings==0:
        base_df=getECdf_single(args[0],with_settings,acTime)
    else:
        base_df,main_settings_df=getECdf_single(args[0],with_settings,acTime)
        main_settings_df=main_settings_df[main_settings_df['Ns'].isin(base_df['Ns'].unique())]
    # add a column with the EC filename
    base_df['filename']=os.path.basename(args[0])
    cols = list(base_df.columns)
    cols = [cols[-1]] + cols[:-1]
    base_df = base_df[cols]
    base_df.attrs={'File 1':base_df.attrs}
    # append data from the other EC lab files
    for i,file in enumerate(args[1:]):
        if with_settings==0:
            next_df=getECdf_single(file)
        else:
            next_df,next_settings_df=getECdf_single(file,with_settings)
            next_settings_df=next_settings_df[next_settings_df['Ns'].isin(next_df['Ns'].unique())]
            # consecutive number of sequence
            next_settings_df['Ns']=next_settings_df['Ns']+main_settings_df['Ns'].iloc[-1]+1
            main_settings_df=pd.concat([main_settings_df,next_settings_df],ignore_index=True)
        # add a column with the EC filename
        next_df['filename']=os.path.basename(file)
        cols = list(next_df.columns)
        cols = [cols[-1]] + cols[:-1]
        next_df = next_df[cols]
        # correct half cycles and cycle numbers
        first_oxred=next_df['ox/red'][0]
        if len(next_df['half cycle'].unique())>1:
            halfcycles_first=next_df.loc[0:next_df[next_df['ox/red']!=first_oxred].index[0]-1,'half cycle']
            cycles_first=next_df.loc[0:next_df[next_df['ox/red']!=first_oxred].index[0]-1,'cycle number']
            if halfcycles_first.unique().size>1:
                next_df.loc[0:next_df[next_df['ox/red']!=first_oxred].index[0]-1,'half cycle']=halfcycles_first.unique()[-1]
            if cycles_first.unique().size>1:
                next_df.loc[0:next_df[next_df['ox/red']!=first_oxred].index[0]-1,'cycle number']=cycles_first.unique()[-1]
        # correct (Q-Qo)/mA.h
        next_df['(Q-Qo)/mA.h']=next_df['(Q-Qo)/mA.h']+base_df['(Q-Qo)/mA.h'].iloc[-1]
        if first_oxred==base_df['ox/red'].iloc[-1]:
            next_df['half cycle']=next_df['half cycle']-next_df['half cycle'].min()+base_df['half cycle'].iloc[-1]
            next_df['cycle number']=next_df['cycle number']-next_df['cycle number'].min()+base_df['cycle number'].iloc[-1]
            next_df.loc[next_df[next_df['half cycle']==next_df['half cycle'].min()].index,'Q charge/discharge/mA.h']+=base_df['Q charge/discharge/mA.h'].iloc[-1]
            if first_oxred==1:
                next_df.loc[next_df[next_df['half cycle']==next_df['half cycle'].min()].index,'Q charge/mA.h']+=base_df['Q charge/mA.h'].iloc[-1]
                next_df.loc[next_df[next_df['cycle number']==next_df['cycle number'].min()].index,'Energy charge/W.h']+=base_df['Energy charge/W.h'].iloc[-1]
            elif first_oxred==0:
                next_df.loc[next_df[next_df['half cycle']==next_df['half cycle'].min()].index,'Q discharge/mA.h']+=base_df['Q discharge/mA.h'].iloc[-1]
                next_df.loc[next_df[next_df['cycle number']==next_df['cycle number'].min()].index,'Energy discharge/W.h']+=base_df['Energy discharge/W.h'].iloc[-1]
        else:
            next_df['half cycle']=next_df['half cycle']-next_df['half cycle'].min()+base_df['half cycle'].iloc[-1]+1
            next_df['cycle number']=next_df['cycle number']-next_df['cycle number'].min()+base_df['cycle number'].iloc[-1]+1
        #correct capacity
        next_df.loc[next_df['half cycle']==next_df['half cycle'].unique()[0],'Capacity/mA.h']+=base_df['Capacity/mA.h'].iloc[-1]
        next_df['Ns']+=base_df['Ns'].iloc[-1]+1
        deltaX=next_df['x'][0]-base_df['x'].iloc[-1]
        next_df['x']+=-deltaX
        next_df.attrs={'File '+str(i+2):next_df.attrs}
        newAttrs = base_df.attrs | next_df.attrs
        base_df=pd.concat([base_df, next_df], ignore_index=True)
        base_df.attrs=newAttrs
    print('EC dataframe ready.')
    base_df.columns = base_df.columns.str.replace(' ', '_')
    if with_settings!=0: # get an additional dataframe for settings only
        return base_df, main_settings_df
    else:
        return base_df

# --------------------------- getting pandas dataframes including all XAS txt files in a folder ---------------------------

def getXASdf(XAS_folder,filters=[],recursive=True):
    '''
    Function to get a pandas dataframe with a list of the XAS files in a folder (files must have a .txt extension).
    
    :XAS_folder: String with the complete path of the folder where the XAS files are.
    :filters: A list of strings containing filter words to select the XAS filenames.
    :recursive: default True if you want to look in the folders recursively.
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
    print('Finished file filtering.')
    if len(files)==0:
        raise ValueError("XAS folder empty. Choose another folder.")
    else:
        XAS_dictlist = []
        for i,f in enumerate(files):
            with open(f, 'r') as openFile:
                lines = openFile.readlines()
                check=True if '#' in lines[0] else False
            if check:
                XAS_dictlist.append(getSingleXASdict(f))
                #XAS_dictlist.append(getSingleXASdict_simplified(f))
            if (i%500==0) and (i>0):
                print(str(i)+" files loaded.")
        print("All files loaded.")
        XAS_df = pd.DataFrame(XAS_dictlist)
        XAS_df=XAS_df.sort_values(by='start time', ignore_index=True)
        XAS_df['stop time']=XAS_df['start time']+pd.TimedeltaIndex(XAS_df['elapsed time/s'], unit='S')
        XAS_df.drop_duplicates(subset=XAS_df.columns[:3],inplace=True)
        XAS_df=XAS_df.sort_values(by='stop time', ignore_index=True)
        cols1=list(XAS_df.columns[:XAS_df.columns.get_loc('elapsed time/s')+1])
        cols2=list(XAS_df.columns[XAS_df.columns.get_loc('elapsed time/s')+2:-1])
        #XAS_df=XAS_df[cols1+['average time XAS']+cols2]
        # convert lists to numpy arrays if possible
        a=(XAS_df.applymap(type) == list).all()
        for col in a.index[a]:
            XAS_df[col]=XAS_df[col].apply(lambda x: np.array(x))
    print('XAS dataframe ready.')
    XAS_df.columns = XAS_df.columns.str.replace(' ', '_')
    #XAS_df['absolute_time/s']=(XAS_df['stop_time']-XAS_df['stop_time'][0]).dt.total_seconds()+XAS_df['elapsed_time/s']
    XAS_df['absolute_time/s']=(XAS_df['start_time']-XAS_df['start_time'][0]).dt.total_seconds()+XAS_df['elapsed_time/s']
    return XAS_df

# --------------------------- getting pandas dataframes including all Raman txt files in a folder ---------------------------

def getRAMANdf(csv_path,filename_col='filename', datetime_col='acquisition_date_time'):
    '''
    Function to get a pandas dataframe with Raman data from files listed in the .csv reference file. The files in the list must be located in the same folder and must have a .txt extension).
    :csv_path: String with the complete path of the .csv reference file.
    :return: Returns a pandas dataframe with the Raman filelist information (this the original csv df + the arrays from the Raman data).
    '''
    csv_df=pd.read_csv(csv_path)
    csv_df.rename(columns={datetime_col: 'acquisition_datetime'}, inplace=True)
    raman_dictlist = []
    for f in csv_df[filename_col]:
        complete_filename=os.path.join(os.path.dirname(csv_path),
                                       os.path.splitext(os.path.basename(f.replace('\\', '/')))[0]+'.txt')
        if os.path.exists(complete_filename):
            one_raman_df=pd.read_csv(complete_filename,names=['raman_shift/cm-1','intensity'])
            raman_data_dict=one_raman_df.to_dict('list')
            raman_data_dict[filename_col]=f
            raman_data_dict['filename_txt']=os.path.splitext(os.path.basename(f.replace('\\', '/')))[0]+'.txt'
            raman_dictlist.append(raman_data_dict)
    spectra_df=pd.DataFrame(raman_dictlist)
    raman_df=pd.merge(csv_df, spectra_df, how='left', on=[filename_col])
    for col in ['raman_shift/cm-1','intensity']:
        raman_df[col] = raman_df[col].apply(lambda d: d if isinstance(d, list) else [])
    raman_df['filename_txt'] = raman_df['filename_txt'].fillna('File not found.')
    # convert lists to np arrays
    a=(raman_df.applymap(type) == list).all()
    for col in a.index[a]:
        raman_df[col]=raman_df[col].apply(lambda x: np.array(x))
    raman_df['acquisition_datetime']=pd.to_datetime(raman_df['acquisition_datetime'], format='%m/%d/%Y %I:%M:%S %p')
    print('Raman dataframe ready.')
    raman_df.columns = raman_df.columns.str.replace(' ', '_')
    return raman_df

# ------------------------- getting pandas dataframes including all XAS txt files in a folder - SAMBA -------------------------

def getXASdf_samba(path1,path2=None,filters1=[],filters2=[],recursive=False):
    '''
    Function to get a pandas dataframe with the information of a set of XAS files (SAMBA beamline format).
    
    :path1: String with the complete path of the folder where the XAS files are (can be hdf, or txt, or both if they are all in the same folder).
    :path2: String with the complete path of the folder where the XAS files are (if the first one was for the hdf files, then the second path must contain the txt files, or viceversa).
    :filters1: A list of strings containing filter words to select the XAS filenames in path1.
    :filters2: A list of strings containing filter words to select the XAS filenames in path2.
    :recursive: default False. Set to True if you want to look in the folders recursively.
    :return: Returns a pandas dataframe with the XAS filelist information.
    '''
    if path2==None:
        path1_hdf = path1+'/*.hdf'
        path1_txt = path1+'/*.txt'
        files_hdf = glob.glob(path1_hdf)
        files_txt = glob.glob(path1_txt)
        files_hdf = [f for f in files_hdf if all(word in f for word in filters1)]
        files_txt = [f for f in files_hdf if all(word in f for word in filters1)]
        if len(files_hdf)==len(files_txt):
            hdf_df=getXASfilesdf_samba_hdf(path1,filters1,recursive)
            txt_df=getXASfilesdf_samba_txt(path1,filters1,recursive)
            merged_df=mergeXASdfs_samba(hdf_df,txt_df)
        try:
            merged_df=getXASfilesdf_samba_hdf(path1,filters1,recursive)
        except:
            merged_df=getXASfilesdf_samba_txt(path1,filters1,recursive)
    else:
        try:
            hdf_df=getXASfilesdf_samba_hdf(path1,filters1,recursive)
            txt_df=getXASfilesdf_samba_txt(path2,filters2,recursive)
        except:
            hdf_df=getXASfilesdf_samba_hdf(path2,filters2,recursive)
            txt_df=getXASfilesdf_samba_txt(path1,filters1,recursive)
        merged_df=mergeXASdfs_samba(hdf_df,txt_df)
        merged_df.columns = merged_df.columns.str.replace(' ', '_')
    return merged_df

def getXASdf_samba_beta(hdf_path,txt_path): # old version
    hdf_df=getXASfilesdf_samba_hdf(hdf_path)
    txt_df=getXASfilesdf_samba_txt(txt_path)
    r = '({})'.format('|'.join(hdf_df['filename']))
    first_merge = txt_df['filename'].str.extract(r, expand=False).fillna(txt_df['filename'])
    #merged_df = hdf_df.merge(txt_df.drop('filename', 1), left_on='filename', right_on=first_merge, how='outer')
    merged_df = txt_df.merge(hdf_df.rename(columns={"filename":'filename hdf'}), left_on=first_merge, right_on='filename hdf', how='outer').drop(columns=['filename hdf'])
    merged_df=merged_df.sort_values(by='average time', ignore_index=True)
    return merged_df

# ------------------------- merging data from two different dataframes based on time -------------------------

def merge_dfs(data1_df,data2_df,data1_type='EC',data2_type='XAS',
              acTime_col_data1='acquisition_datetime',acTime_col_data2='stop_time',
              startTime_col_data2='start_time',interp=False,tol='10 min'):
    '''
    Function to merge EC or Raman and XAS-files dataframes based on datetime.
    Reference time comes from XAS experiment. All EC data will be interpolated.
    
    :data1_df: Pandas dataframe with the data from the EC Lab file or from a list of Raman spectra.
    :data2_df: Pandas dataframe with the data from several XAS files.
    :data1_type: Specifies the nature of the dataset 1 (if it's EC or Raman).
    :data2_type: Specifies the nature of the dataset 2 (XAS by default).
    :acTime_col_data1: Default set to 'acquisition_datetime' column.
    :acTime_col_data2: Default set to 'stop time' column (start time + ellapsed time) but you can edit if necessary.
    :startTime_col_data2: Name of the start time column of the XAS dataframe.
    :interp: Default set to True.
    :tol: If interpolation is not wanted, the time tolerance for merging both dataframes.
    :return: Returns a merged dataframe with both informations.
    '''
    # merge electro and XAS files dataframes based on approximate datetime
    # check if datetime column exists in EC df
    if acTime_col_data1 not in data1_df.columns:
        raise ValueError("Cannot proceed without datetime column.")
    # convert 'acquisition_datetime' column string values in EC df to datetime format
    if pd.api.types.is_datetime64_any_dtype(data1_df[acTime_col_data1].dtype)==False:
        data1_df[acTime_col_data1]=pd.to_datetime(data1_df[acTime_col_data1])
    
    new_data1_cols_dict = {}
    for col in data1_df.columns:
        new_data1_cols_dict[col]=col.replace(col,col+"_"+data1_type)
    new_data2_cols_dict = {}
    for col in data2_df.columns:
        new_data2_cols_dict[col]=col.replace(col,col+"_"+data2_type)
    # cols dtype int
    a=(data1_df.applymap(type)==int).all()
    # cols dtype string
    b=(data1_df.applymap(type)==str).all()
    # cols dtype float
    c=(data1_df.applymap(type)==float).all()
    # cols dtype np.ndarray
    d=(data1_df.applymap(type)==np.ndarray).all()
    
    # make a copy of the data1 dataframe and interpolate values using datetimes from XAS files dataframe as reference
    temp_data1_df = data1_df.copy()
    cycle_col=get_cycle_number_col(temp_data1_df)
    temp_data1_df[cycle_col]=temp_data1_df[cycle_col].interpolate(method='pad')
    # drop trailing (extreme) nan rows
    temp_data1_df.dropna(axis=0,subset=[cycle_col],inplace=True)
    temp_data1_df.reset_index(drop=True, inplace=True)
    # convert columns with dtype int to dtype float (better for np array handling)
    temp_data1_df[a.index[a]] = temp_data1_df[a.index[a]].astype(float)
    # append datetimes from data2_df
    newcol=pd.concat([temp_data1_df[acTime_col_data1],data2_df[acTime_col_data2].rename(acTime_col_data1)],
                     ignore_index=True,join='outer')
    temp_data1_df.drop(columns=acTime_col_data1,inplace=True)
    temp_data1_df=temp_data1_df.join(newcol, how='right')
    ###### DEPRECATED METHOD
    #for i in data2_df[acTime_col_data2]:
    #    temp_data1_df = temp_data1_df.append({acTime_col_data1: i}, ignore_index=True)
    ######
    temp_data1_df = temp_data1_df.sort_values(by=acTime_col_data1,ignore_index=True)
    # nans indexes
    nan_idxs=temp_data1_df[temp_data1_df['filename'].isnull()].index.tolist()
    not_nan_idxs=temp_data1_df[temp_data1_df['filename'].isnull()==False].index.tolist()
    # x reference (absolute time from datetime)
    abstime = (temp_data1_df[acTime_col_data1]-temp_data1_df[acTime_col_data1][0]).dt.total_seconds()
    # interpolations
    
    # ints and strings
    for col in (a.index[a].to_list()+c.index[c].to_list()):
        temp_data1_df[col]=temp_data1_df[col].interpolate(method='pad')
    
    # strings
    #for col in b.index[b]:
    #    temp_data1_df[col]=temp_data1_df[col].interpolate(method='pad')
    # floats
    #num_matrix=temp_data1_df[a.index[a].to_list()+c.index[c].to_list()].to_numpy()
    ffloat_cols=[elem for elem in c.index[c].to_list() if 'cycle_number' not in elem.lower().replace(' ','_')]
    num_matrix=temp_data1_df[ffloat_cols].to_numpy()
    for i,col in enumerate(num_matrix.T):
        num_matrix[:,i][nan_idxs]= np.interp(abstime[nan_idxs], abstime[not_nan_idxs], num_matrix[:,i][not_nan_idxs])
    #temp_data1_df[a.index[a].to_list()+c.index[c].to_list()]=num_matrix
    temp_data1_df[ffloat_cols]=num_matrix
    # arrays (2D interpolation)
    # WARNING! this method assumes that ALL the arrays have the same x scale (e.g. same raman_shift/cm-1 or same energy scale)
    # those who don't share the same scale will be interpolated as well
    
    # non-empty array indexes
    #time_arr = (temp_data1_df[acTime_col_data1]-temp_data1_df[acTime_col_data1][0]).dt.total_seconds()
    for arr_col in d.index[d]:
        # row indexes with arrays
        np_arr_idxs=temp_data1_df[temp_data1_df[arr_col].isnull()==False].index.tolist()
        # remove elements with different scale (or when shape is 0)
        all_shapes=[]
        for arr in temp_data1_df[arr_col][np_arr_idxs]:
            all_shapes.append(arr.shape[0])
        mode_shape=stats.mode(all_shapes)[0][0]
        to_remove=[]
        for i,arr in enumerate(temp_data1_df[arr_col][np_arr_idxs]):
            if arr.shape[0]!=mode_shape:
                to_remove.append(np_arr_idxs[i])
        np_arr_idxs = [e for e in np_arr_idxs if e not in to_remove]
        # stack arrays into a 2D matrix
        int_arr=temp_data1_df.iloc[np_arr_idxs,:][arr_col][np_arr_idxs[0]]
        for i,row in temp_data1_df.iloc[np_arr_idxs[1:],:].iterrows():
            int_arr=np.vstack((int_arr,row[arr_col]))
        art_arr = np.arange(0,int_arr.shape[1])
        interp_grid = RegularGridInterpolator((abstime[np_arr_idxs],art_arr), int_arr,
                                         bounds_error=False, fill_value=None)
        other_idxs = [e for e in abstime.index.to_list() if e not in np_arr_idxs]
        xx = abstime[other_idxs]
        yy = art_arr
        X, Y = np.meshgrid(xx, yy, indexing='ij')
        # interpolator
        int_arr_interpol=interp_grid((X, Y))
        # complete df with interpolated arrays
        temp_dict=dict({arr_col:int_arr.tolist()+int_arr_interpol.tolist()})
        new_idx_list=np_arr_idxs+other_idxs
        other_df=pd.DataFrame(temp_dict,index=new_idx_list)
        other_df.sort_index(inplace=True)
        temp_data1_df[arr_col]=other_df
    # reconvert float (from group a) back to int
    try:
        temp_data1_df[a.index[a]] = temp_data1_df[a.index[a]].astype(int)
    except:
        pass
    #temp_data1_df[acTime_col_data1] = pd.to_datetime(temp_data1_df[acTime_col_data1], format='%Y-%b-%d %H:%M:%S.%f')
    # merge the info from both dfs
    if interp:
        # add indication of interpolation column
        temp_data1_df['interpolated?']='no'
        temp_data1_df.loc[nan_idxs,'interpolated?']='yes'
        merged_df = pd.merge_asof(data2_df.rename(columns=new_data2_cols_dict),
                                  temp_data1_df.rename(columns=new_data1_cols_dict),
                                  left_on=new_data2_cols_dict[acTime_col_data2],
                                  right_on=new_data1_cols_dict[acTime_col_data1],
                                  direction='nearest')
    else:
        merged_df = pd.merge_asof(data2_df.rename(columns=new_data2_cols_dict),
                                  data1_df.rename(columns=new_data1_cols_dict),
                                  left_on=new_data2_cols_dict[acTime_col_data2],
                                  right_on=new_data1_cols_dict[acTime_col_data1],
                                  direction='nearest',
                                  tolerance=pd.Timedelta(tol))

    cycle_col=get_cycle_number_col(merged_df)
    merged_df.dropna(how='all', axis=0, subset=[cycle_col], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    elapsed_col=next(col for col in merged_df.columns if 'elapsed' in col)
    merged_df['absolute_time/s_'+data1_type] = (merged_df[new_data1_cols_dict[acTime_col_data1]]-
                                                merged_df[new_data2_cols_dict[startTime_col_data2]][0]).dt.total_seconds()
    merged_df['absolute_time/s_'+data2_type] = (merged_df[new_data2_cols_dict[acTime_col_data2]]-
                                                merged_df[new_data2_cols_dict[startTime_col_data2]][0]).dt.total_seconds()
    #merged_df['absolute_time/min'] = merged_df['absolute time/s']/60
    #merged_df['absolute_time/h'] = merged_df['absolute time/min']/60
    #merged_df['start time XAS'] = merged_df['start time XAS'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    #merged_df['stop time XAS'] = merged_df['stop time XAS'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    if merged_df.shape[0]==0:
        raise ValueError("Could not correctly merge the EC file to a list of XAS files. Please check the folder selection.")
    else:
        for col in merged_df.columns:
            if pd.api.types.is_datetime64_any_dtype(merged_df[col].dtype):
                merged_df[col] = merged_df[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
                merged_df[col] = pd.to_datetime(merged_df[col], format='%Y-%m-%d %H:%M:%S.%f')
    print('Merged dataframe ready.')
    return merged_df

# -------------------------------------------- extract data to txt files --------------------------------------------
def simplified_output(merged_df,cols,output_name,extension):
    #merged_df.to_csv(output_name+extension, index=False, sep='\t')
    print(output_name+extension)
    merged_df[cols].to_csv(output_name+extension, index=False, sep='\t', mode='w+')
    print('Output file succesfully created!')
    return

def segmented_output(merged_df,cols,output_name,extension):
    #merged_df.to_csv(output_name+extension, index=False, sep='\t')
    print(output_name+extension)
    merged_df[cols].to_csv(output_name+extension, index=False, sep='\t', mode='w+')
    for half_cycle in merged_df['half cycle'].unique():
        sub_df=merged_df[merged_df['half cycle']==half_cycle]
        cycle=sub_df['cycle number'].mode()[0]
        if sub_df['ox/red'].mode()[0]==1:
            sub_df[cols].to_csv(output_name+'_charge'+str(int(cycle))+extension, index=False, sep='\t', mode='w+')
            print(output_name+'_charge'+str(int(cycle))+extension)
        else:
            sub_df[cols].to_csv(output_name+'_discharge'+str(int(cycle))+extension, index=False, sep='\t', mode='w+')
            print(output_name+'_discharge'+str(int(cycle))+extension)
    print('Output file succesfully created!')
    return

def simplified_output_beta(EC_filename,XAS_folder,output_name,EC_starttime=1):
    electrodf=getECdf(EC_filename,acTime=EC_starttime)
    print('EC data obtained.')
    XASdf=getXASdf(XAS_folder)
    XASdf=XASdf[['filename','acquisition_datetime','elapsed time/s','sample T (C)','average time XAS']]
    print('XAS filenames obtained.')
    merged=mergeEC_XASdfs(electrodf,XASdf)
    merged.to_csv(output_name, index=False, sep='\t')
    print('Output file succesfully created!')
    return
#  ------------------------------------------------------- pre-treatment -------------------------------------------------------
def interpol_energy(XAS_path,grid=None,output_prefix='int',extension='.txt'):
    '''
    Function to create a new set of XAS files with an interpolated energy grid. The new files will be located in a folder named 'interpolated' inside of the working folder.
    
    :XAS_path: String with the complete path of the folder where the XAS files are.
    :grid: List containing the parameters of the desired energy grid. It should be defined as [energy,step,energy,step,energy...].
    :output_prefix: Prefix of the output filenames. Default set to 'int'.
    :extension: Output file extension. Default set to txt.
    :return: Creates a set of new files with a new common energy grid.
    '''
    XAS_df=getXASdf_samba(XAS_path)
    energy_col=get_energy_col(XAS_df)
    if grid==None:
        grid=[XAS_df[energy_col][0].min(),0.5,XAS_df[energy_col][0].max()]
    try:
        chunks = [grid[x:x+3] for x in range(0, len(grid)-1, 2)]
        decimals = max([abs(Decimal(str(x)).as_tuple().exponent) for x in grid])
        for chunk in chunks:
            if chunk[2]<=chunk[0] or Decimal(str((chunk[2]-chunk[0])))%Decimal(str(chunk[1]))!=0:
                raise ValueError('Please specify a valid energy grid.')
    except:
        raise ValueError('Please specify a valid energy grid.')
    output_dir=os.path.join(XAS_path, 'interpolated')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # get all columns to interpolate (except energy)
    a=(XAS_df.applymap(type) == np.ndarray).all()
    cols_to_int=a.index[a].to_list()
    cols_to_int.remove(energy_col)
    if 'header' in cols_to_int:
        cols_to_int.remove('header')
    # define energy scale
    E_fine=[]
    for chunk in chunks:
        E_fine.extend(np.arange(chunk[0],chunk[2],chunk[1]))
    for index, row in XAS_df.iterrows():
        for col in cols_to_int:
            interpol = interp1d(row[energy_col], row[col], fill_value='extrapolate')
            int_values=interpol(E_fine)
            #XAS_df.loc[index, col] = int_values
            row[col] = int_values
        #XAS_df.loc[index, energy_col] = np.around(E_fine,decimals)
        row[energy_col] = np.around(E_fine,decimals)
        
        output = os.path.join(output_dir, output_prefix+'_'+row['filename']+extension)
        # get np matrix
        matrix=[]
        cols=[]
        cols_not=[]
        for col, data in row.iteritems():
            if type(data)==np.ndarray and col!='header':
                matrix.append(data)
                cols.append(col)
            else:
                cols_not.append(col)
        matrix=np.transpose(np.array(matrix))
        np.savetxt(output, matrix, delimiter='\t', header='\t'.join(col for col in cols))
        header=[]
        header.append('# File generated with ECXAS python library.\n')
        header.append('# New interpolation grid: '+str(grid)+'\n\n')
        int_file = open(output, "r")
        original_text = int_file.read()
        int_file.close()
        with open(output, 'w') as int_file:
            if 'header' in XAS_df.columns:
                header.extend(list(row['header']))
            int_file.writelines(header)
            int_file.write(original_text)
    print('Interpolated files ready.')
    return
# ---------------------------------------------------------- fitting ----------------------------------------------------------
def fit_lines(x,y,points):
    res_matrix=[]
    idx_matrix=[0]
    i=0
    for point in points[1:]:
        f = x.sub(point).abs().idxmin()
        if point==points[-1]:
            f+=1
        section_x=np.array(x[i:f+1])
        section_y=np.array(y[i:f+1])
        res_matrix.append(stats.linregress(section_x,section_y))
        idx_matrix.append(f)
        i=f
    return res_matrix, idx_matrix

def fit_Eshift_vs_t(df,points=None,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                    colormap='plasma',width=10,height=6,dotsize=10,alpha=0.5):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

    # normalize the colormap with respect to the total number of cycles in the df
    norm = Normalize(vmin=int(df_copy['cycle number'].min()), vmax=int(df_copy['cycle number'].max()))
    cm=plt.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    # build figure
    fig, ax = plt.subplots(figsize=(width,height))
    condition = df_copy['cycle number'].isin(nb_cycle)
    # plot Edge shift
    scatter = ax.scatter(df_copy['absolute time/h'][condition], edge[condition], s=dotsize, c=df_copy['cycle number'][condition], cmap=colormap, norm=norm)
    
    axis_color=sm.to_rgba(df_copy['cycle number'][condition].min())
    ax.spines['left'].set_color(axis_color)
    ax.tick_params(axis='y', colors=axis_color)
    ax.yaxis.label.set_color(axis_color)
    
    # plot fit lines
    if points!=None:
        res,idxs=fit_lines(df_copy['absolute time/h'],edge,points)
        for i,segment in enumerate(res):
            section_x=df_copy['absolute time/h'][idxs[i]:idxs[i+1]+1]
            ax.plot(section_x, segment.intercept + segment.slope*section_x, 'r', label='fitted line '+str(i))

    # plot potential over time
    ax1 = ax.twinx()
    ax1.scatter(df_copy['absolute time/h'][condition],df_copy['Ewe/V'][condition], color='black',s=6, marker="_")
    # axes parameters
    ax.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w')
    # x
    ax.set_xlabel("Time (h)",fontsize=14,labelpad=10)
    margin = 0.02
    full_range_x = df_copy['absolute time/h'].max()-df_copy['absolute time/h'].min()
    x_min=df_copy['absolute time/h'].min()-(full_range_x)*margin/(1-margin*2)
    x_max=df_copy['absolute time/h'].max()+(full_range_x)*margin/(1-margin*2)
    ax.set_xlim(x_min,x_max)
    # y left
    label_y_right='Edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y_right,fontsize=14,labelpad=10)
    # y right
    margin = 0.1
    ax1.set_ylabel("Potential vs. Li/Li$^+$ (V)",fontsize=14,labelpad=10)
    full_range_y_right = df_copy['Ewe/V'].max()-df_copy['Ewe/V'].min()
    y_right_min=df_copy['Ewe/V'].min()-(full_range_y_right)*margin/(1-margin*2)
    y_right_max=df_copy['Ewe/V'].max()+(full_range_y_right)*2*margin/(1-margin*4)
    ax1.set_ylim(y_right_min,y_right_max)
    # put a legend
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(colormap), norm=norm)
    return fig

def fit_only_Eshift_vs_t(df,points=None,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                         colormap='plasma',width=10,height=6,linewidth=2,alpha=0.5):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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
    
    # normalize the colormap with respect to the total number of cycles in the df
    norm = Normalize(vmin=int(df_copy['cycle number'].min()), vmax=int(df_copy['cycle number'].max()))
    #cm=plt.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    # build figure
    fig, ax = plt.subplots(figsize=(width,height))
    condition = df_copy['cycle number'].isin(nb_cycle)
    # plot Edge shift
    #scatter = ax.scatter(df['absolute time/h'][condition], edge[condition], s=dotsize, c=df['cycle number'][condition], cmap=colormap, norm=norm)
    
    axis_color=sm.to_rgba(df_copy['cycle number'][condition].min())
    ax.spines['left'].set_color(axis_color)
    ax.tick_params(axis='y', colors=axis_color)
    ax.yaxis.label.set_color(axis_color)
    
    # plot fit lines
    if points!=None:
        res,idxs=fit_lines(df_copy['absolute time/h'],edge,points)
        for i,segment in enumerate(res):
            section_x=df_copy['absolute time/h'][idxs[i]:idxs[i+1]+1]
            cycle=df_copy[idxs[i]:idxs[i+1]]['cycle number'].mode()[0]
            color_idx=np.where(df_copy['cycle number'].unique()==cycle)[0][0]
            ax.plot(section_x, segment.intercept + segment.slope*section_x, 'r', label='fitted line '+str(i), linewidth=linewidth, color=color_dqdv[color_idx])

    # plot potential over time
    ax1 = ax.twinx()
    ax1.scatter(df_copy['absolute time/h'][condition],df_copy['Ewe/V'][condition], color='black',s=6, marker="_")
    #ax1.plot(df['absolute time/h'][condition],df['Ewe/V'][condition], color='black')
    #for cycle in nb_cycle:
    #    ax1.plot(df['absolute time/h'][df['cycle number']==cycle],df['Ewe/V'][df['cycle number']==cycle], color='black')
    # axes parameters
    ax.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w')
    # x
    ax.set_xlabel("Time (h)",fontsize=14,labelpad=10)
    margin = 0.02
    full_range_x = df_copy['absolute time/h'].max()-df_copy['absolute time/h'].min()
    x_min=df_copy['absolute time/h'].min()-(full_range_x)*margin/(1-margin*2)
    x_max=df_copy['absolute time/h'].max()+(full_range_x)*margin/(1-margin*2)
    ax.set_xlim(x_min,x_max)
    # y left
    label_y_right='Edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y_right,fontsize=14,labelpad=10)
    # y right
    margin = 0.1
    ax1.set_ylabel("Potential vs. Li/Li$^+$ (V)",fontsize=14,labelpad=10)
    full_range_y_right = df_copy['Ewe/V'].max()-df_copy['Ewe/V'].min()
    y_right_min=df_copy['Ewe/V'].min()-(full_range_y_right)*margin/(1-margin*2)
    y_right_max=df_copy['Ewe/V'].max()+(full_range_y_right)*2*margin/(1-margin*4)
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
