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

#others
from collections import OrderedDict

# plotting
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

#test
import test


# -------------------------------------------- getting pandas dataframes from files --------------------------------------------
def getECdf_single(EC_filename, with_settings=0, acTime=1):
    '''
    Function to get a pandas dataframe from an EC Lab file (must have an .mpt extension).
    
    :EC_filename: Complete path of the EC Lab file.
    :with_settings: Default set to 0. Set to 1 if another dataframe with the settings written in the file header is desired.
    :acTime: Usually, the EC Lab file has a line indicating when the acquisition started. If not, you must set it as a string with the format 'm/d/Y H:M:S.f'
    :return: Returns a pandas dataframe with the data from the EC Lab file.
    '''
    # reading the file
    filename, file_extension = os.path.splitext(EC_filename)
    if file_extension!='.mpt':
        raise ValueError("File type not suppported. It must have an .mpt extension.")
    f = open(EC_filename, "r", encoding="ISO-8859-1")
    linesEC = f.readlines()
    rawEC = "".join(linesEC)
    f.close()
    # check which delimiter is used for decimals
    delim=checkDelim(rawEC)
    # get number of header lines
    mainHeaderIdx=int(re.findall("\d+", next(line for line in linesEC if 'Nb header lines' in line))[0])-1
    rawHeader = "".join(linesEC[:mainHeaderIdx])
    # MAIN dataframe
    df=pd.read_csv(EC_filename, header=mainHeaderIdx, encoding='ISO-8859-1', sep="\t", skip_blank_lines=False, decimal=delim)
    df=df.dropna(how='all', axis=1)
    # get acquisition date
    keywordDate='Acquisition started on :'
    if keywordDate in rawEC:
        try:
            acDate=datetime.strptime(next(line for line in linesEC if keywordDate in line).strip(keywordDate).strip(),
                                     "%m/%d/%Y %H:%M:%S.%f")
        except:
            acDate=datetime.strptime(next(line for line in linesEC if keywordDate in line).strip(keywordDate).strip(),
                                     "%m/%d/%Y %H:%M:%S")
    elif acTime!=1:
        try:
            acDate=datetime.strptime(acTime, "%m/%d/%Y %H:%M:%S.%f")
        except:
            acDate=datetime.strptime(acTime, "%m/%d/%Y %H:%M:%S")
    if df['time/s'].dtype=='object':
        df['time/s']=pd.to_datetime(df['time/s'])
        df['time/s']=df['time/s']-df['time/s'][0]
        df['time/s']=df['time/s'].dt.total_seconds()
    try:
        # add new datetime column with format YYYY-MM-DD HH:mm:ss.f
        df['acquisition_datetime']=acDate+pd.TimedeltaIndex(df['time/s'], unit='S')
        # reconvert to string format
        # df['acquisition_datetime']=df['acquisition_datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    except NameError:
        print("There is no information on the start date of the EC acquisition for file "+EC_filename+".")

    # dataframe attributes from header lines
    headerDict={}
    attrs = {}
    newStart = 0
    index = 1
    for i,line in enumerate(linesEC):
        if line.startswith('Ns'):
            setBottom=next(j for j,line2 in enumerate(linesEC[i:]) if len(line2)==1)+i
            setLines=linesEC[i:setBottom]
            if linesEC[i-1].startswith('Modify on '):
                key, value = linesEC[i-1].split(':',1)
                modifyDict={}
                modifyDict[key.strip()]=value.strip()
                if with_settings!=0:
                    df2=ECsettingsdf(setLines,delim)
                    modifyDict = {'Modify '+str(index):modifyDict | {'Settings': df2.to_dict()}}
                    attrs = attrs | modifyDict
                index+=1
            else:
                headerDict=getECHeaderDict(linesEC[newStart:i])
                if with_settings!=0:
                    main_settings_df=ECsettingsdf(setLines,delim)
                    newDict = headerDict | {"Settings": main_settings_df.to_dict()}
                    attrs = attrs | newDict
                newStart=setBottom
    df.attrs = attrs
    if with_settings!=0: # get an additional dataframe for settings only
        return df, main_settings_df
    else:
        return df

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

def ECsettingsdf(setLines,delim):
    '''
    Function to get a pandas dataframe of the EC Lab file settings that written in the header (actually a sub-function used inside getECdf).
    :setLines: Header lines containing the settings data.
    :delim: Type of delimiter used (, or .).
    :return: Returns a pandas dataframe with the settings data.
    '''
    n=len(setLines[0][2:]) - len(setLines[0][2:].lstrip())+2
    setList=[]
    for line in setLines:
        setList.append([line[i:i+n].strip() for i in range(0, len(line), n)])
    set_df=pd.DataFrame(setList).replace('', np.nan).dropna(axis=1, how='all').set_index([0])
    #for c in set_df.columns:
        #try:
            #set_df[c] = pd.to_numeric(set_df[c])
        #except:
            #pass
    # Ns values as column names
    set_df.index.names = [None]
    set_df.columns = set_df.iloc[0]
    set_df = set_df.rename_axis(None, axis=1)
    # convert to numeric type if possible
    set_df = set_df.transpose()
    # WARNING with regex future version of python, correct
    set_df = set_df.apply(lambda x: x.str.replace(delim,'.'))
    for c in set_df.columns:
        try:
            set_df[c] = pd.to_numeric(set_df[c])
        except:
            pass
    #set_df = set_df.transpose()
    return set_df

def ECsettingsdf_transform(set_df):
    # attempt to append 'unit' columns directly to the variable name but sometimes units are not
    # uniform over the entire df (e.g. mA, mA and µA)
    cols = set_df.columns.values
    toDel = []
    toRename = {}
    for i,c in enumerate(cols):
        # check for a regex pattern
        if all(bool(re.match(r"([0-9]+) ([a-zA-Z\u0080-\uFFFF]+)", element)) for element in set_df[c].values):
            # check if units are consistent
            if all(re.match(r"([0-9]+) ([a-zA-Z\u0080-\uFFFF]+)", element).groups()[1] == re.match(r"([0-9]+) ([a-zA-Z\u0080-\uFFFF]+)", set_df[c].values[0]).groups()[1] for element in set_df[c].values):
                toRename[c]=c+' ('+re.match(r"([0-9]+) ([a-zA-Z\u0080-\uFFFF]+)", set_df[c].values[0]).groups()[1]+")"
                set_df[c]=set_df[c].apply(lambda x: re.match(r"([0-9]+) ([a-zA-Z\u0080-\uFFFF]+)", x).groups()[0])
                #print(c+"OK")
        if 'unit '+c in cols:
            if all(element == set_df['unit '+c].values[0] for element in set_df['unit '+c].values):
                toDel.append('unit '+c)
                toRename[c]=c+' ('+set_df['unit '+c].values[0]+")"
        elif 'dunit '+c in cols:
            if all(element == set_df['dunit '+c].values[0] for element in set_df['dunit '+c].values):
                toDel.append('dunit '+c)
                toRename[c]=c+' ('+set_df['dunit '+c].values[0]+")"

    set_df.rename(columns = toRename, inplace = True)
    set_df.drop(columns = toDel, inplace = True)
    return set_df

def getECHeaderDict(text):
    '''
    Function to get a dictionary of the header in an EC Lab mpt file.
    :text: Header text
    :return: Returns a dictionary.
    '''
    headerDict = {}
    newList = []
    for i,line in enumerate(text):
        if not re.match(r'^\s', line) and not re.match(r'^[a-z]', line):
            newList.append(line.strip())
        elif re.match(r'^[a-z]', line) or re.match(r'^\s', line) and line.strip()!='':
            if newList[-1].strip().endswith(':'):
                newList[-1]+=' '+line.strip()+','
            elif newList[-1].strip().endswith(',') or ':' in newList[-1].strip():
                newList[-1]+=' '+line.strip()+','
            else:
                newList[-1]+=' : '+line.strip()+','
    #attention to this line
    newList[0]='File : '+newList[0]
    if len(newList)>2:
        newList[2]='Technique : '+newList[2]
    
    for i,line in enumerate(newList):
        if ':' in line:
            key, value = line.strip(',').split(':',1)
        elif ('Loop' in line) and ('from point number' in line) or ('compliance from' in line):
            frmIdx=line.find('from')
            editLine=line[0:frmIdx]+':'+line[frmIdx:]
            key, value = editLine.split(':',1)
        elif '(software)' in line:
            softIdx=line.find('(software)')
            editLine='Software :'+line[0:softIdx]
            key, value = editLine.split(':',1)
        elif '(firmware)' in line:
            firmIdx=line.find('(firmware)')
            editLine='Firmware :'+line[0:firmIdx]
            key, value = editLine.split(':',1)
        elif ':' not in line:
            editLine=line+': '
            key, value = editLine.split(':',1)
        if key.strip() in headerDict:
            headerDict[key.strip()]+=', '+value.strip()
        else:
            headerDict[key.strip()]=value.strip()
    return headerDict

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

def getRAMANdf(csv_path,filename_col='filename', datetime_col='acquisition_date_time'):
    '''
    Function to get a pandas dataframe with a list of the Raman files in the .csv reference file (files must have a .txt extension).
    
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

def getXASdf_samba_beta(hdf_path,txt_path):
    hdf_df=getXASfilesdf_samba_hdf(hdf_path)
    txt_df=getXASfilesdf_samba_txt(txt_path)
    r = '({})'.format('|'.join(hdf_df['filename']))
    first_merge = txt_df['filename'].str.extract(r, expand=False).fillna(txt_df['filename'])
    #merged_df = hdf_df.merge(txt_df.drop('filename', 1), left_on='filename', right_on=first_merge, how='outer')
    merged_df = txt_df.merge(hdf_df.rename(columns={"filename":'filename hdf'}), left_on=first_merge, right_on='filename hdf', how='outer').drop(columns=['filename hdf'])
    merged_df=merged_df.sort_values(by='average time', ignore_index=True)
    return merged_df

def checkDelim(raw):
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
        #if 'time' in col:
        #    new_data1_cols_dict[col]=col.replace('time','time '+data1_type)
        #elif 'filename' in col:
        #    new_data1_cols_dict[col]=col.replace('filename','filename '+data1_type)
        new_data1_cols_dict[col]=col.replace(col,col+"_"+data1_type)
    new_data2_cols_dict = {}
    for col in data2_df.columns:
        #if 'time' in col:
        #    new_data2_cols_dict[col]=col.replace('time','time '+data2_type)
        #elif 'filename' in col:
        #    new_data2_cols_dict[col]=col.replace('filename','filename '+data2_type)
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

def mergeEC_XASdfs_beta(EC_df,XAS_files_df,which_col_EC='acquisition_datetime',which_col_XAS='stop time',
                   which_col_XAS_start='start time',tol='10 min'):
    '''
    Function to merge electro and XAS-files dataframes based on datetime.
    
    :EC_df: Pandas dataframe with the data from the EC Lab file.
    :XAS_files_df: Pandas dataframe with the data from several XAS files.
    :which_col_EC: Default set to 'acquisition_datetime' column.
    :which_col_XAS: Default set to 'stop time' column (start time + ellapsed time) but you can edit if necessary.
    :return: Returns a merged dataframe with both informations.
    '''
    # merge electro and XAS files dataframes based on approximate datetime
    # check if datetime column exists in EC df
    if which_col_EC not in EC_df.columns:
        raise ValueError("Cannot proceed without start date in EC file.")
    # convert 'acquisition_datetime' column string values in EC df to datetime format
    if pd.api.types.is_datetime64_any_dtype(EC_df[which_col_EC].dtype)==False:
        EC_df[which_col_EC]=pd.to_datetime(EC_df[which_col_EC])
    
    new_ECcols_dict = {}
    for col in EC_df.columns:
        if 'time' in col:
            new_ECcols_dict[col]=col.replace('time','time EC')
        elif 'filename' in col:
            new_ECcols_dict[col]=col.replace('filename','filename EC')
    new_XAScols_dict = {}
    for col in XAS_files_df.columns:
        if 'time' in col:
            new_XAScols_dict[col]=col.replace('time','time XAS')
        elif 'filename' in col:
            new_XAScols_dict[col]=col.replace('filename','filename XAS')
    merged_df = pd.merge_asof(XAS_files_df.rename(columns=new_XAScols_dict),
                              EC_df.rename(columns=new_ECcols_dict),
                              left_on=new_XAScols_dict[which_col_XAS],
                              right_on=new_ECcols_dict[which_col_EC],
                              direction='nearest',
                              tolerance=pd.Timedelta(tol))
    # choose which columns to have in your text file
    cols_XAS = XAS_files_df.rename(columns=new_XAScols_dict).columns.tolist()
    cols_EC = [new_ECcols_dict[which_col_EC],'Ewe/V', '<I>/mA', 'Capacity/mA.h','half cycle','cycle number']
    cols = cols_XAS + cols_EC
    merged_df.dropna(how='all', axis=0, subset=['Ewe/V','<I>/mA'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    elapsed_col=next(col for col in merged_df.columns if 'elapsed' in col)
    merged_df['absolute time/s'] = (merged_df[new_ECcols_dict[which_col_EC]]-merged_df[new_XAScols_dict[which_col_XAS_start]][0]).dt.total_seconds()
    merged_df['absolute time/min'] = merged_df['absolute time/s']/60
    merged_df['absolute time/h'] = merged_df['absolute time/min']/60
    #merged_df['start time XAS'] = merged_df['start time XAS'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    #merged_df['stop time XAS'] = merged_df['stop time XAS'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    if merged_df.shape[0]==0:
        raise ValueError("Could not correctly merge the EC file to a list of XAS files. Please check the folder selection.")
    else:
        for col in merged_df.columns:
            if pd.api.types.is_datetime64_any_dtype(merged_df[col].dtype):
                merged_df[col] = merged_df[col].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
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

# ----------------------------------------------------- plotting pre treatment -----------------------------------------------------
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

# --------------------------------------------------- plotting EC data ---------------------------------------------------
def plotUI_vs_time(df,width=15,height=5):
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

def plotU_vs_capacity(df,nb_cycle='all',width=8,height=5,mass=1000):
    '''
    Function to plot a capacity vs potential graph.
    
    :df: Pandas dataframe with the data from the EC Lab file.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :width: Width of the graph.
    :height: Height of the graph.
    :mass: Mass of active material.
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

def plotdQdU_vs_U(df,nb_cycle='all',reduce_by=1,boxcar=1,savgol=(1,0),colormap='plasma',
                  width=10,height=6,dotsize=10,alpha=1, mass=1000):
    '''
    Function to plot a capacity vs potential graph.
    
    :df: Pandas dataframe with the data from the EC Lab file.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :reduce_by: Factor by which you want to reduce the number of points on your dataframe.
    :boxcar: Factor indicating the size of the moving window of a moving average filter
    :savgol: Tuple (x,y) with the parameters of a Savitzky-Golay filter.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot. Defaul set to 10.
    :alpha: Opacity of the points. Default set to 1.
    :mass: Mass of active material.
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

# --------------------------------------------------- plotting EC XAS data ---------------------------------------------------
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

def plot_all_XAS(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='viridis',pre=20, post=40,width=7,height=4):
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

def plot_2D_XAS_vs_t(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',abstime_col='',
                     colormap='turbo', width=7,height=6,plot_range=None,hlines=False):
    '''
    Function to plot a 2D intensity graph of all the XAS spectra over time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'tab20b'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

def plotEshift_vs_U(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',option=1,
                    colormap='plasma',width=7,height=4,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs potential graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value. Default is the inflection point.
    :intensity_col: Name of the column with the intensity values.
    :option: Parameter to choose the style of the graph. Default set to 1, meaning the figure will have a simplified color code with charge subcycles in red, discharge subcycles in blue, and different markerstyles for each cycle. If set to 2, the figure will have a different color for each cycle (according to a chosen colormap). If set to 3, x axis will be extended (1 sub-figure for each cycle).
    :colormap: Chosen colormap if option set to 2. Default set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot (option 2).
    :alpha: Opacity of the points (option 2).
    :return: Plot.
    '''
    if option==1:
        fig=plotEshift_vs_U_alpha(df,nb_cycle,edge_intensity,intensity_col,width,height,guideline)
    elif option==2:
        fig=plotEshift_vs_U_beta(df,nb_cycle,edge_intensity,intensity_col,colormap,width,height,dotsize,alpha,guideline)
    else:
        fig=plotEshift_vs_U_long(df,nb_cycle,edge_intensity,intensity_col,colormap,guideline=guideline)
    return fig

def plotEshift_vs_U_alpha(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',width=10,height=6,
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

def plotEshift_vs_U_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                         width=7,height=4,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs potential graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

def plotEshift_vs_U_long(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                         colormap='plasma',width=15,height=4,linewidth=1.5,dotsize=18,alpha=0.5,top=0.04,guideline=True):
    '''
    Function to plot edge shift vs potential.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

def plotEshift_vs_x(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',option=1,
                    colormap='plasma',width=7,height=4,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs x graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value. Default is the inflection point.
    :intensity_col: Name of the column with the intensity values.
    :option: Parameter to choose the style of the graph. Default set to 1, meaning the figure will have a simplified color code with charge subcycles in red, discharge subcycles in blue, and different markerstyles for each cycle. If set to 2, the figure will have a different color for each cycle (according to a chosen colormap). If set to 3, x axis will be extended (1 sub-figure for each cycle).
    :colormap: Chosen colormap if option set to 2. Default set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
    :width: Width of the graph.
    :height: Height of the graph.
    :return: Plot.
    '''
    if option==1:
        fig=plotEshift_vs_x_alpha(df,nb_cycle,edge_intensity,intensity_col,width,height,guideline)
    elif option==2:
        fig=plotEshift_vs_x_beta(df,nb_cycle,edge_intensity,intensity_col,colormap,width,height,dotsize,alpha,guideline)
    else:
        fig=plotEshift_vs_x_long(df,nb_cycle,edge_intensity,intensity_col,colormap,guideline=guideline)
    return fig

def plotEshift_vs_x_alpha(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',width=10,height=6,
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

def plotEshift_vs_x_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                         width=10,height=6,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs x graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

def plotEshift_vs_x_long(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                         colormap='plasma',width=15,height=4,linewidth=1.5,dotsize=18,alpha=0.5,top=0.04,guideline=True):
    '''
    Function to plot edge shift vs x.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

def plotx_vs_x(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',x_col='x',option=1,
               colormap='plasma',width=7,height=4,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs x graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value. Default is the inflection point.
    :intensity_col: Name of the column with the intensity values.
    :option: Parameter to choose the style of the graph. Default set to 1, meaning the figure will have a simplified color code with charge subcycles in red, discharge subcycles in blue, and different markerstyles for each cycle. If set to 2, the figure will have a different color for each cycle (according to a chosen colormap). If set to 3, x axis will be extended (1 sub-figure for each cycle).
    :colormap: Chosen colormap if option set to 2. Default set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
    :width: Width of the graph.
    :height: Height of the graph.
    :return: Plot.
    '''
    if option==1:
        fig=plotx_vs_x_alpha(df,nb_cycle,edge_intensity,intensity_col,x_col,width,height,guideline)
    elif option==2:
        fig=plotx_vs_x_beta(df,nb_cycle,edge_intensity,intensity_col,x_col,colormap,width,height,dotsize,alpha,guideline)
    else:
        fig=plotx_vs_x_long(df,nb_cycle,edge_intensity,intensity_col,x_col,colormap,width,height,dotsize,alpha,
                            guideline=guideline)
    return fig

def plotx_vs_x_alpha(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',x_col='x',width=10,height=6,
                     guideline=False):
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

def plotx_vs_x_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',x_col='x',colormap='plasma',
                    width=10,height=6,dotsize=30,alpha=0.8,guideline=False):
    '''
    Function to plot edge shift vs x graph.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :x_col: Name of the column with the x values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

def plotx_vs_x_long(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',x_col='x',
                    colormap='plasma',width=15,height=4,dotsize=18,alpha=0.5,top=0.04,guideline=True,linewidth=1.5):
    '''
    Function to plot edge shift vs x.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

def plotEshift_vs_t(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',width=10,height=4,
                    colormap='plasma',option=1,dotsize=10,alpha=0.5,guideline=False):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value. Default set to inflection point.
    :intensity_col: Name of the column with the intensity values.
    :width: Width of the graph.
    :height: Height of the graph.
    :option: Parameter to choose the style of the graph. Default set to 1, meaning the figure will have a simplified color code with charge subcycles in red, discharge subcycles in blue, and different markerstyles for each cycle. If set to 2, the figure will have a different color for each cycle (according to a chosen colormap).
    :colormap: Chosen colormap if option set to 2. Default set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    if option==1:
        fig=plotEshift_vs_t_alpha(df,nb_cycle,edge_intensity,intensity_col,width,height,guideline)
    else:
        fig=plotEshift_vs_t_beta(df,nb_cycle,edge_intensity,intensity_col,colormap,width,height,dotsize,alpha,guideline)
    return fig
    
def plotEshift_vs_t_alpha(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',width=10,height=6,
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

def plotEshift_vs_t_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                         width=10,height=6,dotsize=10,alpha=0.5,guideline=False):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

def plotdEshiftdt_vs_t(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                       width=10,height=6,dotsize=10,alpha=0.5):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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
def plotEshift_vs_t_stacked(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                            width=10,height=6,dotsize=10,option=1,guideline=False,hspace=.0):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the merged data from the EC Lab file + the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Chosen colormap if option set to 2. Default set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :option: Parameter to choose the style of the graph. Default set to 1, meaning the figure will have a simplified color code with charge subcycles in red, discharge subcycles in blue, and different markerstyles for each cycle. If set to 2, the figure will have a different color for each cycle (according to a chosen colormap).
    :guideline: If True adds a grey line connecting the scattered points.
    :return: Plot.
    '''
    if option==1:
        fig=plotEshift_vs_t_stacked_alpha(df,nb_cycle,edge_intensity,intensity_col,width,height,dotsize,guideline,hspace)
    else:
        fig=plotEshift_vs_t_stacked_beta(df,nb_cycle,edge_intensity,intensity_col,colormap,width,height,dotsize,guideline,
                                         hspace)
    return fig

def plotEshift_vs_t_stacked_alpha(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',width=10,height=6,
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

def plotEshift_vs_t_stacked_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',colormap='plasma',
                                 width=10,height=6,dotsize=10,alpha=0.5,hspace=.0):
    '''
    Function to plot edge shift vs time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'plasma'. You can check additional options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

###### TO CORRECT FROM HERE

def plot3d_XANES_vs_t(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                      colormap_potential='viridis',colormap_cycle='plasma',width=12,height=10,alpha=0.5,
                      plot_range=None):
    '''
    Function to plot a 3D graph of all the XAS spectra over time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap_potential: Name of the colormap you want to use for the plot (according to voltage). Default is set to 'viridis'.
    :colormap_cycle: Name of the colormap you want to use for the arrows pointing the cycles. Default is set to 'plasma'.  Default is set to 'plasma'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
    :width: Width of the graph.
    :height: Height of the graph.
    :alpha: Opacity of the line collections.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
    
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

    # colormap from Ewe/V
    cm=plt.get_cmap(colormap_potential)
    norm = Normalize(vmin=df_copy['Ewe/V'].min(), vmax=df_copy['Ewe/V'].max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

    absorption=[]
    energy=[]
    zs = list(df_copy['absolute time/h'])

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
                zorder=(len(profiles) - i), color=sm.to_rgba(df_copy['Ewe/V'])[i], alpha=0.5)
    

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

    df_copy[df_copy['cycle number']==1].first_valid_index()
    #FIRST ANNOTATION
    cm_cycle=plt.get_cmap(colormap_cycle)
    # color map from the total number of cycles in the DF, not from the lenght of the input nb_cycles
    color_cycle=cm_cycle(np.linspace(0, 1, len(df_copy['cycle number'].unique())))
    for i,cycle in enumerate(df_copy['cycle number'].unique()):
        where=df_copy[df_copy['cycle number']==cycle].first_valid_index()
        maximum=max(df_copy[intensity_col][where])
        index_max=max(range(len(df_copy[intensity_col][where])), key=(df_copy[intensity_col][where]).__getitem__)
        maximum_energy=df_copy[energy_col][where][index_max]
        when=df_copy['absolute time/h'][where]
        x2, y2, _ = proj3d.proj_transform(maximum_energy,when,maximum, ax.get_proj())
        ax.annotate("cycle "+str(int(cycle)), xy = (x2,y2), xytext = (-50, 30), fontsize=12, textcoords = 'offset points', ha = 'center', va = 'bottom', arrowprops = dict(width=0.1,headwidth=7,headlength=8,color=color_cycle[i]))

    def update_position(e):
        for label, x, y, z in labels_and_points:
            x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
            label.xy = x2,y2
            label.update_positions(fig.canvas.renderer)
        fig.canvas.draw()

    fig.canvas.mpl_connect('motion_notify_event', update_position)
    return fig

def plot2d_XANES_vs_t(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                      colormap='tab20b', width=8,height=6,plot_range=None,hlines=False):
    '''
    Function to plot a 2D intensity graph of all the XAS spectra over time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'tab20b'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

def plot2d_XANES_vs_t_beta(df,nb_cycle='all',edge_intensity='inflection',intensity_col='',
                           colormap='tab20b', width=8,height=6,plot_range=None,hlines=False):
    '''
    Function to plot a 2D intensity graph of all the XAS spectra over time.
    
    :df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or number of the cycle you want to plot. Plots all by default.
    :edge_intensity: Intensity value to get the edge energy value.
    :intensity_col: Name of the column with the intensity values.
    :colormap: Name of the colormap you want to use for the plot. Default is set to 'tab20b'. More options here https://matplotlib.org/stable/tutorials/colors/colormaps.html.
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

# ---------------------------------------------------------- MCR ALS ----------------------------------------------------------
#Main Algorithm
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

def plot_scores_3D(df,n_pc=20,intensity_col=''):
    arr=get_intensity_arr(df,intensity_col)
    svd_arr = sp.linalg.svd(arr)
    return
