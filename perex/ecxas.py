import numpy as np, pandas as pd, re, os, pathlib, glob
from datetime import datetime, timedelta
from time import mktime

# scipy signal treatment
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

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

# --------------------------------------------------- extracting data from files ---------------------------------------------------
def getECdf_single(EC_filename, with_settings=0, acTime=1):
    '''
    Function to get a pandas dataframe from an EC Lab file (Must have an .mpt extension).
    
    :EC_filename: Complete path of the file
    :with_settings: Set to 1 if another dataframe with the settings written in the file header is desired.
    :acTime: Usually, the EC Lab file has a line indicating when the acquisition started. If not, you can set it as a string with the format m/d/Y H:M:S.f
    :return: Returns a pandas dataframe with the data from the EC Lab file.
    '''
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
    # MAIN DF
    df=pd.read_csv(EC_filename, header=mainHeaderIdx, encoding='ISO-8859-1', sep="\t", skip_blank_lines=False, decimal=delim)
    df=df.dropna(how='all', axis=1)
    # get acquisition date
    keywordDate='Acquisition started on :'
    if keywordDate in rawEC:
        acDate=datetime.strptime(next(line for line in linesEC if keywordDate in line).strip(keywordDate).strip(), "%m/%d/%Y %H:%M:%S.%f")
    elif acTime!=1:
        acDate=datetime.strptime(acTime, "%m/%d/%Y %H:%M:%S.%f")
    try:
        # add new datetime column with format YYYY-MM-DD HH:mm:ss.f
        df['datetime']=acDate+pd.TimedeltaIndex(df['time/s'], unit='S')
        # reconvert to string format
        # df['datetime']=df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
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
                df2=ECsettingsdf(setLines,delim)
                modifyDict = {'Modify '+str(index):modifyDict | {'Settings': df2.to_dict()}}
                attrs = attrs | modifyDict
                index+=1
            else:
                headerDict=getECHeaderDict(linesEC[newStart:i])
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
    Function to get a pandas dataframe from several EC Lab files (Must have an .mpt extension).
    
    :*args: Complete path(s) of the file(s).
    :with_settings: Set to 1 if another dataframe with the settings written in the file header is desired.
    :acTime: Usually, the EC Lab file has a line indicating when the acquisition started. If not, you can set it as a string with the format m/d/Y H:M:S.f
    :return: Returns a pandas dataframe with the data from EC Lab file(s).
    '''
    if with_settings==0:
        base_df=getECdf_single(args[0])
    else:
        base_df,main_settings_df=getECdf_single(args[0],with_settings)
        main_settings_df=main_settings_df[main_settings_df['Ns'].isin(base_df['Ns'].unique())]
    base_df.attrs={'File 1':base_df.attrs}
    for i,file in enumerate(args[1:]):
        if with_settings==0:
            next_df=getECdf_single(file)
        else:
            next_df,next_settings_df=getECdf_single(file,with_settings)
            next_settings_df=next_settings_df[next_settings_df['Ns'].isin(next_df['Ns'].unique())]
            next_settings_df['Ns']=next_settings_df['Ns']+main_settings_df['Ns'].iloc[-1]+1
            main_settings_df=pd.concat([main_settings_df,next_settings_df],ignore_index=True)
        first_oxred=next_df['ox/red'][0]
        halfcycles_first=next_df.loc[0:next_df[next_df['ox/red']!=first_oxred].index[0]-1,'half cycle']
        cycles_first=next_df.loc[0:next_df[next_df['ox/red']!=first_oxred].index[0]-1,'cycle number']
        if halfcycles_first.unique().size>1:
            #print('half cycles yes')
            next_df.loc[0:next_df[next_df['ox/red']!=first_oxred].index[0]-1,'half cycle']=halfcycles_first.unique()[-1]
        if cycles_first.unique().size>1:
            #print('cycle number yes')
            next_df.loc[0:next_df[next_df['ox/red']!=first_oxred].index[0]-1,'cycle number']=cycles_first.unique()[-1]
        
        #next_df['half cycle']=next_df['half cycle']-next_df['half cycle'].min()+base_df['half cycle'].iloc[-1]
        #next_df['cycle number']=next_df['cycle number']-next_df['cycle number'].min()+base_df['cycle number'].iloc[-1]
        next_df['(Q-Qo)/mA.h']=next_df['(Q-Qo)/mA.h']+base_df['(Q-Qo)/mA.h'].iloc[-1]
        if first_oxred==base_df['ox/red'].iloc[-1]:
            #print('equal first ox red yes')
            next_df['half cycle']=next_df['half cycle']-next_df['half cycle'].min()+base_df['half cycle'].iloc[-1]
            next_df['cycle number']=next_df['cycle number']-next_df['cycle number'].min()+base_df['cycle number'].iloc[-1]
            next_df.loc[next_df[next_df['half cycle']==next_df['half cycle'].min()].index,'Q charge/discharge/mA.h']+=base_df['Q charge/discharge/mA.h'].iloc[-1]
            if first_oxred==1:
                next_df.loc[next_df[next_df['half cycle']==next_df['half cycle'].min()].index,'Q charge/mA.h']+=base_df['Q charge/mA.h'].iloc[-1]
                next_df.loc[next_df[next_df['cycle number']==next_df['cycle number'].min()].index,'Energy charge/W.h']+=base_df['Energy charge/W.h'].iloc[-1]
                #next_df.loc[next_df[next_df['cycle number']==next_df['cycle number'].min()].index,'Capacitance charge/µF']+=base_df['Capacitance charge/µF'].iloc[-1]
            elif first_oxred==0:
                next_df.loc[next_df[next_df['half cycle']==next_df['half cycle'].min()].index,'Q discharge/mA.h']+=base_df['Q discharge/mA.h'].iloc[-1]
                next_df.loc[next_df[next_df['cycle number']==next_df['cycle number'].min()].index,'Energy discharge/W.h']+=base_df['Energy discharge/W.h'].iloc[-1]
                #next_df.loc[next_df[next_df['cycle number']==next_df['cycle number'].min()].index,'Capacitance discharge/µF']+=base_df['Capacitance discharge/µF'].iloc[-1]
        else:
            next_df['half cycle']=next_df['half cycle']-next_df['half cycle'].min()+base_df['half cycle'].iloc[-1]+1
            next_df['cycle number']=next_df['cycle number']-next_df['cycle number'].min()+base_df['cycle number'].iloc[-1]+1
        next_df.loc[next_df['half cycle']==next_df['half cycle'].unique()[0],'Capacity/mA.h']+=base_df['Capacity/mA.h'].iloc[-1]
        next_df['Ns']+=base_df['Ns'].iloc[-1]+1
        deltaX=next_df['x'][0]-base_df['x'].iloc[-1]
        next_df['x']+=-deltaX
        next_df.attrs={'File '+str(i+2):next_df.attrs}
        # next_df['time/s']=base_df['time/s'].iloc[-1]+(datetime.strptime(next_df['datetime'], '%Y-%m-%d %H:%M:%S.%f')-datetime.strptime(base_df['datetime'].iloc[-1], '%Y-%m-%d %H:%M:%S.%f')).total_seconds()
        newAttrs = base_df.attrs | next_df.attrs
        base_df=pd.concat([base_df, next_df], ignore_index=True)
        base_df.attrs=newAttrs
    print('EC dataframe ready.')
    if with_settings!=0: # get an additional dataframe for settings only
        return base_df, main_settings_df
    else:
        return base_df

def ECsettingsdf(setLines,delim):
    '''
    Function to get a pandas dataframe of the EC Lab file settings written in the header (actually a sub-function used inside getECdf).
    
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

def getXASfilesdf(XAS_folder,filters=[],recursive=True):
    '''
    Function to get a pandas dataframe of the XAS files in a folder.
    
    :XAS_folder: Complete path of the folder where the XAS files are.
    :filters: A list of strings containing filter words to select the XAS filenames.
    :recursive: default True if you want to look in the folders recursively.
    :return: Returns a pandas dataframe with the XAS file information.
    '''
    # absolute path to search all text files inside a specific folder
    #path = XAS_folder+'/*.txt'
    #files = glob.glob(path)
    # list to store files name
    files = []
    if type(filters)==str:
        filters=[filters]
    if recursive:
        for (dir_path, dir_names, file_names) in os.walk(XAS_folder):
            files.extend(os.path.join(dir_path, f) for f in file_names if f.endswith(".txt") and all(word in f for word in filters))
    else:
        path = XAS_folder+'/*.txt'
        files = glob.glob(path)
    if len(files)==0:
        raise ValueError("XAS folder empty. Choose another folder.")
    else:
        XAS_dictlist = []
        for f in files:
            with open(f, 'r') as openFile:
                lines = openFile.readlines()
                check=True if '#' in lines[0] else False
            if check:
                XAS_dictlist.append(getSingleXASdict(f))
                #XAS_dictlist.append(getSingleXASdict_simplified(f))
        XAS_df = pd.DataFrame(XAS_dictlist)
        XAS_df=XAS_df.sort_values(by='datetime', ignore_index=True)
        XAS_df['average time XAS']=XAS_df['datetime']+pd.TimedeltaIndex(XAS_df['elapsed time/s'], unit='S')
        XAS_df.drop_duplicates(subset=XAS_df.columns[:3],inplace=True)
        XAS_df=XAS_df.sort_values(by='average time XAS', ignore_index=True)
        cols1=list(XAS_df.columns[:XAS_df.columns.get_loc('elapsed time/s')+1])
        cols2=list(XAS_df.columns[XAS_df.columns.get_loc('elapsed time/s')+2:-1])
        #XAS_df=XAS_df[cols1+['average time XAS']+cols2]
    print('XAS dataframe ready.')
    return XAS_df

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
    startTime=datetime.strptime(next(line for line in linesXAS if '#Time at start=' in line).strip('#Time at start=').strip(), "%Y-%m-%d %H:%M:%S.%f")
    elapsedTime=float(next(line for line in linesXAS if '#Time from start (seconds)=' in line).strip('#Time from start (seconds)=').strip())
    sampleT=float(next(line for line in linesXAS if '#Sample temperature (C)=' in line).strip('#Sample temperature (C)=').strip())
    varNames=['filename','datetime','elapsed time/s','sample T (C)']
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
    startTime=datetime.strptime(next(line for line in linesXAS if '#Time at start=' in line).strip('#Time at start=').strip(), "%Y-%m-%d %H:%M:%S.%f")
    elapsedTime=float(next(line for line in linesXAS if '#Time from start (seconds)=' in line).strip('#Time from start (seconds)=').strip())
    sampleT=float(next(line for line in linesXAS if '#Sample temperature (C)=' in line).strip('#Sample temperature (C)=').strip())
    varNames=['filename','datetime','elapsed time/s','sample T (C)']
    varVals=[os.path.basename(filename),startTime,elapsedTime,sampleT]
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
        # adding edge values
        #E_fine = np.linspace(df['shifted energy'].min(),df['shifted energy'].max(), 100000)
        #interpol = interp1d(df['shifted energy'], df['normalized'])
        #mu_norm_int=interpol(E_fine)
        #deriv=pd.Series(mu_norm_int).diff()/pd.Series(E_fine).diff()
        #edge_0p5=pd.Series(E_fine)[pd.Series(mu_norm_int).sub(0.5).abs().idxmin()]
        #edge_0p7=pd.Series(E_fine)[pd.Series(mu_norm_int).sub(0.7).abs().idxmin()]
        #edge_max_inflection=pd.Series(E_fine)[pd.Series(deriv).idxmax()]
        #varNames=varNames+['edge_0.5','edge_0.7','edge_max_inflection']
        #varVals=varVals+[edge_0p5,edge_0p7,edge_max_inflection]
        #df=df.drop(columns='dI/dE')
    else:
        headerIdx=next(i for i,line in enumerate(linesXAS) if '#Energy' in line)
        df=pd.read_csv(filename, header=headerIdx, sep="\t", skip_blank_lines=False, decimal=delim)
        df.rename(columns={"#Energy":'Energy'},inplace=True)
        if df['Time'].isna().sum()==df.shape[0]:
            df=df.drop(columns='Time')
            df.rename(columns={"iFluo1": "Time"},inplace=True)
        # adding edge values
        #E_fine = np.linspace(df['Energy'].min(),df['Energy'].max(), 100000)
        #interpol = interp1d(df['Energy'], df['mux'])
        #mux_int=interpol(E_fine)
        #deriv=pd.Series(mux_int).diff()/pd.Series(E_fine).diff()
        #edge_max_inflection=pd.Series(E_fine)[deriv.idxmax()]
        #varNames=varNames+['edge_max_inflection']
        #varVals=varVals+[edge_max_inflection]
        #df=df.drop(columns='dmux/dE')
    df=df.astype(float)
    # get number of header lines
    spectrumDataDict = df.to_dict('list')
    headerDataDict = {varNames[i]: varVals[i] for i in range(len(varNames))}
    outputDict=headerDataDict|spectrumDataDict
    return outputDict

def mergeEC_XASdfs(EC_df,XAS_files_df):
    '''
    Function to merge electro and XAS files dataframes based on approximate datetime.
    
    :EC_df: Pandas dataframe with the data from the EC Lab file.
    :XAS_files_df: Pandas dataframe with the data from several XAS files.
    :return: Returns a merged dataframe with both informations.
    '''
    # merge electro and XAS files dataframes based on approximate datetime
    # check if datetime column exists in EC df
    if 'datetime' not in EC_df.columns:
        raise ValueError("Cannot proceed without start date in EC file.")
    # convert 'datetime' column string values in EC df to datetime format
    EC_df['datetime']=pd.to_datetime(EC_df['datetime'])
    merged_df = pd.merge_asof(XAS_files_df.rename(columns={'datetime':'starttime XAS','elapsed time/s':'elapsed time XAS/s'}),
                              EC_df.rename(columns={'datetime':'datetime EC'}),
                              left_on='average time XAS',
                              right_on='datetime EC',
                              direction='nearest',
                              tolerance=pd.Timedelta('10 min'))
    # choose which columns to have in your text file
    cols_XAS = list(XAS_files_df.rename(columns={'datetime':'starttime XAS','elapsed time/s':'elapsed time XAS/s'}).columns)
    cols_EC = ['datetime EC','Ewe/V', '<I>/mA', 'x','Capacity/mA.h','half cycle','cycle number']
    cols = cols_XAS + cols_EC
    #cols = ['filename','starttime XAS','elapsed time XAS/s','average time XAS','sample T (C)','datetime EC','Ewe/V', '<I>/mA', 'x','Capacity/mA.h','half cycle','cycle number']
    merged_df = merged_df[cols]
    merged_df.dropna(how='all', axis=0, subset=['Ewe/V','<I>/mA','x'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df['absolute time/s'] = (merged_df['datetime EC']-merged_df['datetime EC'][0]).dt.total_seconds()+merged_df['elapsed time XAS/s']
    merged_df['absolute time/min'] = merged_df['absolute time/s']/60
    merged_df['absolute time/h'] = merged_df['absolute time/min']/60
    merged_df['starttime XAS'] = merged_df['starttime XAS'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    merged_df['average time XAS'] = merged_df['average time XAS'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    if merged_df.shape[0]==0:
        raise ValueError("Could not correctly merge the EC file to a list of XAS files. Please check the folder selection.")
    else:
        merged_df['datetime EC'] = merged_df['datetime EC'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    print('Merged dataframe ready.')
    return merged_df


def simplified_output(EC_filename,XAS_folder,output_name,EC_starttime=1):
    electrodf=getECdf(EC_filename,acTime=EC_starttime)
    print('EC data obtained.')
    XASdf=getXASfilesdf(XAS_folder)
    XASdf=XASdf[['filename','datetime','elapsed time/s','sample T (C)','average time XAS']]
    print('XAS filenames obtained.')
    merged=mergeEC_XASdfs(electrodf,XASdf)
    merged.to_csv(output_name, index=False, sep='\t')
    print('Output file succesfully created!')
    return

# ---------------------------------------------------------- plotting ----------------------------------------------------------

########## EC data alone ##########
def plotUI_vs_time(electrodf,width=15,height=5):
    '''
    Function to plot a potential/current vs time graph.
    
    :electrodf: Pandas dataframe with the data from the EC Lab file.
    :width: Width of the graph.
    :height: Height of the graph.
    :return: Plot.
    '''
    fig, ax1 = plt.subplots(figsize=(width,height))
    # add an additional y axis for the current
    ax2 = ax1.twinx()
    # get time series
    if 'datetime' in electrodf.columns:
        timeSeries=(electrodf['datetime']-electrodf['datetime'][0]).dt.total_seconds()
    elif electrodf[electrodf['time/s']==0].shape[0]>1:
        timeSeries=list(electrodf['time/s'][:electrodf[electrodf['time/s']==0].index[1]])
        for i in electrodf[electrodf['time/s']==0].index[1:-1]:
            timeSeries.extend(list(electrodf['time/s'][i:i+1]+electrodf['time/s'][i-1]))
    else:
        timeSeries=electrodf['time/s']
    # plot V vs. time
    ax1.plot(timeSeries/3600,electrodf['Ewe/V'],color="blue")
    # plot I vs. time
    ax2.plot(timeSeries/3600,electrodf['<I>/mA'],color="green")
    # set the labels and the colors of the curves
    ax1.set_xlabel("Time (h)",fontsize=14)
    ax1.set_ylabel("Potential vs. Li/Li$^+$ (V)",color="blue",fontsize=14)
    ax2.set_ylabel("Current (mA)",color="green",fontsize=14)
    # let's also color the axes
    ax1.tick_params(axis='y',colors="blue")
    ax2.tick_params(axis='y',colors="green")
    ax2.spines['right'].set_color("green")
    # axes ticks inwards
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax2.tick_params(axis='both', labelsize=13, direction='in')
    return fig

def plotCapacity_vs_U(electrodf,nb_cycle=None,width=8,height=5,mass=1000):
    '''
    Function to plot a capacity vs potential graph.
    
    :electrodf: Pandas dataframe with the data from the EC Lab file.
    :nb_cycle: List of the cycles you want to plot. Or integer with the cycle you want to plot.
    :width: Width of the graph.
    :height: Height of the graph.
    :mass: Mass of active material.
    :return: Plot.
    '''
    mass=mass/1000 # mg to g
    total_cycles=len(electrodf['cycle number'].unique())
    
    try:
        nb_cycle=int(nb_cycle)
        if nb_cycle not in electrodf['cycle number'].unique():
            raise ValueError("Not a valid nb_cycle.")
        else:
            nb_cycle=[nb_cycle]
    except ValueError: raise ValueError('Not a valid nb_cycle.')
    except: pass
        
    if not nb_cycle:
        nb_cycle=electrodf['cycle number'].unique()
    elif not all(elem in electrodf['cycle number'].unique() for elem in nb_cycle):
        raise ValueError('The chosen cycle numbers are not in the sequence.')

    color_charge = cm.seismic(np.linspace(0, 0.5, total_cycles+2))[1:-1]
    color_discharge = cm.seismic(np.linspace(1, 0.5, total_cycles+2))[1:-1]
    # build figure
    fig, ax = plt.subplots(figsize=(width,height))
    for half_cycle in electrodf['half cycle'].unique():
        subdf=electrodf[electrodf['half cycle']==half_cycle]
        cycle_number=int(subdf['cycle number'].mean())
        if cycle_number in nb_cycle:
            if (half_cycle % 2) == 0:
                #ax.scatter(subdf['Capacity/mA.h']/mass,subdf['Ewe/V'],s=0.2,color=color_charge[int(np.floor(i/2))])
                ax.scatter(subdf['Capacity/mA.h']/mass,subdf['Ewe/V'],s=0.2,color=color_charge[int(np.where(electrodf['cycle number'].unique()==cycle_number)[0])])
                ax.annotate(str(int(cycle_number)), (subdf['Capacity/mA.h'].iloc[-1]/mass, subdf['Ewe/V'].iloc[-1]+0.01), color=color_charge[int(np.where(electrodf['cycle number'].unique()==cycle_number)[0])], va='bottom')
            else:
                #ax.scatter(subdf['Capacity/mA.h']/mass,subdf['Ewe/V'],s=0.2,color=color_discharge[int(np.floor(i/2))])
                ax.scatter(subdf['Capacity/mA.h']/mass,subdf['Ewe/V'],s=0.2,color=color_discharge[int(np.where(electrodf['cycle number'].unique()==cycle_number)[0])])
                ax.annotate(str(int(cycle_number)), (subdf['Capacity/mA.h'].iloc[-1]/mass, subdf['Ewe/V'].iloc[-1]-0.01), color=color_discharge[int(np.where(electrodf['cycle number'].unique()==cycle_number)[0])], va='top')
    
    # set axes labels and limits
    ax.set_ylabel("Potential vs. Li/Li$^+$ (V)",fontsize=14)
    if mass==1:
        ax.set_xlabel("Capacity (mAh)",fontsize=14)
    else:
        ax.set_xlabel("Capacity (mAh$\cdot$g$^{-1}$)",fontsize=14)
    margin=0.05
    xmin=electrodf['Capacity/mA.h'].min()-(electrodf['Capacity/mA.h'].max()-electrodf['Capacity/mA.h'].min())*margin/(1-margin*2)
    xmax=electrodf['Capacity/mA.h'].max()+(electrodf['Capacity/mA.h'].max()-electrodf['Capacity/mA.h'].min())*margin/(1-margin*2)
    ymin=electrodf['Ewe/V'].min()-(electrodf['Ewe/V'].max()-electrodf['Ewe/V'].min())*margin/(1-margin*2)
    ymax=electrodf['Ewe/V'].max()+(electrodf['Ewe/V'].max()-electrodf['Ewe/V'].min())*margin/(1-margin*2)
    ax.set_xlim(xmin/mass,xmax/mass)
    ax.set_ylim(ymin,ymax)
    ax.tick_params(axis='both', labelsize=13, direction='in')
    return fig

def plotdQdU_vs_U(electrodf,nb_cycle=None,reduce_by=1,boxcar=1,savgol=(1,0),colormap='plasma',width=10,height=6,dotsize=10,alpha=1, mass=1000):
    '''
    Function to plot a capacity vs potential graph.
    
    :electrodf: Pandas dataframe with the data from the EC Lab file.
    :nb_cycle: List of the cycles you want to plot. Or integer with the cycle you want to plot.
    :reduce_by: Factor by which you want to reduce the number of points on your dataframe.
    :boxcar: Factor indicating the size of the moving window of a moving average filter
    :savgol: Tuple (x,y) with the parameters of the Savitzky-Golay filter that you want to apply.
    :colormap: Name of the colormap you want to use for the plot.
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Transparency of the points.
    :mass: Mass of active material.
    :return: Plot.
    '''
    mass=mass/1000 # mg to g
    newECdf=electrodf.copy(deep=True)
    total_cycles=len(electrodf['cycle number'].unique())
    
    try:
        nb_cycle=int(nb_cycle)
        if nb_cycle not in electrodf['cycle number'].unique():
            raise ValueError("Not a valid nb_cycle.")
        else:
            nb_cycle=[nb_cycle]
    except ValueError: raise ValueError('Not a valid nb_cycle.')
    except: pass
        
    if not nb_cycle:
        nb_cycle=electrodf['cycle number'].unique()
    elif not all(elem in electrodf['cycle number'].unique() for elem in nb_cycle):
        raise ValueError('The chosen cycle numbers are not in the sequence.')

    newECdf['dQdV']=(newECdf['Capacity/mA.h']/mass).diff()/newECdf['Ewe/V'].diff()
    # apply filters/smoothing
    newECdf=newECdf.iloc[::reduce_by]
    newECdf['dQdV']=newECdf['dQdV'].rolling(boxcar).mean()
    newECdf['dQdV']=savgol_filter(newECdf['dQdV'],savgol[0],savgol[1])

    cm=plt.get_cmap(colormap)
    # color map from the total number of cycles in the DF, not from the length of the input nb_cycles
    color_dqdv=cm(np.linspace(0, 1, len(newECdf['cycle number'].unique())))
    
    # build figure
    fig, ax = plt.subplots(figsize=(width,height))

    for i, cycle in enumerate(newECdf['cycle number'].unique()):
        if cycle in nb_cycle:
            subdf=newECdf[newECdf['cycle number']==cycle]
            ax.scatter(subdf['Ewe/V'],subdf['dQdV'],s=dotsize,color=color_dqdv[i],label='cycle '+str(int(cycle)),alpha=alpha)

    # set axes labels and limits
    ax.tick_params(axis='both', labelsize=13, direction='in')
    # x
    margin=0.05
    xmin=electrodf['Ewe/V'].min()-(electrodf['Ewe/V'].max()-electrodf['Ewe/V'].min())*margin/(1-margin*2)
    xmax=electrodf['Ewe/V'].max()+(electrodf['Ewe/V'].max()-electrodf['Ewe/V'].min())*margin/(1-margin*2)
    ax.set_xlim(xmin,xmax)
    ax.set_xlabel("Potential vs. Li/Li$^+$ (V)",fontsize=14)
    # y
    ylim=(abs(newECdf['dQdV'].quantile(0.95))+abs(newECdf['dQdV'].quantile(0.05)))/2
    ax.set_ylim(-ylim,ylim)
    if mass==1:
        ax.set_ylabel("dQ/dV (mAh$\cdot$V$^{-1}$)",fontsize=14)
    else:
        ax.set_ylabel("dQ/dV (mAh$\cdot$g$^{-1}\cdot$V$^{-1}$)",fontsize=14)

    # put colorbar legend
    norm = Normalize(vmin=int(electrodf['cycle number'].min()), vmax=int(electrodf['cycle number'].max()))
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    if len(nb_cycle)>5:
        cbar = fig.colorbar(sm)
        cbar.set_label('Cycle', rotation=270, labelpad=10, fontsize=14)
    else:
        leg = ax.legend(loc='upper left',prop={'size': 14},markerscale=2)
    return fig


########## EC data + XAS data ##########

def get_edge(df,intensity='inflection'):
    '''
    Function to get the interpolated energy value at a certain intesity.
    
    :df: Pandas dataframe with the XAS spectra data.
    :intensity: Intensity value to get the edge energy value.
    :return: List of energy values.
    '''
    edge_energy=[]
    for index, row in df.iterrows():
        E_fine = np.linspace(min(row['shifted energy']),max(row['shifted energy']), 100000)
        interpol = interp1d(row['shifted energy'], row['normalized'])
        mu_norm_int=interpol(E_fine)
        deriv=pd.Series(mu_norm_int).diff()/pd.Series(E_fine).diff()
        if type(intensity)==float or type(intensity)==int:
            edge_energy.append(pd.Series(E_fine)[pd.Series(mu_norm_int).sub(intensity).abs().idxmin()])
        elif intensity=='inflection':
            edge_energy.append(pd.Series(E_fine)[pd.Series(deriv).idxmax()])
        else:
            raise ValueError('Not a valid intensity value.')
        edge_df=pd.Series(edge_energy)
    return edge_df


def plotEshift_vs_U(merged_df,nb_cycle=None,edge_intensity='inflection',colormap='plasma',width=10,height=6,dotsize=10,alpha=0.5):
    '''
    Function to plot edge shift vs potential graph.
    
    :merged_df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or integer with the cycle you want to plot.
    :edge_intensity: Intensity value to get the edge energy value.
    :colormap: Name of the colormap you want to use for the plot.
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Transparency of the points.
    :return: Plot.
    '''
    edge=get_edge(merged_df,intensity=edge_intensity)
    
    try:
        nb_cycle=int(nb_cycle)
        if nb_cycle not in merged_df['cycle number'].unique():
            raise ValueError("Not a valid nb_cycle.")
        else:
            nb_cycle=[nb_cycle]
    except ValueError: raise ValueError('Not a valid nb_cycle.')
    except: pass
        
    if not nb_cycle:
        nb_cycle=merged_df['cycle number'].unique()
    elif not all(elem in merged_df['cycle number'].unique() for elem in nb_cycle):
        raise ValueError('The chosen cycle numbers are not in the sequence.')

    #cm=plt.get_cmap(colormap)
    # color map from the total number of cycles in the DF, not from the length of the input nb_cycles
    #color_dqdv=cm(np.linspace(0, 1, len(merged_df['cycle number'].unique())))
    
    # normalize the colormap with respect to the total number of cycles in the df
    norm = Normalize(vmin=int(merged_df['cycle number'].min()), vmax=int(merged_df['cycle number'].max()))
    # build plot
    fig, ax = plt.subplots(figsize=(width,height))

    #for i, cycle in enumerate(merged_df['cycle number'].unique()):
        #if cycle in nb_cycle:
            #subdf=merged_df[merged_df['cycle number']==cycle]
            #ax.scatter(subdf['Ewe/V'],subdf[Eshift_col_name],s=dotsize,color=color_dqdv[i],label='cycle '+str(int(cycle)),alpha=alpha)
    
    condition = merged_df['cycle number'].isin(nb_cycle)
    # plot Edge shift
    scatter = ax.scatter(merged_df['Ewe/V'][condition],edge[condition],s=dotsize,c=merged_df['cycle number'][condition], cmap=colormap, norm=norm,alpha=alpha)
    
    # axes labels and limits
    margin=0.05
    ax.tick_params(axis='both', labelsize=13, direction='in')
    # x
    xmin=merged_df['Ewe/V'].min()-(merged_df['Ewe/V'].max()-merged_df['Ewe/V'].min())*margin/(1-margin*2)
    xmax=merged_df['Ewe/V'].max()+(merged_df['Ewe/V'].max()-merged_df['Ewe/V'].min())*margin/(1-margin*2)
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


def plotEshift_vs_t(merged_df,nb_cycle=None,edge_intensity='inflection',colormap='plasma',width=10,height=6,dotsize=10,alpha=0.5):
    '''
    Function to plot edge shift vs time.
    
    :merged_df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or integer with the cycle you want to plot.
    :edge_intensity: Intensity value to get the edge energy value.
    :colormap: Name of the colormap you want to use for the plot.
    :width: Width of the graph.
    :height: Height of the graph.
    :dotsize: Size of the dot of the scatter plot.
    :alpha: Transparency of the points.
    :return: Plot.
    '''
    total_cycles=len(merged_df['cycle number'].unique())
    
    edge=get_edge(merged_df,intensity=edge_intensity)
    
    try:
        nb_cycle=int(nb_cycle)
        if nb_cycle not in merged_df['cycle number'].unique():
            raise ValueError("Not a valid nb_cycle.")
        else:
            nb_cycle=[nb_cycle]
    except ValueError: raise ValueError('Not a valid nb_cycle.')
    except: pass
        
    if not nb_cycle:
        nb_cycle=merged_df['cycle number'].unique()

    #colormap according to total number of cycles
    #cm=plt.get_cmap(colormap)
    #color_dqdv=cm(np.linspace(0, 1, len(merged_df['cycle number'].unique())))
    
    # normalize the colormap with respect to the total number of cycles in the df
    norm = Normalize(vmin=int(merged_df['cycle number'].min()), vmax=int(merged_df['cycle number'].max()))
    
    # build figure
    fig, ax = plt.subplots(figsize=(width,height))
    condition = merged_df['cycle number'].isin(nb_cycle)
    # plot Edge shift
    scatter = ax.scatter(merged_df['absolute time/h'][condition], edge[condition], s=dotsize, c=merged_df['cycle number'][condition], cmap=colormap, norm=norm)
    # plot potential over time
    ax1 = ax.twinx()
    ax1.scatter(merged_df['absolute time/h'][condition],merged_df['Ewe/V'][condition], color='black',s=2)
    # axes parameters
    ax.tick_params(axis='both', labelsize=13, direction='in')
    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax.minorticks_on()
    ax.tick_params(which="minor", axis="x", direction="in")
    ax.tick_params(which="minor", axis="y", color='w')
    # x
    ax.set_xlabel("Time (h)",fontsize=14,labelpad=10)
    margin = 0.05
    full_range_x = merged_df['absolute time/h'].max()-merged_df['absolute time/h'].min()
    x_min=merged_df['absolute time/h'].min()-(full_range_x)*margin/(1-margin*2)
    x_max=merged_df['absolute time/h'].max()+(full_range_x)*margin/(1-margin*2)
    ax.set_xlim(x_min,x_max)
    # y left
    label_y_right='Edge @ J='+str(edge_intensity)+' (eV)'
    ax.set_ylabel(label_y_right,fontsize=14,labelpad=10)
    # y right
    ax1.set_ylabel("Potential vs. Li/Li$^+$ (V)",fontsize=14,labelpad=10)
    full_range_y_right = merged_df['Ewe/V'].max()-merged_df['Ewe/V'].min()
    y_right_min=merged_df['Ewe/V'].min()-(full_range_y_right)*margin/(1-margin*2)
    y_right_max=merged_df['Ewe/V'].max()+(full_range_y_right)*margin/(1-margin*2)
    #ax1.set_ylim(y_right_min,y_right_max)
    ax1.set_ylim(2.8,5) # check later
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


#### FOR THE MOMENT THESE ONLY WORK FOR DATAFRAMES WITH NORMALISED XAS FILES

def plotAbsorptionDiff_beta(merged_df,nb_cycle=None,edge_intensity='inflection',colormap='viridis',width=12,height=6,plot_range=None):
    '''
    Function to plot the differential absorption intensity vs energy.
    
    :merged_df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or integer with the cycle you want to plot.
    :edge_intensity: Intensity value to get the edge energy value.
    :colormap: Name of the colormap you want to use for the plot.
    :width: Width of the graph.
    :height: Height of the graph.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
    # if no cycle number is selected then it just plots the first one
    # think about a grid space where all of them can be plotted! future work
    # rate of change between one cycle and another
    # third derivative??
    
        
    total_cycles=len(merged_df['cycle number'].unique())
    edge=get_edge(merged_df,intensity=edge_intensity)
        
    if not nb_cycle:
        nb_cycle=merged_df['cycle number'].min()
        
    try:
        nb_cycle=int(nb_cycle)
        if nb_cycle not in merged_df['cycle number'].unique():
            raise ValueError("Not a valid nb_cycle.")
        else:
            nb_cycle=[nb_cycle]
    except:
        raise ValueError('Not a valid nb_cycle.')
    
    if not all(elem in merged_df['cycle number'].unique() for elem in nb_cycle):
        raise ValueError('The chosen cycle numbers are not in the sequence.')
        
    if plot_range:
        if ((len(plot_range)==2) & (type(plot_range)==list)) & (all([isinstance(item, (int,float)) for item in plot_range])):
            rang=plot_range
        else:
            raise ValueError("Not a valid plot_range.")
    else:
        rang=[edge.mean()-20,edge.mean()+40]
    # colors
    cm=plt.get_cmap(colormap)
    norm = Normalize(vmin=merged_df['Ewe/V'].min(), vmax=merged_df['Ewe/V'].max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    colors = sm.to_rgba(merged_df['Ewe/V'])
    #  build plot
    fig = plt.figure(figsize=(width,height))
    gs = gridspec.GridSpec(1, 2) 

    # the first subplot
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    condition = merged_df['cycle number']==nb_cycle
    charge,discharge=merged_df['half cycle'][condition].unique()
    #char,dess=output_df['half cycle'][output_df['cycle number']==1].unique()
    for i in merged_df[merged_df['half cycle']==charge].index:
        ax0.plot(pd.Series(merged_df['shifted energy'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],pd.Series(merged_df['normalized'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])]-pd.Series(merged_df['normalized'][0])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],color=colors[i])
    for i in merged_df[merged_df['half cycle']==discharge].index:
        ax1.plot(pd.Series(merged_df['shifted energy'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],pd.Series(merged_df['normalized'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])]-pd.Series(merged_df['normalized'][0])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],color=colors[i])

    xticks = ax0.xaxis.get_major_ticks()

    ax0.text(0.02, 0.95,'charge', fontsize=13,transform=ax0.transAxes)
    ax1.text(0.02, 0.95,'discharge', fontsize=13,transform=ax1.transAxes)


    plt.subplots_adjust(wspace=.05)
    ax1.tick_params(axis='both', labelsize=13)

    # x and y labels
    ax1.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
    ax0.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
    ax0.tick_params(axis='both', labelsize=13, direction='in')
    ax1.set_yticks([])

    ax0.set_ylabel('Absorption difference (pristine as ref)',fontsize=14,labelpad=10)


    ax0.minorticks_on()
    ax0.tick_params(which='both', labelsize=13, direction='in')

    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax1.minorticks_on()
    ax1.tick_params(which="minor", axis="x", direction="in")
    ax1.tick_params(which="minor", axis="y", color='w')



    cbar_ax = fig.add_axes([0.92, 0.125, 0.025, 0.755])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Potential vs. Li/Li$^+$ (V)',fontsize=14, rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=13)

    fig.suptitle('Cycle '+str(nb_cycle), fontsize=16)
    return fig

def plotAbsorptionDiff(merged_df,nb_cycle=None,edge_intensity='inflection',colormap='viridis',width=12,height=5,plot_range=None):
    '''
    Function to plot the differential absorption intensity vs energy.
    
    :merged_df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or integer with the cycle you want to plot.
    :edge_intensity: Intensity value to get the edge energy value.
    :colormap: Name of the colormap you want to use for the plot.
    :width: Width of the graph.
    :height: Height of the graph.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
    # if no cycle number is selected then it just plots the first one
    total_cycles=len(merged_df['cycle number'].unique())

    edge=get_edge(merged_df,intensity=edge_intensity)
    try:
        if type(nb_cycle)==list:
            if not all(elem in merged_df['cycle number'].unique() for elem in nb_cycle):
                raise ValueError('The chosen cycle numbers are not in the sequence.')
        elif not nb_cycle:
            nb_cycle=merged_df['cycle number'].min()
        else:
            try:
                nb_cycle=int(nb_cycle)
                if nb_cycle not in merged_df['cycle number'].unique():
                    raise ValueError("Not a valid nb_cycle.")
                else:
                    nb_cycle=[nb_cycle]
            except:
                pass
    except:
        raise ValueError("Not a valid nb_cycle.")
        
    
    if not all(elem in merged_df['cycle number'].unique() for elem in nb_cycle):
        raise ValueError('The chosen cycle numbers are not in the sequence.')
    height=height*len(nb_cycle)
    if plot_range:
        if ((len(plot_range)==2) & (type(plot_range)==list)) & (all([isinstance(item, (int,float)) for item in plot_range])):
            rang=plot_range
        else:
            raise ValueError("Not a valid plot_range.")
    else:
        rang=[edge.mean()-20,edge.mean()+40]
    # colors
    cm=plt.get_cmap(colormap)
    norm = Normalize(vmin=merged_df['Ewe/V'].min(), vmax=merged_df['Ewe/V'].max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    colors = sm.to_rgba(merged_df['Ewe/V'])
    #  build plot
    fig = plt.figure(figsize=(width,height))
    gs = gridspec.GridSpec(1*len(nb_cycle), 2) 

    for j,cycle in enumerate(nb_cycle):
        # the first subplot
        ax0 = plt.subplot(gs[j,0])
        ax1 = plt.subplot(gs[j,1])

        condition = merged_df['cycle number']==cycle
        charge,discharge=merged_df['half cycle'][condition].unique()

        for i in merged_df[merged_df['half cycle']==charge].index:
            plt.subplot(gs[j,0]).plot(pd.Series(merged_df['shifted energy'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],pd.Series(merged_df['normalized'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])]-pd.Series(merged_df['normalized'][0])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],color=colors[i])
        for i in merged_df[merged_df['half cycle']==discharge].index:
            plt.subplot(gs[j,1]).plot(pd.Series(merged_df['shifted energy'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],pd.Series(merged_df['normalized'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])]-pd.Series(merged_df['normalized'][0])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],color=colors[i])

        xticks = ax0.xaxis.get_major_ticks()
        ax0.text(0.02, 0.9,'Cycle '+str(cycle), fontsize=13,transform=ax0.transAxes)
        if cycle==nb_cycle[-1]:
            ax0.text(0.8, 0.1,'charge', fontsize=13,transform=ax0.transAxes)
            ax1.text(0.8, 0.1,'discharge', fontsize=13,transform=ax1.transAxes)
        else:
            ax0.set_xticks([])
            ax1.set_xticks([])


        plt.subplots_adjust(wspace=.05) # no vertical space between plots
        plt.subplots_adjust(hspace=.0) # no horizontal space between plots
        ax1.tick_params(axis='both', labelsize=13)

        # labels and ticks

        ax0.minorticks_on()
        ax0.tick_params(which='both', axis='both',labelsize=13, direction='in')
        ax0.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
        # remove last tick label and first tick label for the necessary subplots
        yticks = ax0.yaxis.get_major_ticks()
        if i!=0:
            yticks[-1].label1.set_visible(False)
        if cycle!=nb_cycle[-1] and i>1:
            yticks[0].label1.set_visible(False)
        
        ax1.set_yticks([])
        ax1.minorticks_on()
        ax1.tick_params(axis='both', labelsize=13, direction='in')
        ax1.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
        ax1.tick_params(which="minor", axis="x", direction="in")
        ax1.tick_params(which="minor", axis="y", color='w')

    #shrink=0.8
    cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.45])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Potential vs. Li/Li$^+$ (V)',fontsize=14, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=13)

    fig.supylabel('Absorption difference (pristine as ref)', fontsize=16)
    return fig

def plotAbsorptionDerivative_beta(merged_df,nb_cycle=0,edge_intensity='inflection',deriv=1,colormap='viridis',width=12,height=6,plot_range=None):
    '''
    Function to plot the derivated absorption intensity vs energy.
    
    :merged_df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or integer with the cycle you want to plot.
    :edge_intensity: Intensity value to get the edge energy value.
    :deriv: Order of derivative to plot.
    :colormap: Name of the colormap you want to use for the plot.
    :width: Width of the graph.
    :height: Height of the graph.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
    # if no cycle number is selected then it just plots the first one
    # think about a grid space where all of them can be plotted! future work
    # rate of change between one cycle and another
    # third derivative??
    # try to also plot the edge as scatter --- how?
        
    total_cycles=len(merged_df['cycle number'].unique())
    edge=get_edge(merged_df,intensity=edge_intensity)
    try:
        nb_cycle=int(nb_cycle)
        if nb_cycle not in merged_df['cycle number'].unique():
            raise ValueError("Not a valid nb_cycle.")
    except:
        raise ValueError('Not a valid nb_cycle.')
    
    if plot_range:
        if ((len(plot_range)==2) & (type(plot_range)==list)) & (all([isinstance(item, (int,float)) for item in plot_range])):
            rang=plot_range
        else:
            raise ValueError("Not a valid plot_range.")
    else:
        rang=[edge.mean()-20,edge.mean()+40]
    # colors
    cm=plt.get_cmap(colormap)
    norm = Normalize(vmin=merged_df['Ewe/V'].min(), vmax=merged_df['Ewe/V'].max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    colors = sm.to_rgba(merged_df['Ewe/V'])
    #  build plot
    fig = plt.figure(figsize=(width,height))
    gs = gridspec.GridSpec(1, 2) 

    # the first subplot
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    condition = merged_df['cycle number']==nb_cycle
    charge,discharge=merged_df['half cycle'][condition].unique()
    #char,dess=output_df['half cycle'][output_df['cycle number']==1].unique()
    for i in merged_df[merged_df['half cycle']==charge].index:
        derivated_series=pd.Series(merged_df['normalized'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])]
        for j in range(deriv):
            derivated_series=derivated_series.diff()
        ax0.plot(pd.Series(merged_df['shifted energy'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],derivated_series,color=colors[i])
    for i in merged_df[merged_df['half cycle']==discharge].index:
        derivated_series=pd.Series(merged_df['normalized'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])]
        for j in range(deriv):
            derivated_series=derivated_series.diff()
        ax1.plot(pd.Series(merged_df['shifted energy'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],derivated_series,color=colors[i])

    xticks = ax0.xaxis.get_major_ticks()

    ax0.text(0.02, 0.95,'charge', fontsize=13,transform=ax0.transAxes)
    ax1.text(0.02, 0.95,'discharge', fontsize=13,transform=ax1.transAxes)


    plt.subplots_adjust(wspace=.05)
    ax1.tick_params(axis='both', labelsize=13)

    # x and y labels
    ax1.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
    ax0.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
    ax0.tick_params(axis='both', labelsize=13, direction='in')
    ax1.set_yticks([])

    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    ax0.set_ylabel('Absorption '+ordinal(deriv)+' derivative',fontsize=14,labelpad=10)


    ax0.minorticks_on()
    ax0.tick_params(which='both', labelsize=13, direction='in')

    ax1.tick_params(axis='both', labelsize=13, direction='in')
    ax1.minorticks_on()
    ax1.tick_params(which="minor", axis="x", direction="in")
    ax1.tick_params(which="minor", axis="y", color='w')



    cbar_ax = fig.add_axes([0.92, 0.125, 0.025, 0.755])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Potential vs. Li/Li$^+$ (V)',fontsize=14, rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=13)

    fig.suptitle('Cycle '+str(nb_cycle), fontsize=16)
    return fig

def plotAbsorptionDerivative(merged_df,nb_cycle=0,edge_intensity='inflection',deriv=1,colormap='viridis',width=12,height=6,plot_range=None):
    '''
    Function to plot the differential absorption intensity vs energy.
    
    :merged_df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or integer with the cycle you want to plot.
    :edge_intensity: Intensity value to get the edge energy value.
    :colormap: Name of the colormap you want to use for the plot.
    :width: Width of the graph.
    :height: Height of the graph.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
    # if no cycle number is selected then it just plots the first one
    total_cycles=len(merged_df['cycle number'].unique())
    edge=get_edge(merged_df,intensity=edge_intensity)
    try:
        if type(nb_cycle)==list:
            if not all(elem in merged_df['cycle number'].unique() for elem in nb_cycle):
                raise ValueError('The chosen cycle numbers are not in the sequence.')
        elif not nb_cycle:
            nb_cycle=merged_df['cycle number'].min()
        else:
            try:
                nb_cycle=int(nb_cycle)
                if nb_cycle not in merged_df['cycle number'].unique():
                    raise ValueError("Not a valid nb_cycle.")
                else:
                    nb_cycle=[nb_cycle]
            except:
                pass
    except:
        raise ValueError("Not a valid nb_cycle.")
        
    
    if not all(elem in merged_df['cycle number'].unique() for elem in nb_cycle):
        raise ValueError('The chosen cycle numbers are not in the sequence.')
    height=height*len(nb_cycle)
    if plot_range:
        if ((len(plot_range)==2) & (type(plot_range)==list)) & (all([isinstance(item, (int,float)) for item in plot_range])):
            rang=plot_range
        else:
            raise ValueError("Not a valid plot_range.")
    else:
        rang=[edge.mean()-20,edge.mean()+40]
    # colors
    cm=plt.get_cmap(colormap)
    norm = Normalize(vmin=merged_df['Ewe/V'].min(), vmax=merged_df['Ewe/V'].max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    colors = sm.to_rgba(merged_df['Ewe/V'])
    #  build plot
    fig = plt.figure(figsize=(width,height))
    gs = gridspec.GridSpec(1*len(nb_cycle), 2) 

    for j,cycle in enumerate(nb_cycle):
        # the first subplot
        ax0 = plt.subplot(gs[j,0])
        ax1 = plt.subplot(gs[j,1])

        condition = merged_df['cycle number']==cycle
        charge,discharge=merged_df['half cycle'][condition].unique()

        for i in merged_df[merged_df['half cycle']==charge].index:
            derivated_series=pd.Series(merged_df['normalized'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])]
            for k in range(deriv):
                derivated_series=derivated_series.diff()
            ax0.plot(pd.Series(merged_df['shifted energy'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],derivated_series,color=colors[i])
        for i in merged_df[merged_df['half cycle']==discharge].index:
            derivated_series=pd.Series(merged_df['normalized'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])]
            for k in range(deriv):
                derivated_series=derivated_series.diff()
            ax1.plot(pd.Series(merged_df['shifted energy'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])],derivated_series,color=colors[i])
        #### added

        xticks = ax0.xaxis.get_major_ticks()
        ax0.text(0.02, 0.9,'Cycle '+str(cycle), fontsize=13,transform=ax0.transAxes)
        if cycle==nb_cycle[-1]:
            ax0.text(0.8, 0.1,'charge', fontsize=13,transform=ax0.transAxes)
            ax1.text(0.8, 0.1,'discharge', fontsize=13,transform=ax1.transAxes)


        plt.subplots_adjust(wspace=.0) # no vertical space between plots
        plt.subplots_adjust(hspace=.0) # no horizontal space between plots
        ax1.tick_params(axis='both', labelsize=13)

        # labels and ticks

        ax0.minorticks_on()
        ax0.tick_params(which='both', axis='both',labelsize=13, direction='in')
        ax0.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
        # remove last tick label and first tick label for the necessary subplots
        yticks = ax0.yaxis.get_major_ticks()
        if i!=0:
            yticks[-1].label1.set_visible(False)
        if cycle!=nb_cycle[-1] and i>1:
            yticks[0].label1.set_visible(False)
        
        ax1.set_yticks([])
        ax1.minorticks_on()
        ax1.tick_params(axis='both', labelsize=13, direction='in')
        ax1.set_xlabel('Energy (eV)',fontsize=14,labelpad=10)
        ax1.tick_params(which="minor", axis="x", direction="in")
        ax1.tick_params(which="minor", axis="y", color='w')

    #shrink=0.8
    cbar_ax = fig.add_axes([0.92, 0.125, 0.02, 0.45])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Potential vs. Li/Li$^+$ (V)',fontsize=14, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=13)
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    fig.supylabel('Absorption '+ordinal(deriv)+' derivative',fontsize=14)

    return fig

def plot3d_XANES_vs_t(merged_df,nb_cycle=None,edge_intensity='inflection',
                      colormap_potential='viridis',colormap_cycle='plasma',
                      width=12,height=10,alpha=0.5,plot_range=None):
    '''
    Function to plot a 3D graph of all the XAS spectra over time.
    
    :merged_df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or integer with the cycle you want to plot.
    :edge_intensity: Intensity value to get the edge energy value.
    :colormap_potential: Name of the colormap you want to use for the plot (according to voltage).
    :colormap_cycle: Name of the colormap you want to use for the arrows pointing the cycles.
    :width: Width of the graph.
    :height: Height of the graph.
    :alpha: Transparency of the line collections.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
    
    total_cycles=len(merged_df['cycle number'].unique())
    edge=get_edge(merged_df,intensity=edge_intensity)
        
    try:
        nb_cycle=int(nb_cycle)
        if nb_cycle not in merged_df['cycle number'].unique():
            raise ValueError("Not a valid nb_cycle.")
    except ValueError: raise ValueError('Not a valid nb_cycle.')
    except: pass
        
    if not nb_cycle:
        nb_cycle=merged_df['cycle number'].unique()
        
    if plot_range:
        rang=plot_range
    else:
        rang=[edge.mean()-20,edge.mean()+40]

    # colormap from Ewe/V
    cm=plt.get_cmap(colormap_potential)
    norm = Normalize(vmin=merged_df['Ewe/V'].min(), vmax=merged_df['Ewe/V'].max())
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

    absorption=[]
    energy=[]
    zs = list(merged_df['absolute time/h'])

    for i in range(merged_df.shape[0]):
        x_data=list(pd.Series(merged_df['shifted energy'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])])
        y_data=list(pd.Series(merged_df['normalized'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])])
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
                zorder=(len(profiles) - i), color=sm.to_rgba(merged_df['Ewe/V'])[i], alpha=0.5)
    

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

    merged_df[merged_df['cycle number']==1].first_valid_index()
    #FIRST ANNOTATION
    cm_cycle=plt.get_cmap(colormap_cycle)
    # color map from the total number of cycles in the DF, not from the lenght of the input nb_cycles
    color_cycle=cm_cycle(np.linspace(0, 1, len(merged_df['cycle number'].unique())))
    for i,cycle in enumerate(merged_df['cycle number'].unique()):
        where=merged_df[merged_df['cycle number']==cycle].first_valid_index()
        maximum=max(merged_df['normalized'][where])
        index_max=max(range(len(merged_df['normalized'][where])), key=(merged_df['normalized'][where]).__getitem__)
        maximum_energy=merged_df['shifted energy'][where][index_max]
        when=merged_df['absolute time/h'][where]
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

def plot2d_XANES_vs_t(merged_df,nb_cycle=None,edge_intensity='inflection',colormap='tab20b', width=8,height=6,plot_range=None,hlines=False):
    '''
    Function to plot a 2D intensity graph of all the XAS spectra over time.
    
    :merged_df: Pandas dataframe with the data from the EC Lab file merged with the XAS files data.
    :nb_cycle: List of the cycles you want to plot. Or integer with the cycle you want to plot.
    :edge_intensity: Intensity value to get the edge energy value.
    :colormap: Name of the colormap you want to use for the plot.
    :width: Width of the graph.
    :height: Height of the graph.
    :plot_range: List [x,y] containing the energy range of the plot.
    :return: Plot.
    '''
        # if no cycle number is selected then it just plots all of them
    total_cycles=len(merged_df['cycle number'].unique())
    edge=get_edge(merged_df,intensity=edge_intensity)
    try:
        if type(nb_cycle)==list:
            if not all(elem in merged_df['cycle number'].unique() for elem in nb_cycle):
                raise ValueError('The chosen cycle numbers are not in the sequence.')
        elif not nb_cycle:
            nb_cycle=merged_df['cycle number'].unique()
        else:
            try:
                nb_cycle=int(nb_cycle)
                if nb_cycle not in merged_df['cycle number'].unique():
                    raise ValueError("Not a valid nb_cycle.")
                else:
                    nb_cycle=[nb_cycle]
            except:
                pass
    except:
        raise ValueError("Not a valid nb_cycle.")

    if plot_range:
        if ((len(plot_range)==2) & (type(plot_range)==list)) & (all([isinstance(item, (int,float)) for item in plot_range])):
            rang=plot_range
        else:
            raise ValueError("Not a valid plot_range.")
    else:
        rang=[edge.mean()-20,edge.mean()+40]
    
    #sub_df=merged_df[merged_df['cycle number'].isin(nb_cycle)]
    all_absorption=[]
    for i in merged_df.index:
        #x_data=list(pd.Series(merged_df['shifted energy'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])])
        y_data=list(pd.Series(merged_df['normalized'][i])[pd.Series(merged_df['shifted energy'][i]).between(rang[0],rang[1])])
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
        sub_df=merged_df[merged_df['cycle number']==cycle]
        absorption=[]
        energy=[]
        times=[]
        zs = list(sub_df['absolute time/h'])
        for i in sub_df.index:
            x_data=list(pd.Series(sub_df['shifted energy'][i])[pd.Series(sub_df['shifted energy'][i]).between(rang[0],rang[1])])
            y_data=list(pd.Series(sub_df['normalized'][i])[pd.Series(sub_df['shifted energy'][i]).between(rang[0],rang[1])])
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
    ax0.set_ylim(0,merged_df['absolute time/h'].max())
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
    ax1.set_ylim(0,merged_df['absolute time/h'].max())
    
    # add horizontal lines at potential maxima
    #peaks, _ = find_peaks(sub_df['Ewe/V'], prominence=0.1)
    #for idx in peaks:
        #ax0.axhline(merged_df['absolute time/h'][idx], color='b', ls='dashed', linewidth=0.7)
        #ax1.axhline(merged_df['absolute time/h'][idx], color='b', ls='dashed', linewidth=0.7)
    return fig
