import numpy as np, pandas as pd, re, os, pathlib, glob
from datetime import datetime, timedelta
from time import mktime
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter
#from galvani import BioLogic


def getECdf(EC_filename, with_settings=0, with_header_dict=0,acTime=1):
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
        df['datetime']=df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    except NameError:
        print("There is no information on the start date of the EC acquisition for file "+EC_filename+".")
    # settings lines
    if any(i for i,line in enumerate(linesEC) if line.startswith('Ns')):
        # get settings dataframe
        settingsHeaderIdx=next(i for i,line in enumerate(linesEC) if line.startswith('Ns'))
        settingsBottomIdx=next(i for i,line in enumerate(linesEC[settingsHeaderIdx:]) if len(line)==1)+settingsHeaderIdx
        setLines=linesEC[settingsHeaderIdx:settingsBottomIdx]
    # settings df
    try: setLines
    except NameError: setLines = None
    if with_settings==1:
        if setLines:
            # get settings dataframe
            df2=ECsettingsdf(setLines,delim)
            headerLineswoutSettings=linesEC[:settingsHeaderIdx]
            headerLineswoutSettings.extend(linesEC[settingsBottomIdx:mainHeaderIdx])
        else:
            headerLineswoutSettings=linesEC[:mainHeaderIdx]
            df2=None
            print('Not able to get settings dataframe')
        headerDict=getECHeaderDict(headerLineswoutSettings)
        df.attrs = headerDict
        return df, df2
    else:
        if setLines:
            headerLineswoutSettings=linesEC[:settingsHeaderIdx]
            headerLineswoutSettings.extend(linesEC[settingsBottomIdx:mainHeaderIdx])
        else:
            headerLineswoutSettings=linesEC[:mainHeaderIdx]
        headerDict=getECHeaderDict(headerLineswoutSettings)
        df.attrs = headerDict
        return df

def getECdffromseveral(*args):
    base_df=getECdf(args[0])
    for file in args[1:]:
        next_df=getECdf(file)
        #next_df['Capacity/mA.h']=next_df['Capacity/mA.h']+base_df['Capacity/mA.h'].iloc[-1]
        #next_df.loc[next_df['cycle number'].isin([0,1]),'Capacity/mA.h']=next_df['Capacity/mA.h']+base_df['Capacity/mA.h'].iloc[-1]
        next_df.loc[next_df['cycle number']>0,'cycle number']=next_df['cycle number']-1
        #next_df.loc[next_df['cycle number'].isin([0,1]),'Capacity/mA.h']=next_df['Capacity/mA.h']+base_df['Capacity/mA.h'].iloc[-1]
        next_df['cycle number']=next_df['cycle number']+base_df['cycle number'].iloc[-1]
        next_df.loc[next_df['half cycle']>0,'half cycle']=next_df['half cycle']-next((x for x in next_df['half cycle'].unique() if x), None)
        next_df['half cycle']=next_df['half cycle']+base_df['half cycle'].iloc[-1]
        next_df.loc[next_df['half cycle']==next_df['half cycle'].unique()[0],'Capacity/mA.h']=next_df['Capacity/mA.h']+base_df['Capacity/mA.h'].iloc[-1]
        next_df['Ns']=next_df['Ns']+base_df['Ns'].iloc[-1]
        # next_df['time/s']=base_df['time/s'].iloc[-1]+(datetime.strptime(next_df['datetime'], '%Y-%m-%d %H:%M:%S.%f')-datetime.strptime(base_df['datetime'].iloc[-1], '%Y-%m-%d %H:%M:%S.%f')).total_seconds()
        base_df=pd.concat([base_df, next_df], ignore_index=True)
    return base_df

def ECsettingsdf(setLines,delim):
    if delim==',':
        setLines=[(re.sub('\s{2,}', '\t',line)).replace(',','.') for line in setLines]
    elif delim=='.':
        setLines=[(re.sub('\s{2,}', '\t',line)) for line in setLines]
    setLines2=[line.strip().split('\t') for line in setLines]
    # raw settings df
    set_df=pd.DataFrame(list(map(list, zip(*setLines2))))
    new_header=set_df.iloc[0] #grab the first row for the header
    set_df=set_df[1:] #take the data without the header row
    set_df.columns = new_header #set the header row as the df header
    # modified settings df
    #set_df=ECsettingsdf_transform(set_df)
    # transform to numeric variables if possible
    for c in new_header:
        try:
            set_df[c] = pd.to_numeric(set_df[c])
        except:
            pass
    #set_df.reset_index(drop=True,inplace=True)
    set_df=set_df.transpose()
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
    #WARNING
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

def getXASfilesdf(XAS_folder,filterlist=[]):
    # absolute path to search all text files inside a specific folder
    #path = XAS_folder+'/*.txt'
    #files = glob.glob(path)
    # list to store files name
    files = []

    for (dir_path, dir_names, file_names) in os.walk(XAS_folder):
        files.extend(os.path.join(dir_path, f) for f in file_names if f.endswith(".txt") and all(word in f for word in filterlist))

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
        XAS_df = pd.DataFrame(XAS_dictlist)
        XAS_df=XAS_df.sort_values(by='datetime', ignore_index=True)
        XAS_df['average time XAS']=XAS_df['datetime']+pd.TimedeltaIndex(XAS_df['elapsed time/s'], unit='S')
        XAS_df.drop_duplicates(subset=XAS_df.columns[:3],inplace=True)
        XAS_df=XAS_df.sort_values(by='average time XAS', ignore_index=True)
        cols1=list(XAS_df.columns[:XAS_df.columns.get_loc('elapsed time/s')+1])
        cols2=list(XAS_df.columns[XAS_df.columns.get_loc('elapsed time/s')+2:-1])
        #XAS_df=XAS_df[cols1+['average time XAS']+cols2]
    return XAS_df

def checkDelim(raw):
    if raw.count('.')>raw.count(','):
        delim='.'
    else:
        delim=','
    return delim

def getSingleXASdict(filename):
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

def mergeEC_XASdfs(EC_df,XAS_files_df):
    # merge electro and XAS files dataframes based on approximate datetime
    # check if datetime column exists in EC df
    if 'datetime' not in EC_df.columns:
        raise ValueError("Cannot proceed without start date.")
    # convert 'datetime' column string values in EC df to datetime format
    EC_df['datetime']=pd.to_datetime(EC_df['datetime'])
    #merged_df = pd.merge_asof(left=XAS_files_df,right=EC_df,on='datetime',direction='nearest',tolerance=pd.Timedelta('10 min'))
    merged_df = pd.merge_asof(XAS_files_df.rename(columns={'datetime':'starttime XAS','elapsed time/s':'elapsed time XAS/s'}),
                              EC_df.rename(columns={'datetime':'datetime EC'}),
                              left_on='average time XAS',
                              right_on='datetime EC',
                              direction='nearest',
                              tolerance=pd.Timedelta('10 min'))
    #merged_df['endtime XAS']=merged_df['datetime XAS']+pd.TimedeltaIndex(merged_df['elapsed time XAS/s'], unit='S')
    #merged_df.rename(columns = {'datetime XAS':'starttime XAS'}, inplace = True)
    # choose which columns to have in your text file
    cols_XAS = list(XAS_files_df.rename(columns={'datetime':'starttime XAS','elapsed time/s':'elapsed time XAS/s'}).columns)
    cols_EC = ['datetime EC','Ewe/V', '<I>/mA', 'x','Capacity/mA.h','half cycle','cycle number']
    cols = cols_XAS + cols_EC
    #cols = ['filename','starttime XAS','elapsed time XAS/s','average time XAS','sample T (C)','datetime EC','Ewe/V', '<I>/mA', 'x','Capacity/mA.h','half cycle','cycle number']
    merged_df = merged_df[cols]
    merged_df.dropna(how='all', axis=0, subset=['Ewe/V','<I>/mA','x'], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    merged_df['absolute time/s'] = (merged_df['datetime EC']-merged_df['datetime EC'][0]).dt.total_seconds()+merged_df['elapsed time XAS/s']
    #merged_df['absolute time/s'] = merged_df['elapsed time XAS/s'].cumsum()
    merged_df['absolute time/min'] = merged_df['absolute time/s']/60
    merged_df['absolute time/h'] = merged_df['absolute time/min']/60
    merged_df['starttime XAS'] = merged_df['starttime XAS'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    merged_df['average time XAS'] = merged_df['average time XAS'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    #if merged_df.dropna(how='all', axis=0, subset=['Ewe/V','<I>/mA','x']).shape[0]==0:
    if merged_df.shape[0]==0:
        raise ValueError("Could not correctly merge the EC file to a list of XAS files. Please check the folder selection.")
    else:
        merged_df['datetime EC'] = merged_df['datetime EC'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
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

def plotUI_vs_time(electrodf,length=15,height=5):
    #electrodf=getECdf(ec_path)
    fig, ax1 = plt.subplots(figsize=(length,height))
    # add an additional y axis for the current
    ax2 = ax1.twinx()
    # get time series
    if 'datetime' in electrodf.columns:
        timeSeries=(electrodf['datetime']-electrodf['datetime'][0]).dt.total_seconds()
    elif electrodf[electrodf['time/s']==0].shape[0]>1:
        timeSeries=list(electrodf['time/s'][:electrodf[electrodf['time/s']==0].index[1]])
        for i in electrodf[electrodf['time/s']==0].index[1:-1]:
            timeSeries.extend(list(electrodf['time/s'][i:i+1]+electrodf['time/s'][i-1]))
        ax1.plot(newTimeSeries/3600,electrodf['Ewe/V'],color="blue")
    else:
        timeSeries=electrodf['time/s']/3600
    # plot V vs. time
    ax1.plot(timeSeries/3600,electrodf['Ewe/V'],color="blue")
    # and plot I vs. time
    ax2.plot(timeSeries/3600,electrodf['<I>/mA'],color="green")
    # set the labels and the colors of your curves
    ax1.set_xlabel("Time/h",fontsize=14)
    ax1.set_ylabel("Ewe/V",color="blue",fontsize=14)
    ax2.set_ylabel("<I>/mA",color="green",fontsize=14)

    # let's also color the axes
    ax1.tick_params(axis='y',colors="blue")
    ax2.tick_params(axis='y',colors="green")
    ax2.spines['right'].set_color("green")
    ax1.tick_params(axis='both', labelsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    return

def plotCapacity_vs_U(electrodf,nb_cycle=None,length=8,height=5,mass=1000):
    #electrodf=getECdf(ec_path)
    mass=mass/1000 # mg to g
    min_cycles=int(electrodf['cycle number'].min())
    #total_cycles=int(electrodf['cycle number'].max())
    total_cycles=len(electrodf['cycle number'].unique())
    min_half_cycles=int(electrodf['half cycle'].min())
    #total_half_cycles=int(electrodf['half cycle'].max())
    total_half_cycles=len(electrodf['half cycle'].unique())
    
    try:
        nb_cycle=int(nb_cycle)
        if nb_cycle>total_cycles:
            raise ValueError("nb_cycle higher than cycles performed.")
        elif nb_cycle<min_cycles:
            raise ValueError("Not a valid nb_cycle.")
        else:
            nb_cycle=[nb_cycle]
    except ValueError: raise ValueError('Not a valid nb_cycle.')
    except: pass
        
    if not nb_cycle:
        #if nb_cycle==0:
        #    raise ValueError("Not a valid nb_cycle.")
        #else:
        #nb_cycle=range(min_cycles,total_cycles+1)
        nb_cycle=electrodf['cycle number'].unique()
    #elif 0 in nb_cycle:
    #    raise ValueError("A cycle number must be higher than 0.")

    #color_charge = cm.seismic(np.linspace(0, 0.5, len(nb_cycle)+2))[1:-1]
    color_charge = cm.seismic(np.linspace(0, 0.5, total_cycles+2))[1:-1]
    #color_discharge = cm.seismic(np.linspace(1, 0.5, len(nb_cycle)+2))[1:-1]
    color_discharge = cm.seismic(np.linspace(1, 0.5, total_cycles+2))[1:-1]
    
    fig, ax = plt.subplots(figsize=(length,height))
    i=0
    #for half_cycle in range(min_half_cycles,total_half_cycles+1):
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
            #i+=0.5
    
    ax.set_ylabel("Potential V vs. Li/Li$^+$",fontsize=14)
    if mass==1:
        ax.set_xlabel("Capacity mAh",fontsize=14)
    else:
        ax.set_xlabel("Capacity mAh$\cdot$g$^{-1}$",fontsize=14)
    margin=0.05
    xmin=electrodf['Capacity/mA.h'].min()-(electrodf['Capacity/mA.h'].max()-electrodf['Capacity/mA.h'].min())*margin/(1-margin*2)
    xmax=electrodf['Capacity/mA.h'].max()+(electrodf['Capacity/mA.h'].max()-electrodf['Capacity/mA.h'].min())*margin/(1-margin*2)
    ymin=electrodf['Ewe/V'].min()-(electrodf['Ewe/V'].max()-electrodf['Ewe/V'].min())*margin/(1-margin*2)
    ymax=electrodf['Ewe/V'].max()+(electrodf['Ewe/V'].max()-electrodf['Ewe/V'].min())*margin/(1-margin*2)
    ax.set_xlim(xmin/mass,xmax/mass)
    ax.set_ylim(ymin,ymax)
    ax.tick_params(axis='both', labelsize=12)
    return

def plotdQdU_vs_U(electrodf,nb_cycle=None,reduce_by=1,boxcar=1,savgol=(1,0),colormap='plasma',length=10,height=6):
    newECdf=electrodf.copy(deep=True)
    #electrodf=getECdf(ec_path)
    min_cycles=int(electrodf['cycle number'].min())
    max_cycles=int(electrodf['cycle number'].max())
    total_cycles=len(electrodf['cycle number'].unique())
    min_half_cycles=int(electrodf['half cycle'].min())
    max_half_cycles=int(electrodf['half cycle'].max())
    total_half_cycles=len(electrodf['half cycle'].unique())
    
    try:
        nb_cycle=int(nb_cycle)
        if nb_cycle not in electrodf['cycle number'].unique():
            raise ValueError("Not a valid nb_cycle.")
        else:
            nb_cycle=[nb_cycle]
    except ValueError: raise ValueError('Not a valid nb_cycle.')
    except: pass
        
    if not nb_cycle:
        #if nb_cycle==0:
        #    raise ValueError("Not a valid nb_cycle.")
        #else:
        nb_cycle=electrodf['cycle number'].unique()
    #elif 0 in nb_cycle:
     #   raise ValueError("A cycle number must be higher than 0.")

    newECdf['dQdV']=newECdf['Capacity/mA.h'].diff()/newECdf['Ewe/V'].diff()
    # apply filters/smoothing
    newECdf=newECdf.iloc[::reduce_by]
    newECdf['dQdV']=newECdf['dQdV'].rolling(boxcar).mean()
    newECdf['dQdV']=savgol_filter(newECdf['dQdV'],savgol[0],savgol[1])

    cm=plt.get_cmap(colormap)
    # color map from the total number of cycles in the DF, not from the lenght of the input nb_cycles
    color_dqdv=cm(np.linspace(0, 1, len(newECdf['cycle number'].unique())))
    
    fig, ax = plt.subplots(figsize=(length,height))
    #coloridx=0
    for i, cycle in enumerate(newECdf['cycle number'].unique()):
        if cycle in nb_cycle:
            subdf=newECdf[newECdf['cycle number']==cycle]
            ax.scatter(subdf['Ewe/V'],subdf['dQdV'],s=2,color=color_dqdv[i],label='cycle '+str(int(cycle)))

    # y limits
    ylim=(abs(newECdf['dQdV'].quantile(0.95))+abs(newECdf['dQdV'].quantile(0.05)))/2
    ax.set_ylabel("dQ/dV",fontsize=14)
    ax.set_xlabel("Potential V vs. Li/Li$^+$",fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    #ax.set_xlim(3.,4.3)
    ax.set_ylim(-ylim,ylim)
    
    norm = Normalize(vmin=min_cycles, vmax=total_cycles)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    if len(nb_cycle)>5:
        cbar = fig.colorbar(sm)
        cbar.set_label('Cycle', rotation=270, labelpad=10)
    else:
        leg = ax.legend(loc='best',prop={'size': 12})
        #change the marker size
        #for legendHandles in leg.legendHandles:
        #    legendHandles._legmarker.set_markersize(6)
    return

