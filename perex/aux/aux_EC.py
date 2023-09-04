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

def getECdf_single(EC_filename, with_settings=0, acTime=1):
    '''
    Function to get a pandas dataframe from a single EC Lab file (must have an .mpt extension).
    
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
                #if with_settings!=0:
                main_settings_df=ECsettingsdf(setLines,delim)
                newDict = headerDict | {"Settings": main_settings_df.to_dict()}
                attrs = attrs | newDict
                newStart=setBottom
    df.attrs = attrs
    if with_settings!=0: # get an additional dataframe for settings only
        return df, main_settings_df
    else:
        return df

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
    '''
    Attempt to append 'unit' columns directly to the variable name but sometimes units are not uniform over the entire df (e.g. mA, mA and µA)
    '''
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
