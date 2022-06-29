import numpy as np, pandas as pd, re, os, pathlib, glob
from datetime import datetime, timedelta
from time import mktime
#from galvani import BioLogic


def getECdf(EC_filename, with_settings=0, with_header_dict=0):
    filename, file_extension = os.path.splitext(EC_filename)
    if file_extension!='.mpt': #any(line for line in enumerate(linesEC) if line.startswith('Nb header lines'))==False or 
        raise ValueError("File type not suppported. It must have an .mpt extension.")
    else:
        f = open(EC_filename, "r", encoding="ISO-8859-1")
        linesEC = f.readlines()
        rawEC = "".join(linesEC)
        f.close()
        # check which delimiter is used for decimals
        if rawEC.count('.')>rawEC.count(','):
            delim='.'
        else:
            delim=','
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
            # add new datetime column with format YYYY-MM-DD HH:mm:ss.f
            df['datetime']=acDate+pd.TimedeltaIndex(df['time/s'], unit='S')
            # reconvert to string format
            df['datetime']=df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
        # settings lines
        if any(i for i,line in enumerate(linesEC) if line.startswith('Ns')):
                # get settings dataframe
                settingsHeaderIdx=next(i for i,line in enumerate(linesEC) if line.startswith('Ns'))
                settingsBottomIdx=next(i for i,line in enumerate(linesEC[settingsHeaderIdx:]) if len(line)==1)+settingsHeaderIdx
                setLines=linesEC[settingsHeaderIdx:settingsBottomIdx]
        # settings df
        if with_settings==1:
            try: setLines
            except NameError: setLines = None
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
            headerLineswoutSettings=linesEC[:mainHeaderIdx]
            headerDict=getECHeaderDict(headerLineswoutSettings)
            df.attrs = headerDict
            return df

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
            if key.strip() in headerDict:
                headerDict[key.strip()]+=', '+value.strip()
            else:
                headerDict[key.strip()]=value.strip()
        elif ('Loop' in line) and ('from point number' in line):
            frmIdx=line.find('from')
            editLine=line[0:frmIdx]+':'+line[frmIdx:]
            key, value = editLine.split(':',1)
            headerDict[key.strip()]=value.strip()
    return headerDict

def getXASfilesdf(XAS_folder):
    # absolute path to search all text files inside a specific folder
    path = XAS_folder+'/*.txt'
    files = glob.glob(path)
    if len(files)==0:
        raise ValueError("XAS folder empty. Choose another folder.")
    else:
        XAS_filelist = []
        for f in files:
            with open(f, 'r') as openFile:
                lines = openFile.readlines()
                keywordStartTime = '#Time at start='
                keywordElapsedTime = '#Time from start (seconds)='
                startTime=datetime.strptime(next(line for line in lines if keywordStartTime in line).strip(keywordStartTime).strip(), "%Y-%m-%d %H:%M:%S.%f")
                elapsedTime=float(next(line for line in lines if keywordElapsedTime in line).strip(keywordElapsedTime).strip())
                XAS_filelist.append([os.path.basename(f),startTime,elapsedTime])
        #XAS files df
        XAS_files_df=pd.DataFrame(columns = ['filename','datetime','elapsed time/s'],data=XAS_filelist)
        XAS_files_df=XAS_files_df.sort_values(by='datetime', ignore_index=True)
    return XAS_files_df

def mergeEC_XASdfs(EC_df,XAS_files_df):
    # merge electro and XAS files dataframes based on approximate datetime
    # convert 'datetime' column string values in EC df to datetime format
    EC_df['datetime']=pd.to_datetime(EC_df['datetime'])
    #merged_df = pd.merge_asof(left=XAS_files_df,right=EC_df,on='datetime',direction='nearest',tolerance=pd.Timedelta('10 min'))
    merged_df = pd.merge_asof(XAS_files_df.rename(columns={'datetime':'datetime XAS','elapsed time/s':'elapsed time XAS/s'}),
                              EC_df.rename(columns={'datetime':'datetime EC'}),
                              left_on='datetime XAS',
                              right_on='datetime EC',
                              direction='nearest',
                              tolerance=pd.Timedelta('10 min'))
    merged_df['endtime XAS']=merged_df['datetime XAS']+pd.TimedeltaIndex(merged_df['elapsed time XAS/s'], unit='S')
    merged_df.rename(columns = {'datetime XAS':'starttime XAS'}, inplace = True)
    # choose which columns to have in your text file
    cols = ['filename','starttime XAS','elapsed time XAS/s','endtime XAS','datetime EC','Ewe/V', '<I>/mA', 'x']
    merged_df = merged_df[cols]
    merged_df['absolute time/s'] = (merged_df['datetime EC']-merged_df['datetime EC'][0]).dt.total_seconds()
    merged_df['absolute time/min'] = merged_df['absolute time/s']/60
    merged_df['absolute time/h'] = merged_df['absolute time/min']/60
    merged_df['starttime XAS'] = merged_df['starttime XAS'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    merged_df['endtime XAS'] = merged_df['endtime XAS'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    if merged_df.dropna(how='all', axis=0, subset=['Ewe/V','<I>/mA','x']).shape[0]==0:
        raise ValueError("Could not correctly merge the EC file to a list of XAS files. Please check the folder selection.")
    else:
        merged_df['datetime EC'] = merged_df['datetime EC'].dt.strftime('%Y-%m-%d %H:%M:%S.%f').str[:-3]
    return merged_df

def get_output(EC_filename,XAS_folder,output_name):
    electrodf=getECdf(EC_filename)
    XASdf=getXASfilesdf(XAS_folder)
    merged=mergeEC_XASdfs(electrodf,XASdf)
    merged.to_csv(output_name, index=False, sep='\t')
    return
