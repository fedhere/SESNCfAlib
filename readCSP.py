
import pandas as pd
import os
import numpy as np
import sys

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib")+"/templates"
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)


import snclasses as snstuff
reload(snstuff)
import templutils as templutils
import utils as snutils
import fitutils as fitutils
import myastrotools as myas
import matplotlib as mpl
import glob

def readlatex(fname, skip=0, names=[]):
    allcsps = pd.read_csv(os.getenv("SESNCFAlib") +
                          "/../literaturedata/cspsesn/" + fname, sep='&',
                          header=None,
                          skiprows=skip,
                          skipfooter=3,
                          engine='python', names = names)

    allcsps.index = allcsps.JD
    tmp = allcsps.index.map(lambda x: x.replace("\multicolumn{%d}{c}{"%len(names),"").\
                            replace("}\\","").replace("~","").\
                            replace(" ","").replace("\\",""))

    for i in range(len(tmp)):
        if tmp[i].startswith("SN"):
            tmp[i] = tmp[i] 
        else:
            tmp[i] = tmp[i-1]
        
    allcsps.index = tmp
    allcsps["JD"] = pd.to_numeric(allcsps.JD, errors="coerce")
    allcsps.index = allcsps.index.map(lambda x:
                                      x if x.startswith("SN") else None)
    return allcsps
##read Opt Swope
allcspsOptSwope = readlatex("Tab4.tex", skip=16, names=["JD","u","g","r","i","B","V"])

# read opt duPont

allcspsOptdp = readlatex("Tab5.tex", skip=15, names=["JD","u","g","r","i","B","V"])


#read NIR
allcspsNIR1 = readlatex("Tab6.tex", skip=13, names=["JD","Y","J","H"])
allcspsNIR2 = readlatex("Tab7.tex", skip=13, names=["JD","Y","J","H"])
allcspsNIR3 = readlatex("Tab8.tex", skip=13, names=["JD","Y","J","H"])

allcspsOpt = pd.concat([allcspsOptSwope, allcspsOptdp])
allcspsNIR = pd.concat([allcspsNIR1, allcspsNIR2, allcspsNIR3])
allcsps = pd.concat([allcspsOpt, allcspsNIR])

allsne = allcsps.index.drop_duplicates()

dtypes = {
    'names': ('mjd', 'w2', 'dw2', 'm2', 'dm2', 'w1', 'dw1', 'U', 'dU', 'V',
              'dV', 'B', 'dB', 'R', 'dR', 'I', 'dI', 'u', 'du', 'b', 'db', 'v',
              'dv', 'g', 'dg', 'r', 'dr', 'i', 'di', 'z', 'dz', 'Y', 'dY', 'J',
              'dJ', 'H', 'dH', 'K', 'dK'),
    'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',
                'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',
                'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',
                'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4')
}


for sn in allsne:#['SN2009bb']:
     tmp = allcsps[allcsps.index == sn]

     bands = ['B','V','u','g','r','i','J','H','Y']
     for c in bands:
          
          tmp["d" + c] = [float('.' + 
                                cc.split('(')[1].replace(' ','').replace(')','').replace('\\','')) 
                          if (not (isinstance(cc, float) and  np.isnan(cc)) and not (cc is None) 
                              and not (not isinstance(cc, float) and 'dot' in cc)) else np.nan for cc in tmp[c].values ]
          tmp[c] = [float(cc.split('(')[0]) if (not (isinstance(cc, float) and  np.isnan(cc)) and not (cc is None) 
                                                and not (not isinstance(cc, float) and 'dot' in cc)) else np.nan for cc in tmp[c].values ]

               
     N = len(tmp)
     snarray = np.zeros(N, dtype=dtypes)
     for i, c in enumerate(dtypes['names']):
          if c == 'mjd':
               snarray['mjd'] = tmp.JD.values - 2400000.5  #* 6 
          elif c in bands :
               snarray[c] = tmp[c].values
               snarray['d' + c] = tmp['d' + c].values
          elif not c.startswith('d'):
               snarray[c] = np.zeros(len(tmp)) * np.nan

     vmax=None
     verbose=True
     thissn = snstuff.mysn(sn, noload=True, verbose=verbose)
     thissn.readinfofileall(verbose=False, earliest=False, loose=True)
     thissn.setVmax(loose=True)
     if not vmax is None:
          thissn.Vmax = vmax
          print ("Vmax", thissn.Vmax)
     if verbose: 
          thissn.printsn(photometry=True)
          thissn.formatlitsn(snarray, csp=True)
