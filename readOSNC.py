import pandas as pd
import numpy as np
import json
import os
import glob 
import inspect
import optparse
import time
import copy
import os
import pylab as pl
import numpy as np
import scipy
import json
import sys
import pickle as pkl

import scipy as sp
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
from scipy import stats as spstats 
from scipy import integrate

from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline,splrep, splev
from scipy import interpolate

s = json.load( open(os.getenv ('PUI2015')+"/fbb_matplotlibrc.json") )
pl.rcParams.update(s)

cmd_folder = os.path.realpath(os.getenv("SESNCFAlib"))

if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

import snclasses as snstuff
import templutils as templutils
import utils as snutils
import fitutils as fitutils
import myastrotools as myas
import matplotlib as mpl

sn = sys.argv[1]
#'SN1993J'
#sn = 'SN2008bo'
print sn

tmp = pd.read_json("../sne.space_downloads/"+sn+".json")
snkey = tmp.columns
tmp = tmp[snkey[0]]

myref = -99
D11ref = -99
for ref in tmp['sources']:
    if 'reference' in ref.keys() and 'Bianco' in ref['reference']:
        myref=ref['alias']
    if 'reference' in ref.keys() and 'Drout' in ref['reference']:
         D11ref = ref['alias']

if not 'photometry' in tmp.keys(): sys.exit()
N = len(tmp['photometry'])
if N<=1: sys.exit()

dtypes={'names':('mjd','w2','dw2','m2','dm2','w1','dw1','U','dU','V','dV','B','dB','R','dR','I','dI','u','du','b','db','v','dv','g','dg','r','dr','i','di','z','dz','Y','dY','J','dJ','H','dH','K','dK'),
 'formats':('f4','f4','f4','f4', 'f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4','f4', 'f4','f4','f4','f4','f4','f4','f4','f4','f4','f4')}


snarray = np.zeros(N, dtype=dtypes)
print N
for i,dp in enumerate(tmp['photometry']):
    print (dp)
    #print ref['time']
    for j in range(len(snarray[i])):
        snarray[i][j] = np.nan 
    
    if 'time' in dp.keys() and 'band' in dp.keys() and  dp.keys() and 'magnitude' in dp.keys():
        print "here", dp['source'] , myref, D11ref
        if dp['source'] == myref: continue
        if dp['source'] == D11ref: continue
        band = dp['band']
        #print band
        if band.endswith("'"): band = band.strip("'")
        if band == 'Ks': band = 'K'
        elif band == 'W1': band = 'w1'
        elif band == 'W2': band = 'w2'
        elif band == 'M2': band = 'm2'
        print band, dtypes['names']
        if not band in dtypes['names']: continue
        print i, band, dp['magnitude']
    
        snarray[i]['mjd'] = dp['time']
        snarray[i][band.replace("'","")] = dp['magnitude']
        if 'e_magnitude' in dp:
             snarray[i]['d'+band.replace("'","")] = dp['e_magnitude']
        else:
             snarray[i]['d'+band.replace("'","")] = 0.5
        #print snarray
thissn = snstuff.mysn(sn, noload=True)
thissn.printsn(photometry=True)
thissn.formatlitsn(snarray)
