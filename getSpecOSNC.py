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
print (sn)

tmp = pd.read_json("../sne.space_downloads/"+sn+".json")

snkey = tmp.columns
tmp = tmp[snkey[0]]
print (tmp)


if not 'spectra' in tmp.keys():
     print ("no spectra")
     sys.exit()
N = len(tmp['spectra'])
if N<1:
     print ("no spectra")
     sys.exit()

for i,sp in enumerate(tmp['spectra']):
     name = sp['time']
     fout = open("%s_%s_spec.dat"%(sn, name), "w")
     fout.write( "#%s %s \n"%(sp["u_time"], sp["u_wavelengths"]))
     for dp in sp['data']:
          print (dp)
          fout.write("%f %e \n"%(float(dp[0]), float(dp[1])))
     fout.close()
