#!/usr/bin/env python

import sys,os,glob
#,re,numpy,math,pyfits,glob,shutil,glob
import optparse
import numpy as np
import pylab as pl
from scipy import stats as stats 
from scipy import optimize
from scipy.interpolate import interp1d
import math 
from mpmath import polyroots
import time
from scipy.signal import cspline1d, cspline1d_eval
import pickle as pkl


from templutils import *

allbands = ['U','B','V','R','I','g','r','i','z']
bandscols = {'U':1,'B':3,'V':5,'R':7,'I':9,'g':13, 'r':15, 'i':17,'z':11}

def smoothListGaussian(data,strippedXs=False,degree=15):  
     window=degree*2-1  
     weight=np.array([1.0]*window)  
     weightGauss=[]  
     for i in range(window):  
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(np.exp((4*(frac))**2))  
         weightGauss.append(gauss)  
     weight=np.array(weightGauss)*weight  
     smoothed=np.zeros(len(data),float)
     for i in range(len(smoothed)-window):
          smoothed[i+window/2+1]=np.sum(np.array(data[i:i+window])*weight)/np.sum(weight)  
     smoothed[0:window/2]=data[0:window/2]
     smoothed[-window/2:]=data[-window/2:]
     return np.array(smoothed)  

if __name__=='__main__':
    parser = optparse.OptionParser(usage="makelittemplate.py ", conflict_handler="resolve")
    parser.add_option('--showonly', default=False, action="store_true",
                      help='show only')
    parser.add_option('-b','--band', default='V' , type="string",
                      help='degree of polinomial fit')
    parser.add_option('-i','--inst', default='all' , type="string",
                      help='instrument to be used: shooter, mini, keploer, or all')
    options,  args = parser.parse_args()
    if len(args)>1:
        sys.argv.append('--help')
    
        options,  args = parser.parse_args()
        sys.exit(0)
        
    if options.band == 'V': band = 'V'
    else:band = options.band
    if band not in allbands:
        print "band "+band+" not available"
        sys.exit()


     import globaltemplate
     template= globaltemplate.template#.template[b]['Ib']
     templates=np.loadtxt('templog.tmp',usecols=(0,1,3,4,5),dtype={'names': ('sn','type','stretch','xoff','yoff'),'formats': ('S6','S6','f','f','f')}, comments='#')

     
     su=setupvars()
