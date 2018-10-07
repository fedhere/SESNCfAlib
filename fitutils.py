import sys,os,glob, inspect
#,re,numpy,math,pyfits,glob,shutil,glob
import optparse
import scipy as sp
import numpy as np
import pylab as pl
from scipy import optimize
from scipy.interpolate import interp1d
from mpmath import polyroots
import pprint, pickle, copy
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]) + "/templates")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
from templutils import *

import time
global counterhere
counterhere=0

EPS = 1.0E-20

def checkfit(y, yfit, closenough):#, sig, fracdev, ngood, weights, scaleres):
    bfac=6.0
    dev = y-yfit
    ##following idl robust_sigma
    # First, the median absolute deviation MAD about the median:
    #    dev0=np.median(dev)
    mad=np.median(np.abs(dev))/0.6745
    # If the MAD=0, try the MEAN absolute deviation:
    if mad<EPS: mad=np.mean(np.abs(dev))/0.8
    if mad<EPS: 
        print ("fit already good enough")
        return (0,0,0)
    # Now the biweighted value:
    else:
        u   = ((dev)/(6.*mad))**2

        try:
            q=np.where(u < 1.0)[0]
            count=len(q)
        except:
            print ("something strange about the distribution, robustcheck failed")
            return (-1,0,0)
        if count < 3:
            print ('ROBUST_SIGMA: This distribution is TOO WEIRD! Returning -1')
            return (-1,0,0)
        else:
            n= np.nansum(dev)
            sig=n*np.sum((dev[q]**2) * ((1.0-u[q])**4))
            den=np.sum((1.0-u[q])*(1.0-5.0*u[q]))
            sig=np.sqrt(sig/(den*(den-1)))
            
#; If the standard deviation = 0 then we're done:
    if sig<EPS:  
        print ("standard deviation is tiny: fit is fine")
        return (0,sig,0)
    if closenough > 0:
        try:
            q=np.where(np.abs(yfit)>EPS)[0]
            if len(q) < 3:
                fracdev=0.0
            else:
                fracdev=np.median(np.abs(dev[q]/yfit[q]))
                if fracdev < closenough: 
                    print ("fit good enough")
                    return (0,sig,0)
        except:
            return (0,sig,0)

# Calculate the (bi)weights:
    b=np.abs(dev)/(6*sig)
    try:
        s=np.where(b>1.0)[0]
        if len(s)>0: b[s]=1
    except:
        s=[]
    ngood = len(y)-len(s)
    if ngood < 10:
        print ("too few good points left")
        return (2,sig,0)
    w=(1.0-b*b)
    w/=sum(w)
    return (1,sig,w)

def myrobustpolyfit(x,y,deg,weights):
    ############my own version of robustpolyfit.pro
    pars = np.polynomial.polynomial.polyfit(x,y,deg=deg,w=weights)
    try: 
        closenough = np.max(0.3*np.sqrt(0.5/(len(y)-1)),EPS)
    except:
        closenough = EPS
    (cf,sig,w)=checkfit(y, np.poly1d(np.array(pars[::-1]))(x), 
                        closenough)

    if cf ==1:
        diff= 1.0e10
        sig_1= min((100.*sig), EPS)
        nit = 0
        while (diff > closenough) and (nit<nitmax):
            print ("iteration ", nit)
            nit=nit+1
            sig2 = sig1
            sig1=sig

            g = np.where(w>0)[0]
            try:
                ng = len(g)
            except :
                ng=0
            if Ng < len(w):
              #;Throw out points with zero weight
                x = x[g]
                y = y[g]
                w = w[g]
            pars = np.polynomial.polynomial.polyfit(x,y,deg=deg,w=weights)
            (cf,sig,w)=checkfit(y, 
                                np.poly1d(np.array(pars[::-1]))(x), 
                                closenough)
            if cf == 0:
                nit=nitmax
                print ("fit converged!")
            if cf == 2:
                print ("too few good point for robust iterations")
                nit=nitmax
            diff = min((np.abs(sig1-sig)/sig),(abs(sig2-sig)/sig))
    if cf == 2:
        print ("too few good point for robust iterations")
    if cf ==0: 
        print ("fit converged!")

    return pars


def mypolynomial(x,pars):#pars1,pars2,pars3,pars4):
    y=0.0
#    pars=[pars1,pars2,pars3,pars4]
#    a=[0.1,10.0,1000.,10000]
    for i,c in enumerate(pars):
        y+=c*x**i
#    y=pars[0]+pars[1]*x+pars[2]*x*x+pars[3]*x*x*x
#        print "mypoly", y
    return y

def mytemplate(x,pars,b,template=None, sntype=None):
     if template==None:
          print ("no template povided")
          import globaltemplate
          template= globaltemplate.template#.template[b]['Ib']
          #     template=Mytempclass()
          #     template.loadtemplatefile()
     template.gettempfuncy(b)
     
#     print ((x-pars[1])*pars[0])
#     pl.figure(1000)
#     pl.ylim(21,17)
#     pl.plot((x-pars[1])*pars[0],pars[0]*(template.template[b].tempfuncy((x-pars[1])*pars[0]))+pars[2], 'ro')
#     pl.show()
#     sys.exit()
     #    print "templatefit:", pars[0]*(template.tempfuncy()(x-pars[1]))+pars[2]
 #    print "from fit function",x-pars[1], template.tempfuncy()(x-pars[1])
     if len(pars)==4:
          newfit= pars[0]*(template.template[b].tempfuncy((x-pars[1])*pars[3]))+pars[2]
          return newfit
     elif len(pars)==3:
          newfit= pars[0]*(template.template[b].tempfuncy((x-pars[1])*pars[0]))+pars[2]
          return newfit
     else:
          print ("ERROR: wrong number of parameters, you can pass 3 or 4")
          sys.exit(0)


def reversetemplate(x,pars,splfct, Vmax):
     return (splfct(((x+pars['xoffset']-Vmax)/pars['xstretch'])+Vmax)-pars['yoffset'])/pars['stretch']


def residuals(pars,x,y,e,functype,band,template=None):  # Residuals function needed by kmpfit
    pl.plot (x,y,'b.')
#    print "here ",pars
#    pars[0]/=1000.
#    pars[1]/=10.0
#    pl.plot(x,mypolynomial(x,pars),'r.')
    global counterhere
    counterhere=counterhere+1
    if functype=='poly':
        resids=(y-(mypolynomial(x,pars)))/e   
    elif functype=='template': 
        optimized=mytemplate(x,pars,band,template)         
        resids=(y-(optimized))/e
        nanindx=np.where(~np.isnan(resids))[0]
        resids=resids[nanindx]   
#        print resids
#        pl.ylim(18,13)#max(optimized),min(optimized))
        pl.plot(x,optimized, 'k--', alpha =0.03*counterhere) 
#        pl.plot(x,y, 'bo') 
#        pl.draw()
#        pl.savefig("blah%s_%04d.png"%(band,counterhere))
    else: 
        print ("function can only be 'template' or 'poly'")
        sys.exit()
#    pl.show()
#    print sum(resids**2)
    m = np.ma.masked_array(resids, np.isnan(resids))   
    return resids

def sumsqres(pars,x,y,e, functype,band,template=None):
    if functype=='template':
         res=residuals(pars,x,y,e,functype,band,template)
         if len(res)==0:
              res=np.array([10e9])
#         print sum(res**2)/len(res)
         return sum(res**2)/len(res)

    else:
         return sum(residuals(pars,x,y,e,functype,band)**2)
