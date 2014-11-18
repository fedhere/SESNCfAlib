import sys,os,glob, inspect
#,re,numpy,math,pyfits,glob,shutil,glob
import scipy as sp
import numpy as np
import pylab as pl
import itertools
import time
#from sort2vectors import sort2vectors
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#from matplotlib import  FontProperties
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]) + "/templates")
import numpy as np
import glob, pickle
import os,inspect,sys



cmd_folder =  os.path.realpath(os.environ['UTILPATH'])
print cmd_folder 
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

from myastrotools import absmag

from scipy.interpolate import interp1d,splrep,splev
#from matplotlib.pyplot import gca
from scipy.stats import nanmean,nanmedian

from utils import *
#from fitutils import *
from plotutils import *
#from templates import *
from scipy.stats import ks_2samp
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]) + "/templates")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

from templutils import *

from matplotlib.pyplot import (figure, axes, plot, xlabel, ylabel, title,
     grid, savefig, show)

coffset={'U':-3.3,'u':-3.3,'B':-2.3,'V':0,'R':1.8,'r':1.8,'I':3.5,'i':3.5,'J':8.5,'H':10.1,'K':10.5}

goodU=['sn2004aw','sn2005bf','sn2005hg','sn2006aj','sn2007gr','sn2009iz','sn2009jf']
goodIR=['sn2005bf','sn2005hg','sn2005kl','sn2005mf','sn2006fo','sn2006jc','sn2007c','sn2007gr','sn2007uy','sn2009er','sn2009iz','sn2009jf', 'sn2006aj','sn2008d']

survey_dates=(53232.00000,53597.00000,55058.000000, 56000.)
kp1={'CfA3':'CfA3-kep','CfA4':'kep1'}
survey=['fsho','mini',kp1,'CfA4-kep2']

class sntype:
    def __init__(self,sntype):
        self.su=setupvars()
        self.type=sntype
        self.count=0

        self.photometry={}
        for b in self.su.bands:
            self.photometry[b]={'mjd':np.zeros((0),float),'mag':np.zeros((0),float),'dmag':np.zeros((0),float), 'camsys':['']}

        self.colors={}
        for c in self.su.cs:
            self.colors[c]={'mjd':[],'mag':[],'dmag':[]}
#np.zeros((0),float),'mag':np.zeros((0),float),'dmag':np.zeros((0),float)}

        self.colormeans={}
        for c in self.su.cs:
            self.colormeans[c]={'epoch':[],'median':[],'std':[]}

        self.maxcol={}
        for c in self.su.cs:
            self.maxcol[c]={'mean':0,'median':0,'std':0, 'n':0}

        self.dist=-1

    def printtype(self):
        print "######### SN TYPE "+self.type+" ############"
        print "number of lcvs: %d"%self.count
        for c in self.su.cs.iterkeys():
            print c+" max color (median, std, n datapoints): ",self.maxcol[c]['median'],self.maxcol[c]['std'],self.maxcol[c]['n']

    def sncount(self,snlist):
        for sn in snlist:
            if sn.type == self.type: self.count+=1

    def addcolor(self, band, sn):
        #check type:
        if sn.type == self.type:
            self.colors[band]['mjd']=np.concatenate((np.array(self.colors[band]['mjd']),np.array(sn.colors[band]['mjd'])))
            self.colors[band]['mag']=self.colors[band]['mag']+sn.colors[band]['mag']
            self.colors[band]['dmag']=self.colors[band]['dmag']+sn.colors[band]['dmag']
        else:
            print "the supernova you passed is not the right type!"
            

    def plottype(self,photometry=False,band='',color=False,c='', fig=0, show=False, verbose=False, save=False, alpha=1.0):
        print "######## PLOTTING: ",self.type, " including ", self.count, "sn #############"
    
        if photometry:
            print "not implemented yet"
            return(-1)

        if color:
            if photometry:
                print "need new fig number"
            myfig = pl.figure(fig)
            ax=myfig.add_subplot(1,1,1)
            legends=[]        
            notused=[]
            if c== '': mybands=[k for k in self.su.cs.iterkeys()]
            else: mybands=[c]
            myylim=(0,0)
            myxlim=(-10,10)

            for b in mybands:
                if len(self.colors[b]['mjd'])==0:
                    if verbose:
                        print "nothing to plot for ",b
                        notused.append(b)
                    continue
                if verbose:
                    print "plotting band ",b," for ", self.name
                    print self.colors[b]['mjd'],self.colors[b]['mag']

                m1yxlim=(float(min(myxlim[0],min(self.colors[b]['mjd'])-10)),
                        float(max(myxlim[1],max(self.colors[b]['mjd'])+10)))
                myylim=(float(min(myylim[0],min(self.colors[b]['mag'])-0.5)),
                        float( max(myylim[1],max(self.colors[b]['mag'])+0.5)))

                myplot_setlabel(xlabel='JD - 2453000.00',ylabel='color',title=self.type)
                legends.append(myplot_err(self.colors[b]['mjd'],
                                          self.colors[b]['mag'],
                                          yerr=self.colors[b]['dmag'],
                                          xlim=myxlim,ylim=myylim, symbol='%so'%self.su.mycolors[b[0]], alpha=alpha))#                           

            loc=1
            pl.legend(legends,mybands, loc=loc, ncol=1,prop={'size':8}, numpoints = 1)                
            for i in notused: mybands.remove(i)
            if save:
                pl.savefig(self.type+".color_"+''.join(mybands)+'.png',bbox_inches='tight')
        if show:
            pl.show()

        return myfig

class snstats:
    def __init__(self):
        try:
            os.environ['SESNPATH']
        except KeyError:
            print "must set environmental variable SESNPATH"
            sys.exit()
        self.band=''
        self.maxjd  =[0.0,0.0]
        self.m15data=[0.0,0.0]
        self.dm15   =0.0
        self.dm15lin=[0.0,0.0]
        self.Rdm15   =0.0
        self.Rdm15lin=[0.0,0.0]
        self.polydeg=0.0
        self.polyrchisq =0.0
        self.polyresid=None
        self.templrchisq=0.0
        self.templresid=None
        self.tlim=[0.0,0.0]
        self.maglim=[0.0,0.0] 
        self.templatefit={'stretch':1.0, 'xoffset':0.0, 'xstretch':1.0, 'yoffset':0.0}
        self.stretch=0.0
        self.norm=0.0

        self.flagmissmax=0
        self.flagmiss15=0
        self.flagbadfit=0
        self.success=0

    def printstats(self):
        print "############## sn statistics: ###########"
        print  "maxjd ", self.maxjd[:] ,"band ",self.band
        print  "m15   ", self.m15data[:] ,  "band ",self.band
        print  "dm15  ", self.dm15      ,"band ",self.band
        print  "dm15 l ", self.dm15lin  ,   "band ",self.band
        print  "deg    ", self.polydeg   
        
        print  "poly  chisq  ", self.polyrchisq
        if not self.polyresid == None:
            print  "poly  resids ", sum((self.polyresid)*(self.polyresid))
        print  "templ chisq  ", self.templrchisq
        if not self.templresid == None:
            print  "templ resids ", sum((self.templresid)*(self.templresid))
        print "########################\n\n"
        

class mysn:
    def __init__(self, name, verbose=False):
        if '/' in name:
            self.name=name.split('/')[-1].split('.')
            for s in self.name:
		if 'sn9' in s:
                    self.name = s.replace('sn','sn19')
		elif 'sn0' in s:
                    self.name = s.replace('sn','sn20')
		elif 'sn1' in s:
                    self.name = s.replace('sn','sn20')
        else:
            self.name=name
        self.snnameshort=self.name.replace('sn10','').replace('sn20','').strip()
        if len(self.snnameshort) == 3:
             self.snnameshort=self.snnameshort.upper()
        print self.snnameshort
          
        try:
            if verbose:
                print "SESNPATH environmental variable: ",os.environ['SESNPATH'],self.name
                print "file:",os.environ['SESNPATH']+"/nirphot/PAIRITEL_Ibc/Ibc/lcs/mag/"+self.name+'_*'
            if self.name[-2].isdigit():
                self.fnir = glob.glob(os.environ['SESNPATH']+"/nirphot/PAIRITEL_Ibc/Ibc/lcs/mag/"+self.name[:-1]+self.name[-1].upper()+'_*')[0]
                if verbose:
                    print self.fnir
            else:
                self.fnir = glob.glob(os.environ['SESNPATH']+"/nirphot/PAIRITEL_Ibc/Ibc/lcs/mag/"+self.name+'_*')[0]
            if verbose:
                print self.fnir
                print "NIR filename:",os.environ['SESNPATH']+"/nirphot/PAIRITEL_Ibc/Ibc/lcs/mag/"+self.name+'_*'
        except IndexError:
            self.fnir = None
            pass
            
        self.su=setupvars()
        self.nomaxdate=False
        self.type=''
        self.camsystem=''
        self.pipeline=''
        self.nir=False
        self.n=0
        self.Vmax=0.0        
        self.dVmax=0.0        
        self.Vmaxmag=0.0
        self.filters={}
        for b in self.su.bands:
            self.filters[b]=0
        self.polysol={}
        self.snspline={}
        self.templsol={}
        self.solution={}
        self.photometry={}
        self.stats={}
        self.colors={}
        self.maxcolors={}
        self.maxmags={}
        self.flagmissmax=True
        self.lc={}
        self.ebmvtot=0.0
        for b in self.su.bands:
            self.photometry[b]={'mjd':np.zeros((0),float),'mag':np.zeros((0),float),'dmag':np.zeros((0),float),'extmag':np.zeros((0),float), 'camsys':[''], 'natmag':np.zeros((0),float)}
            self.stats[b]=snstats()
            self.polysol[b]=None
            self.snspline[b]=None
            self.templsol[b]=None            
            self.solution[b]={'sol':None,'deg':None, 'pars':None, 'resid':None}
            self.maxmags[b]={'epoch':0.0,'mag':float('NaN'), 'dmag':float('NaN')}

        for c in self.su.cs:
            self.maxcolors[c]={'epoch':0.0,'color':float('NaN'), 'dcolor':float('NaN')}
            self.colors[c]={'mjd':[],'mag':[],'dmag':[]}#np.zeros((0),float),'mag':np.zeros((0),float),'dmag':np.zeros((0),float)}
                            
        self.polyfit= None
        self.metadata={}
#        try:
#        except :
#            pass
        self.Rmax={}
        self.dr15=0.0

    def setVmax(self,loose=True,earliest=False, verbose=True, D11=False):
         if verbose: print "Vmax: ", self.Vmax, self.flagmissmax         
         try:
              self.Vmax=float(self.metadata['CfA VJD bootstrap'])
              if self.Vmax<2400000:
                   self.Vmax=self.Vmax+2400000.5
              self.dVmax=float(self.metadata['CfA VJD bootstrap error'])
         except:
              try:
                   self.Vmax=float(self.metadata['MaxVJD'])
                   self.dVmax=1.5
              except:
                   if D11:
                        if verbose:  print "trying D11 V max"
                        try:
                             self.Vmax=float(self.metadata['D11Vmaxdate'])
                             self.dVmax=float(self.metadata['D11Vmaxdateerr'])
                        except:               
                             self.Vmag=None
                             self.flagmissmax=True  
                   if not self.Vmax:
                         if not loose:
                             self.Vmag=None
                             self.flagmissmax=True
                         else:    
                              if verbose:  print "trying with other color max's"
                              try:
                                   if verbose: print "Rmax: ", self.metadata['CfA RJD bootstrap']
                                   Rmax=float(self.metadata['CfA RJD bootstrap'])
                                   dRmax=float(self.metadata['CfA RJD error'])
                                   if verbose: print "here Rmax", Rmax,dRmax
                                   Rmaxflag=True
                              except:
                                   Rmaxflag=False
                                   pass
                              try:
                                   if verbose: print "Bmax: ", self.metadata['CfA BJD bootstrap']
                                   Bmax=float(self.metadata['CfA BJD bootstrap'])
                                   dBmax=float(self.metadata['CfA BJD error'])
                                   if verbose: print "here Bmax", Bmax,dBmax
                                   Bmaxflag=True
                              except:
                                   Bmaxflag=False
                                   pass
                              try:
                                   if verbose: print "Imax: ", self.metadata['CfA IJD bootstrap']
                                   Imax=float(self.metadata['CfA IJD bootstrap'])
                                   dImax=float(self.metadata['CfA IJD error'])
                                   if verbose: print "here Imax", Imax  ,dImax                  
                                   Imaxflag=True
                              except:
                                   Imaxflag=False
                                   pass

                              if verbose: print "Rmaxflag: ",Rmaxflag
                              if verbose: print "Bmaxflag: ",Bmaxflag
                              if verbose: print "Imaxflag: ",Imaxflag
                              if  Rmaxflag+Bmaxflag+Imaxflag>=2:
                                   if Bmaxflag and Rmaxflag:
                                        self.Vmax=np.mean([Rmax-1.5+2400000.5,Bmax+2.3+2400000.5])
                                        self.dVmax=np.sqrt(sum([dRmax**2,dBmax**2,1.3**2,1.3**2]))
                                   elif Bmaxflag and Imaxflag:
                                        self.Vmax=np.mean([Imax-3.1+2400000.5,Bmax+2.3+2400000.5])
                                        self.dVmax=np.sqrt(sum([dImax**2,dBmax**2,1.3**2,1.5**2]))
                                   elif Rmaxflag and Imaxflag:
                                        self.Vmax=np.mean([Imax-3.1+2400000.5,Rmax-1.5+2400000.5])
                                        self.dVmax=np.sqrt(sum([dRmax**2,dImax**2,1.3**2,1.5**2]))
                                   self.flagmissmax=False
                              elif Rmaxflag+Bmaxflag+Imaxflag>=1 and loose:
                                   if Imaxflag:
                                        self.Vmax=Imax-3.1+2400000.5
                                        self.dVmax=np.sqrt(sum([dImax**2,1.5**2]))
                                        if verbose: print self.Vmax,Imax,self.dVmax
                                   if Rmaxflag:
                                        self.Vmax=Rmax-1.5+2400000.5
                                        self.dVmax=np.sqrt(sum([dRmax**2,1.3**2]))
                                        if verbose: print self.Vmax,Rmax,self.dVmax
                                   if Bmaxflag:
                                        self.Vmax=Bmax+2.3+2400000.5
                                        self.dVmax=np.sqrt(sum([dBmax**2,1.3**2]))
                                        if verbose: print self.Vmax,Bmax,self.dVmax
                                   self.flagmissmax=False
                              else:
                                   if earliest:
                                        self.Vmax=earliestv
                                        self.flagmissmax=False
                                   else:
                                        self.Vmag=None
                                        self.flagmissmax=True



         if verbose: print "Vmax: ", self.Vmax, self.flagmissmax


    def readinfofileall(self,verbose=False, earliest=False, loose=False, D11=False):
        import csv, os, sys
        if verbose: print "environmental variable for lib:",os.getenv("SESNCFAlib")
        if os.getenv("SESNCFAlib")=='':
             print "must set environmental variable SESNCFAlib"
             sys.exit()
        input_file = csv.DictReader(open(os.getenv("SESNCFAlib")+"/CfA.SNIbc.BIGINFO.csv"))
        for row in input_file:
            if row['SNname'].lower().strip()  ==  self.name.lower().strip():
               for k in row.keys():
                    if verbose: 
                         print k
                    self.metadata[k]=row[k]
                    if verbose: 
                         print self.metadata[k]
        self.Vmaxflag=False
        self.setVmax(loose=loose, verbose=verbose, D11=D11)
        self.type=self.metadata['Type']

    def setsn(self,sntype,Vmax,ndata=None,filters=None, camsystem=None,pipeline=None):
        self.type=sntype
        self.n=ndata
        try:
            self.Vmax=float(Vmax)
            self.flagmissmax=False
        except:
            self.Vmax=Vmax
        if camsystem:
            self.camcode=camsystem
        if pipeline:
            self.pipeline=pipeline

    def setphot(self):
        uniqpc= set(self.lc['photcode'])
        for b in self.filters.iterkeys():
            for i in uniqpc:
                if i == self.su.photcodes[b][0] or i == self.su.photcodes[b][1]  :
                    n=sum(self.lc['photcode']==i)
                    self.filters[b]=n
                    self.photometry[b]={'mjd':np.zeros(n,float),'mag':np.zeros(n,float),'dmag':np.zeros(n,float), 'camsys':['S4']*n}

    def setsnabsR(self):
        print "Rmax1",self.Rmax
        from cosmdist import cosmo_dist
        if is_empty(self.metadata):
            print "reading info file"
            self.readinfofileall(verbose=False, earliest=False, loose=True)
            print "done reading"
        print self.filters['r'],self.filters['R']
        if self.filters['r']==0 and self.filters['R']>0:
            if not self.getmagmax('R') == -1:
                 self.Rmax['mjd'],self.Rmax['mag'],self.Rmax['dmag'] =self.maxmags['R']['epoch'],self.maxmags['R']['mag'],self.maxmags['R']['dmag']
                 r15=self.getepochmags('R',epoch=(self.Rmax['mjd']+15.0))
                 self.Rmax['dm15']=self.Rmax['mag']-r15[1]
                 self.Rmax['ddm15']=np.sqrt(self.Rmax['dmag']**2+r15[2]**2)

        else:
            if not self.getmagmax('r') == -1:
                 self.Rmax['mjd'],self.Rmax['mag'],self.Rmax['dmag'] =self.maxmags['r']['epoch'],self.maxmags['r']['mag'],self.maxmags['r']['dmag']
                 imag=self.getepochmags('i', epoch=self.Rmax['mjd'])
                 self.Rmax['mag']= self.Rmax['mag'] - 0.2936*(self.Rmax['mag'] - imag[1]) - 0.1439
                 self.Rmax['dmag']= np.sqrt((self.Rmax['dmag'])**2)# + imag[2]**2)
                 r15=self.getepochmags('r',epoch=(self.Rmax['mjd']+15.0))
                 i15=self.getepochmags('i',epoch=(self.Rmax['mjd']+15.0))
                 self.Rmax['dm15']=self.Rmax['mag']- (r15[1] )#- 0.2936*(self.Rmax['mag'] - i15[1]) - 0.1439)
                 self.Rmax['ddm15']=np.sqrt(self.Rmax['dmag']**2)#+(r15[2]**2+i15[2]**2))
        print "Rmax",self.Rmax
        if not is_empty(self.Rmax):
             self.dist=cosmo_dist([0],[float(self.metadata['z'])],lum=1,Mpc=1)[0]
             if self.dist==-1:
                  self.dist=float(self.metadata['distance Mpc'])
             self.Rmax['absmag']=absmag(self.Rmax['mag'],self.dist,dunits='Mpc')
#float(self.metadata['distance Mpc'])
        for k in self.Rmax.keys():
            print "here",k,self.Rmax[k]
#       pl.show()
        
            

    def printsn(self,template=False,printlc=False,photometry=False,color=False, extended=False, band=None, cband=None,fout=None, nat=False):
        print "\n\n\n##############  THIS SUPERNOVA IS: ###############\n"
        print "name: ",self.name
        print "type: ",self.type
        print "Vmax date: %.3f"%self.Vmax
        print "Vmax  mag: %.2f"%self.Vmaxmag
        print "filters: ",self.filters
        try:Vmax=float(self.Vmax)
        except:Vmax=0.0
        if Vmax>2400000.5: Vmax-=2400000.5
        if band:
            bands=[band]
        else:
            bands=self.su.bands
        if cband:
            cbands=[cband]
        else:
            cbands=self.su.cs
        if printlc:
            print "all lightcurve: ", self.lc
        if fout:
            f= open(fout,'w')
        if photometry:            
            print "##############  photometry by band: ###############"
            print bands
            for b in bands:
                if self.filters[b] == 0:
                    continue
                if fout == None:
                    print "#band ",b, "mjd\t \tphase\t \tmag \tdmag"
                    for i in range(self.filters[b]):
                        print "\t%.3f"%self.photometry[b]['mjd'][i],
                        print "\t%.3f\t"%(self.photometry[b]['mjd'][i]-Vmax),
                        print "\t%.2f"%self.photometry[b]['mag'][i],
                        print "\t%.2f"%self.photometry[b]['dmag'][i],
                        if nat:
                            print "\t%s"%self.photometry[b]['camsys'][i]
                        else:
                            print ""
                else:
                    print >>f, "#band ",b, "mjd\t \tphase\t \tmag \tdmag"                
                    for i in range(self.filters[b]):
                        print >>f,"\t%.3f"%self.photometry[b]['mjd'][i],
                        print >>f,"\t%.3f\t"%(self.photometry[b]['mjd'][i]-Vmax),
                        print >>f,"\t%.2f"%self.photometry[b]['mag'][i],
                        print >>f,"\t%.2f"%self.photometry[b]['dmag'][i],
                        if nat:
                            print >>f,"\t%s"%self.photometry[b]['camsys'][i]
                        else:
                            print >>f,""
                    
        if color:
            for c in cbands:
                print "\n\n\n",c
                print "\n\n\ncolors : ",self.colors[c]

                if len(self.colors[c]['mjd']) == 0:
                    continue
                if fout == None:
                    print "#band ",c, "mjd\t \tphase\t \tcolor \tdmag"
                    for i in range(len(self.colors[c]['mjd'])):
                        print "\t%.3f\t"%(self.colors[c]['mjd'][i]+Vmax),
                        print "\t%.3f"%self.colors[c]['mjd'][i],
                        print "\t%.2f"%self.colors[c]['mag'][i],
                        print "\t%.2f"%self.colors[c]['dmag'][i]
                else:
                    print >>f,"#band ",c, "mjd\t \tphase\t \tcolor \tdmag"
                    for i in range(len(self.colors[c]['mjd'])):
                        print >>f,"\t%.3f\t"%(self.colors[c]['mjd'][i]+Vmax),
                        print >>f,"\t%.3f"%self.colors[c]['mjd'][i],
                        print >>f,"\t%.2f"%self.colors[c]['mag'][i],
                        print >>f,"\t%.2f"%self.colors[c]['dmag'][i]

        if template:
            for b in self.su.bands:
                print b," band: "
                print "  stretch:  ", self.stats[b].templatefit['stretch']
                print "  x-stretch:", self.stats[b].templatefit['xstretch']
                print "  x-offset: ", self.stats[b].templatefit['xoffset']
                print "  y-offset: ", self.stats[b].templatefit['yoffset']

        if extended:
            for b in self.su.bands:
                if self.filters[b] == 0:
                    continue
                print b," band: "
                self.stats[b].printstats() 
        print "\n##################################################\n\n\n"


    def printsn_fitstable(self, fout=None):
        import pyfits as pf
        print "\n\n\n##############  THIS SUPERNOVA IS: ###############\n"
        print "name: ",self.name
        print "type: ",self.type
        print "Vmax date: %.3f"%self.Vmax
        print "Vmax  mag: %.2f"%self.Vmaxmag
        print "filters: ",self.filters
        bands=self.su.bands
        allcamsys=[]
        for b in bands:
            allcamsys+=self.photometry[b]['camsys']
        allcamsys = [a for a in set(allcamsys) if not a == '']
        fitsfmt={}
        if not fout:
            fout=self.name+".fits"
        col=[\
             pf.Column(name='SNname',         format='8A',  unit='none', array=[self.name]),\
             pf.Column(name='SNtype',         format='10A', unit='none', array=[self.type]),\
             pf.Column(name='Vmaxdate',       format='D',   unit='MJD',  array=[self.Vmax]),\
             pf.Column(name='Vmax',           format='D',   unit='mag',  array=[self.Vmaxmag]),\
             pf.Column(name='pipeversion',format='10A', unit='none', array=allcamsys),\
        ]
        for b in bands:
            if b == 'i':
                bb='ip'
            elif b == 'u':
                bb='up'
            elif b == 'r':
                bb='rp'
            else:
                bb=b
            if self.filters[b] == 0:
                continue
#            fitsfmt[b]=str(self.filters[b])+'D'
            fitsfmt[b]='D'
            col=col+[\
                     pf.Column(name=bb+'pipeversion',format='10A', unit='none', array=[a for a in set(self.photometry[b]['camsys'])]),\

                     pf.Column(name=bb+'epochs', format=fitsfmt[b],unit='MJD',array=self.photometry[b]['mjd']),\
                     pf.Column(name=bb,          format=fitsfmt[b],unit='mag',array=self.photometry[b]['mag']),\
                     pf.Column(name='d'+bb,      format=fitsfmt[b],unit='mag',array=self.photometry[b]['dmag']),\
                     pf.Column(name=bb+'_nat',      format=fitsfmt[b],unit='mag',array=self.photometry[b]['natmag']),\
             ]
        '''
        col=col+[pf.Column(name='bands',          format='2A',  unit='none', array=[b for b in bands if self.filters[b] > 0])]
        '''
        # create headers
        table_hdu      = pf.new_table(col)
        table_hdu.name = "TDC Challenge Light Curves"
        phdu           = pf.PrimaryHDU()
        hdulist        = pf.HDUList([phdu, table_hdu])

        # write to file
        hdulist.writeto(fout, clobber=True)

                    
        print "\n##################################################\n\n\n"


    def printsn_textable(self,template=False,printlc=False,photometry=False,color=False, extended=False, band=None, cband=None,fout=None):
        print "#name: ",self.name
        print "#type: ",self.type
        print "#Vmax date: %.3f"%self.Vmax
#        print "Vmax  mag: %.2f"%self.Vmaxmag
        print "#filters: ",self.filters
        bands=self.su.bands
        if fout:
            fout=fout.replace('.tex','opt.tex')
            print fout
            fo= open(fout,'w')
            fout=fout.replace('opt.tex','nir.tex')
            print fout
            fir= open(fout,'w')
            print fo,fir
        import operator
        print "\n################################################\n"
        
        maxn=max(self.filters.iteritems(), key=operator.itemgetter(1))[0]
        maxn=self.filters[maxn]
        if self.filters['u']==0:
            del self.filters['u']
            myu='U'
        elif self.filters['U']==0:
            del self.filters['U']            
            myu='u\''
        if self.filters['r']==0:
            del self.filters['r']
            myr='R'
        elif self.filters['R']==0:
            del self.filters['R']            
            myr='r\''
        if self.filters['i']==0:
            del self.filters['i']
            myi='I'
        elif self.filters['I']==0:
            del self.filters['I']            
            myi='i\''
        if not fout == None:
            print >>fo,'''\\begin{deluxetable*}{ccccccccccccccc}
\\tablecolumns{15}
\\singlespace
\\setlength{\\tabcolsep}{0.0001in}
\\tablewidth{514.88pt}
\\tablewidth{0pc}
\\tabletypesize{\\scriptsize}
\\tablecaption{\\protect{\\mathrm{'''+self.name.replace('sn','SN~')+'''}} Optical Photometry}'''

            print >>fir,'''\\begin{deluxetable*}{ccccccccc}
\\tablecolumns{9}
\\singlespace
\\setlength{\\tabcolsep}{0.0001in}
\\tablewidth{514.88pt}
\\tablewidth{0pc}
\\tabletypesize{\\scriptsize}
\\tablecaption{\\protect{\\mathrm{'''+self.name.replace('sn','SN~')+'''}} NIR Photometry}'''
        
        if fout == None:
            print "mjdtU\tdU\tmjd\tB\tdB\tmjd\tV\tdV\tmjd\t"+myr+"\td"+myr+"\tmjd\t"+myi+"\td"+myi+"\tmjd\tH\tdH\tmjd\tJ\tdJ\tmjd\tK_s\tdK_s"
        else:
            f=fo
            print >>f, "\\tablehead{\\colhead{MJD}&"
            print >>f, "\\colhead{$"+myu+"$}&"
            print >>f, "\\colhead{d$"+myu+"$}&"

            print >>f, "\\colhead{MJD}&"
            print >>f, "\\colhead{$B$}&"
            print >>f, "\\colhead{d$B$}&"

            print >>f, "\\colhead{MJD}&"
            print >>f, "\\colhead{$V$}&"
            print >>f, "\\colhead{d$V$}&"

            print >>f, "\\colhead{MJD}&"
            print >>f, "\\colhead{$"+myr+"$}&"
            print >>f, "\\colhead{d$"+myr+"$}&"

            print >>f, "\\colhead{MJD}&"
            print >>f, "\\colhead{$"+myi+"$}&"
            print >>f, "\\colhead{d$"+myi+"$}}"
            print >>f, "\\startdata" 
            f=fir
            print >>f, "\\tablehead{\\colhead{MJD}&"
            print >>f, "\\colhead{$H$}&"
            print >>f, "\\colhead{d$H$}&"

            print >>f, "\\colhead{MJD}&"
            print >>f, "\\colhead{$J$}&"
            print >>f, "\\colhead{d$J$}&"

            print >>f, "\\colhead{MJD}&"
            print >>f, "\\colhead{$K_s$}&"
            print >>f, "\\colhead{d$K_s$}}"
            print >>f, "\\startdata" 

        if fout:
             f=fo
        for i in range(maxn):
            for b in [myu[0],'V','B',myr[0],myi[0]]:
                if i < len(self.photometry[b]['mjd']):
                    if fout == None:
                        print "%.3f\t"%self.photometry[b]['mjd'][i], 
                        print "%.2f\t"%self.photometry[b]['mag'][i],
                        print "%.2f\t"%self.photometry[b]['dmag'][i],
                    else:
                        print >>f,"%.3f &"%self.photometry[b]['mjd'][i],
                        print >>f,"%.2f &"%self.photometry[b]['mag'][i],
                        if myi[0]  in b:
                            print >>f,"%.2f\\\\ "%self.photometry[b]['dmag'][i],
                        else:
                            print >>f,"%.2f & "%self.photometry[b]['dmag'][i],
                        
                else:
                    if fout == None:
                        print "-\t","-\t","-\t",
                    else:
                        if  b.startswith(myi[0]):
                            print >>f,"-&","-&","-\\\\",
                        else:
                            print >>f,"-&","-&","-&",

            if fout == None:
                print ""
            else:
                print >>f,""
        if fout:
            print >>f,'''\\enddata
\\label{tab:snoptphot}
\\end{deluxetable*}'''                

        if fout :f=fir
        for i in range(maxn):
            for b in ['H','J','K']:

                if i < len(self.photometry[b]['mjd']):
                    if fout == None:
                        print "%.3f\t"%self.photometry[b]['mjd'][i], 
                        print "%.2f\t"%self.photometry[b]['mag'][i],
                        print "%.2f\t"%self.photometry[b]['dmag'][i],
                    else:
                        print >>f,"%.3f &"%self.photometry[b]['mjd'][i],
                        print >>f,"%.2f &"%self.photometry[b]['mag'][i],
                        if 'K' in b:
                            print >>f,"%.2f\\\\ "%self.photometry[b]['dmag'][i],
                        else:
                            print >>f,"%.2f & "%self.photometry[b]['dmag'][i],
                        
                else:
                    if fout == None:
                        print "-\t","-\t","-\t",
                    else:
                        if  'K' in b:
                            print >>f,"-&","-&","-\\\\",
                        else:
                            print >>f,"-&","-&","-&",

            if fout == None:
                print ""
            else:
                print >>f,""
        if fout:
            print >>f,'''\\enddata
\\label{tab:snnirphot}
\\end{deluxetable*}'''                



    def colorcolorplot(self, band1='B-V', band2='r-i', fig=float('NaN'),legends=[], label='', labsize=24, plotred=True):
        #b-v vs v-r
        if len(legends)==0:
            legends=[]
        if np.isnan(fig):
            fig=1000
            myfig=pl.figure()
        else:
            myfig=pl.figure(fig)
        print self.maxcolors[band1]['color'],self.maxcolors[band2]['color']

        ax=myfig.add_subplot(1,1,1)

        print "color-color",self.name,self.type,self.Vmax, band1,self.maxcolors[band1]['color'],band2,self.maxcolors[band2]['color'],self.maxcolors[band1]['dcolor'],band2,self.maxcolors[band2]['dcolor'], 
#        print self.su.mysymbols[self.type]
        typekey=self.type
        print typekey, self.su.mytypecolors.keys()
        if typekey not in self.su.mytypecolors.keys():
            typekey='other'
            legends=  myplot_err(self.maxcolors[band1]['color'],self.maxcolors[band2]['color'],
                             yerr=self.maxcolors[band1]['dcolor'],
                             xerr=self.maxcolors[band2]['dcolor'],
                             symbol=self.su.mytypecolors[typekey]+self.su.mysymbols[typekey],
                                 alpha=1, offset=0, fig = fig, fcolor=self.su.mytypecolors[typekey], ms=15, markeredgewidth=2)        
        else:
            legends=  myplot_err(self.maxcolors[band1]['color'],self.maxcolors[band2]['color'],
                             yerr=self.maxcolors[band1]['dcolor'],
                             xerr=self.maxcolors[band2]['dcolor'],
                             symbol=self.su.mytypecolors[typekey]+self.su.mysymbols[typekey],
                                 alpha=0.5, offset=0, fig = fig, fcolor=self.su.mytypecolors[typekey], ms=15)

        ax.annotate("", xy=(2.6, 2.6), xycoords='data',
                     xytext=(2.3,2.3 ), textcoords='data',ha='center',va='center',
                     arrowprops=dict(arrowstyle="->",color='#b20000'),)
        myplot_setlabel(xlabel=band1,ylabel=band2,title=None, label=label, xy=(0.75,0.8), labsize=labsize)
        if plotred:
            pl.figtext(0.88,0.9,"red", fontsize=labsize)

#pl.plot(self.maxcolors[band1]['color'],self.maxcolors[band2]['color'],c=self.su.mytypecolors[typekey],marker=self.su.mysymbols[typekey], markersize=8, alpha=0.5)
        return(fig, legends,(self.maxcolors[band1]['color'],self.maxcolors[band2]['color']))

    def plotsn(self,photometry=False,band='',color=False,c='', fig=None, show=False, verbose=False, save=False, savepng=False, symbol='',title='',Vmax=None, plottemplate=False, plotpoly=False, plotspline=False, relim=True, xlim=None, ylim=None, offsets=False, ylabel='Mag',aspect=1, nir=False,allbands=True, fcolor=None, legendloc=1, nbins=None, singleplot=False, noylabel=False):

        from pylab import rc
        rc('axes', linewidth=2)
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 24
        mpl.rcParams['font.family'] = 'Times New Roman'
        #    mpl.rcParams['font.serif'] = 'Times'                                                                                                       
        mpl.rcParams['axes.labelsize'] = 25
        mpl.rcParams['xtick.labelsize'] = 25.
        mpl.rcParams['ytick.labelsize'] = 25.
        if save:
            mpl.rcParams['ytick.major.pad']='6'
        offset=0.0
        boffsets={'U':2,'u':2,'B':1,'V':0,'R':-1,'I':-2,'r':-1,'i':-2, 'J':-3,'H':-4,'K':-5}
        print "\n##############  PLOTTING SUPERNOVA : ",self.name,"###############"
	myfig=None
        if fig==None:
            fig=0
        if photometry:
            print "plotting...",
            myfig = pl.figure(fig)#, figsize=(30,60))
            ax=myfig.add_subplot(1,1,1)
                
            ax.minorticks_on()
            majorFormatter = FormatStrFormatter('%d')
            minorLocator   = MultipleLocator(0.2)
#            majorLocator   = MultipleLocator()
#            ax.yaxis.set_minor_locator(minorLocator)
            ax.yaxis.set_major_formatter(majorFormatter)


            adjustFigAspect(myfig,aspect=aspect)

            legends=[]        
            notused=[]
            if band== '': 
                if allbands:  mybands=self.su.bands
                else: mybands=self.su.bandsnonir               
            
            else: mybands=[band]
            
#            if self.stats[mybands[0]].maxjd[1]==0.0:
#                self.getstats(mybands[0])
            if xlim:
                myxlim = (xlim[0]-53000,xlim[1]-53000)
                
            elif not relim:
                xlim = pl.xlim()
            else:
                if not self.stats[mybands[0]].maxjd == [0.0,0.0]:
                    xlim=[float(self.stats[mybands[0]].maxjd[0])-53010,float(self.stats[mybands[0]].maxjd[0])+10-53000]
                else:
                    xlim=[self.Vmax-2453010.5,self.Vmax-2453000+10]
                    #print xlim
            if ylim: 
                myylim = ylim    
                if verbose: print "ylim ",myylim
            elif not self.stats[mybands[0]].maxjd[0]==0.0:
                myylim=(0,0)
            elif self.Vmax:
                if self.Vmax > 10e5:
                    myxlim=(float(self.Vmax)-10-2453000,
                            float(self.Vmax)+10-2453000)
                else:
                    myxlim=(float(self.Vmax)-10-53000,
                            float(self.Vmax)+10-53000)
                myylim=(0,0)
            else:
                bandind=0
                while len(self.photometry[mybands[bandind]]['mjd'])==0:
                    bandind+=1
                    if bandind>len(mybands):
                        pass
#                print mybands[bandind],self.photometry[mybands[bandind]]['mjd']
                myxlim = (min(self.photometry[mybands[bandind]]['mjd'])-20-53000,
                          max(self.photometry[mybands[bandind]]['mjd'])+20-53000)
                myylim=(0,0)

            if int(str(int(myxlim[1]))[-1])<5:
                myxlim=(myxlim[0],myxlim[1]+5)

#            majorLocator   = MultipleLocator(int((myxlim[1]-myxlim[0])/6))
            majorFormatter = FormatStrFormatter('%d')
            bandswdata=[]
            #                if title=='':
            #                    title =self.name 
            ylabel=ylabel.replace('+0','')
            if self.name[-2].isdigit():                
                label=self.name.upper()
            else:
                label=self.name
            label = label.replace('sn','SN ') 

            ax.locator_params(tight=True, nbins=4)
            majorLocator   = MultipleLocator(5)
            try:
                 if (ylim[0]-ylim[1])<10:
                      majorLocator   = MultipleLocator(2)
            except:
                 pass
            ax.yaxis.set_major_locator(majorLocator)


            if noylabel:
                myplot_setlabel(xlabel='JD - 2453000.00',title=title, label=label, ax=ax, ylabel="  ", rightticks=True, labsize=21)
            else:
                myplot_setlabel(xlabel='JD - 2453000.00',ylabel=ylabel,title=title, label=label, ax=ax, labsize=21)


            for b in mybands:
                if offsets:offset = boffsets[b]
                else: offset=0.0
                if self.filters[b]==0:
                   if verbose:
                        print "nothing to plot for ",b
                        notused.append(b)
                   continue
                bandswdata.append(b.replace('r','r\'').replace('i','i\''))

                if verbose: print "plotting band ",b," for ", self.name                
                if not relim or ylim:
                    ylim=pl.ylim()
                if relim:
                    xlim=[min(myxlim[0],min(self.photometry[b]['mjd'])-10-53000),max(myxlim[1],max(self.photometry[b]['mjd'])+10-53000)]
                    myxlim=xlim
                    if myylim == (0,0):
                            ylim=(20,0)
                            myylim=(max(ylim[1],max(self.photometry[b]['mag'])+1+offset),
                                    min(ylim[0],min(self.photometry[b]['mag'])-1+offset))
                            if verbose: print "this is the new myylim",myylim
                    else:
                            myylim=(max(ylim[1],(max(myylim[0],max(self.photometry[b]['mag'])+1+offset))),
                                    min(ylim[0],min(myylim[1],min(self.photometry[b]['mag'])-1+offset)))
                if 'J' in b or 'H' in b or 'K' in b:
                    fcolor='None'
                else:
                    fcolor=self.su.mycolors[b]
                if symbol=='':
                    symbol='%s%s'%(self.su.mycolors[b],self.su.myshapes[b])
                l,= myplot_err(np.asarray(self.photometry[b]['mjd'])-53000.0,
                               self.photometry[b]['mag'],
                               yerr=self.photometry[b]['dmag'],
                               xlim=myxlim,
                               ylim=myylim, symbol=symbol,offset=offset, fcolor=fcolor)

                legends.append(l)
                pl.legend(legends,bandswdata, loc=legendloc, ncol=1,prop={'size':12}, numpoints = 1)

                symbol=''
                if plotpoly or plottemplate:
                    if self.Vmax:
                        fullxrange=np.arange(float(self.Vmax)-2453000.0-10.,float(self.Vmax)-2453000.0+40.0,0.1)
                    else:
                        try:
                            fullxrange=np.arange(self.stats['V'].maxjd[0]-10.,self.stats['V'].maxjd[0]+40.,0.1)
                        except:
                            continue


                    if plotpoly and self.solution[b]['sol']:
                        myplot_err(fullxrange,self.solution[b]['sol'](fullxrange),symbol='%s-'%self.su.mycolors[b], offset=offset)
                    if plottemplate and not self.stats[b].templrchisq == 0:
                        myplot_err(fullxrange,self.templsol[b](fullxrange,[self.stats[b].templatefit['stretch'],self.stats[b].templatefit['xoffset'],self.stats[b].templatefit['yoffset'],self.stats[b].templatefit['xstretch']],b), symbol='%s--'%self.su.mycolors[b], offset=offset)
                if plottemplate :
                     if savepng :
                    
                          pl.savefig(self.name+"_"+b+".template.png", bbox_inches='tight')
                     if save :
                         thisname=self.name+"_"+b+".template.pdf"
                         pl.savefig(thisname)
                         thisdir=            os.environ['SESNPATH']
                         os.system("perl %s/pdfcrop.pl %s"%(thisdir,thisname))
                     if plotspline:
                         if verbose: print "plotting spline"
                         x = self.photometry[b]['mjd'].astype(np.float64) 
                         fullxrange=np.arange(min(x),max(x),0.1)
                         a=self.snspline[b](fullxrange)
                         smoothed=smooth(a,window_len=5)
                         #		    smoothed=sp.signal.filter.medfilter(a,5)
                         #                    myplot_err(fullxrange,self.snspline[b](fullxrange),symbol='%s-'%self.su.mycolors[b], offset=offset)
                         #                    results = zip([x[0] for x in results], smoothed)
                         
                         myplot_err(fullxrange,smoothed,symbol='%s-'%self.su.mycolors[b], offset=offset, settopx=True)

            if Vmax and not self.flagmissmax:
                if self.Vmax:
                    try:
                        myplotarrow(float(self.Vmax)-2453000,min(self.photometry['V']['mag'])-0.5,label="V max")
                    except:
                        pass
#                else:
#                    try:
#                        myplotarrow(self.stats['V'].maxjd[0],min(self.photometry['V']['mag'])-0.5,label="V max")
#                    except:
#                        pass
            if verbose: print "Vmax:",self.Vmax, self.flagmissmax
            ax.tick_params('both', length=10, width=1, which='major')
            ax.tick_params('both', length=5, width=1, which='minor')
            if not self.flagmissmax:
                ax2 = ax.twiny()
                ax2.tick_params('both', length=10, width=1, which='major')
                ax2.tick_params('both', length=5, width=1, which='minor')
                ax2.set_xlabel("phase (days)")

                if verbose: print "putting second axis"
                Vmax4plot=self.Vmax
                if Vmax4plot > 2400000:
                    Vmax4plot -= 2400000
                if Vmax4plot > 52000:
                    Vmax4plot -= 53000
                ax2.set_xlim((myxlim[0]-Vmax4plot,myxlim[1]-Vmax4plot))
                if (myxlim[1]-myxlim[0])<100:
                    ax2.xaxis.set_major_locator(MultipleLocator(20))
                ax2.xaxis.set_minor_locator(MultipleLocator(10))
            for i in notused: mybands.remove(i)
            if savepng:
                pl.savefig(self.name+"_"+''.join(mybands)+'.png', bbox_inches='tight')
            if save:
                thisname=self.name+"_"+''.join(mybands)+'.pdf'
                pl.savefig(thisname)
                print "running pdfcrop.pl"

                os.system("perl %s/pdfcrop.pl %s"%(os.environ['SESNPATH'],thisname))
                    
            if nir:
                legends=[]
                myfig = pl.figure(fig+10)
                ax=myfig.add_subplot(1,1,1)
                if band== '': mybands=self.su.bandsnir
                else: mybands=[band]
                if xlim:
                    myxlim = xlim-53000.0
                
                elif not relim:
                    xlim = pl.xlim()
                else:
                    xlim=[float(self.stats[mybands[0]].maxjd[0])-53010,float(self.stats[mybands[0]].maxjd[0])-53000+10]
                if ylim: 
                    myylim = ylim    
                elif not self.stats[mybands[0]].maxjd[0]==0.0:
                    myxlim=(min(xlim[0],float(self.stats[mybands[0]].maxjd[0])-10-53000),
                            max(xlim[1],float(self.stats[mybands[0]].maxjd[0])+10-53000))
                elif self.Vmax:
                    if self.Vmax > 10e5:
                        myxlim=(float(self.Vmax)-10-2400000-53000,
                                float(self.Vmax)+10-2400000-53000)
                    else:
                        myxlim=(float(self.Vmax)-10-53000,
                                float(self.Vmax)+10-53000)
                else: 
                    myxlim = (min(self.photometry[mybands[0]]['mjd'])-20-53000,
                              max(self.photometry[mybands[0]]['mjd'])+20-53000)
                bandswdata=[]
                for b in mybands:
                    if offsets:offset = boffsets[b]
                    else: offset=0.0
                    if verbose: print "band here", b
                    if self.filters[b]==0:
                        if verbose: print "nothing to plot for ",b
                        notused.append(b)
                        continue
                    bandswdata.append(b)

                    if verbose:print "plotting band ",b," for ", self.name
                    if not relim or ylim:
                        ylim=pl.ylim()
                    elif relim:
                        ylim=[float(self.stats[mybands[0]].maxjd[1])+10,float(self.stats[mybands[0]].maxjd[1])-1-5]
                        if myylim == (0,0):
                            myylim=(max(ylim[1],max(self.photometry[b]['mag'])+1),
                                    min(ylim[0],min(self.photometry[b]['mag'])-1))
                        else:
                            myylim=(max(ylim[1],(max(myylim[0],max(self.photometry[b]['mag'])+1))),
                                    min(ylim[0],min(myylim[1],min(self.photometry[b]['mag'])-1)))
                    if verbose: print "myylim",myylim
#                    if title=='':
#                        title =self.name 
#                    myplot_setlabel(xlabel='JD - 2453000.00',ylabel=ylabel,title=title, ax=ax, label=label)
                    if symbol=='':
                        symbol='%s%s'%(self.su.mycolors[b],self.su.myshapes[b])

                    legends.append(myplot_err(self.photometry[b]['mjd']-53000.0,
                                              self.photometry[b]['mag'],
                                              yerr=self.photometry[b]['dmag'],
                                              xlim=myxlim,
                                              ylim=myylim, symbol=symbol,offset=offset, fig=fig+10))
                    
                    symbol=''
                    if verbose: print "vmax",self.Vmax
                    if plotpoly or plottemplate:
                        if self.Vmax:
                            fullxrange=np.arange(float(self.Vmax)-2453000.0-10.,float(self.Vmax)-2453000.0+40.0,0.1)
                        else:
                            try:
                                fullxrange=np.arange(self.stats['V'].maxjd[0]-10.,self.stats['V'].maxjd[0]+40.,0.1)
                            except:
                                continue


                        if plotpoly and self.solution[b]['sol']:
                            myplot_err(fullxrange,self.solution[b]['sol'](fullxrange),symbol='%s-'%self.su.mycolors[b], offset=offset)
                        if plottemplate and not self.stats[b].templrchisq == 0:
                            myplot_err(fullxrange,self.templsol[b](fullxrange,[self.stats[b].templatefit['stretch'],self.stats[b].templatefit['xoffset'],self.stats[b].templatefit['yoffset'],self.stats[b].templatefit['xstretch']],b), symbol='%s--'%self.su.mycolors[b], offset=offset)
                    if savepng:
                        pl.savefig(self.name+"_"+b+".template.png", bbox_inches='tight')
                    if save:
                        thisname=self.name+"_"+b+".template.pdf"
                        pl.savefig(thisname)
                        os.system("perl %s/pdfcrop.pl %s"%(os.environ['SESNPATH'],thisname))
                    if plotspline:
                        if versbose: print "plotting spline"
                        x = self.photometry[b]['mjd'].astype(np.float64) 
                        fullxrange=np.arange(min(x),max(x),0.1)
                        a=self.snspline[b](fullxrange)
                        smoothed=smooth(a,window_len=5)
                        #		    smoothed=sp.signal.filter.medfilter(a,5)
                        #                    myplot_err(fullxrange,self.snspline[b](fullxrange),symbol='%s-'%self.su.mycolors[b], offset=offset)
                        #                    results = zip([x[0] for x in results], smoothed)
                        
                        myplot_err(fullxrange,smoothed,symbol='%s-'%self.su.mycolors[b], offset=offset)
                        #                    print smoothed
                        
                if Vmax:
                    if self.Vmax:
                        myplotarrow(float(self.Vmax)-2453000,min(self.photometry['V']['mag'])-0.5,label="V max")
#                    else:
#                        try:
#                            myplotarrow(self.stats['V'].maxjd[0],min(self.photometry['V']['mag'])-0.5,label="V max")
#                        except:
#                            pass

                pl.legend(legends[::-1],bandswdata[::-1], loc=1, ncol=1,prop={'size':12}, numpoints = 1)
                for i in notused: mybands.remove(i)
                if savepng:
                    bnds="UBVRIriHJK.png"
                    pl.savefig(self.name+"_"+bnds, bbox_inches='tight')
                if save:
                    bnds="UBVRIriHJK.pdf"
                    thisname=self.name+"_"+bnds
                    pl.savefig(thisname)
                    os.system("perl %s/pdfcrop.pl %s"%(os.environ['SESNPATH'],thisname))
                    
        if color:
            rc('axes', linewidth=1)
            if photometry:
                print "need new fig number"
                fig =100+fig
            myfig = pl.figure(fig)
            if not singleplot:
                ax=myfig.add_subplot(2,1,1)
            else:
                adjustFigAspect(myfig,aspect=2)
                ax=myfig.add_subplot(1,1,1)                

            ax.minorticks_on()

            majorFormatter = FormatStrFormatter('%.1f')
            minorLocator   = MultipleLocator(0.2)
            majorLocator   = MultipleLocator(1.0)
            if '06jc'  in self.name:
                majorLocator   = MultipleLocator(2.0)
            ax.yaxis.set_minor_locator(minorLocator)
            ax.yaxis.set_major_locator(majorLocator)
            ax.yaxis.set_major_formatter(majorFormatter)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(16) 
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16) 
                # specify integer or one of preset strings, e.g.
                #tick.label.set_fontsize('x-small') 

#            adjustFigAspect(myfig)#,aspect=aspect)
            legends=[]        
            notused=[]
            if c== '': 
                 mybands=[k for k in self.su.cs.iterkeys()]
                 if mybands.index("B-i"):
                      del mybands[mybands.index("B-i")]
                 print mybands
            else: mybands=[c]
            myylim=(0,0)
            myxlim=(-15,85)
            workingbands=[]
            for b in mybands:
                if len(self.colors[b]['mjd'])==0:
                    if verbose:
                        print "nothing to plot for ",b
                    notused.append(b)
                    continue
                if verbose:
                    print "plotting band ",b," for ", self.name
                    print self.colors[b]['mjd'],self.colors[b]['mag']

                    #                myxlim=(float(min(myxlim[0],min(self.colors[b]['mjd'])-10)),
                    #                        float(max(myxlim[1],max(self.colors[b]['mjd'])+10)))
                myylim=(float(min(myylim[0],min(self.colors[b]['mag'])-0.5)),
                        float( max(myylim[1],max(self.colors[b]['mag'])+0.5)))
                if self.name[-2].isdigit():                
                    thename=self.name[:-1]+self.name[-1].upper()
                else:
                    thename=self.name
                myplot_setlabel(xlabel='',ylabel='color (mag)',label=thename , ax=ax, labsize=15)
                if '06jc' in self.name:
                    myxlim=(0,85)
                    myylim=(-2,7.5)

                l,=  myplot_err(self.colors[b]['mjd'],#)-53000.0,
                                self.colors[b]['mag'],
                                yerr=self.colors[b]['dmag'],
                                xlim=myxlim,ylim=myylim, symbol='%so'%self.su.mycolorcolors[b],offset=offset,alpha=0.5)                 
                workingbands.append(b)
                legends.append(l)


            if '06jc' in self.name:
                loc=2
                ncol=4
                pl.xlim(pl.xlim()[0]-10,pl.xlim()[1])
            else:
                loc=1
                ncol=1

#            sort2vectors(v1,v2)
            pl.legend(legends,workingbands, loc=loc, ncol=ncol,prop={'size':12}, numpoints = 1)
            if singleplot:
                ax.set_xlabel("phase (days)")
            else:
                ax2=myfig.add_subplot(2,1,2, sharex=ax)
                ax2.minorticks_on()
                majorFormatter = FormatStrFormatter('%.1f')
                minorLocator   = MultipleLocator(0.2)
                ax2.yaxis.set_minor_locator(minorLocator)
                ax2.yaxis.set_major_formatter(majorFormatter)

                for b in mybands:
                    if len(self.colors[b]['mjd'])==0:
                        if verbose:
                            print "nothing to plot for ",b
                        notused.append(b)
                        continue
                    if verbose:
                        print "plotting band ",b," for ", self.name
                        print self.colors[b]['mjd'],self.colors[b]['mag']

                    #                myxlim=(float(min(myxlim[0],min(self.colors[b]['mjd'])-10)),
                    #                        float(max(myxlim[1],max(self.colors[b]['mjd'])+10)))
                    myylim=(float(min(myylim[0],min(self.colors[b]['mag'])-0.5)),
                            float( max(myylim[1],max(self.colors[b]['mag'])+0.5)))

                    myplot_setlabel(xlabel='phase (days)',ylabel='color' , ax=ax2)

                    if '06jc' in self.name:
                        
                        ax2.annotate("red", xy=(-10, 2.05), xycoords='data',
                                     xytext=(-10,1.4 ), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->",color='#b20000'),)
                        ax2.annotate("blue", xy=(-10, -0.65), xycoords='data',
                                     xytext=(-10,-0.2 ), textcoords='data',  ha='center',
                                     arrowprops=dict(arrowstyle="->",color='#0066cc'),)
                    else:
                        ax2.annotate("red", xy=(80, 2.05), xycoords='data',
                                     xytext=(80,1.4 ), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->",color='#b20000'),)
                        ax2.annotate("blue", xy=(80, -0.65), xycoords='data',
                                     xytext=(80,-0.2 ), textcoords='data', ha='center',
                                     arrowprops=dict(arrowstyle="->",color='#0066cc'),)
                    myplot_hist(self.colors[b]['mjd'],#)-53000.0,
                                self.colors[b]['mag'],
                                xlim=myxlim,ylim=(-0.75,2.2), symbol='%so'%self.su.mycolorcolors[b],offset=offset, ax=ax2, nbins=nbins)#    
                
                    #print b,b[0],self.su.mycolors[b[0]]
                    #                legends.append(l)
            for i in notused: mybands.remove(i)
            if savepng:
                pl.savefig(self.name+"_color"+'.png', bbox_inches='tight', dpi=150)
            if save:
                thisname=self.name+"_color"+'.pdf'
                pl.savefig(thisname)
                os.system("perl %s/pdfcrop.pl %s"%(os.environ['SESNPATH'],thisname))
        if show:
            pl.show()

        return myfig

    def cleanphot(self):
        self.Vmaxmag=0.0
        self.filters={}
        for b in self.su.bands:
            self.filters[b]=0
        self.polysol={}
        self.snspline={}
        self.templsol={}
        self.solution={}
        self.photometry={}
        self.stats={}
        self.colors={}
        self.maxcolors={}
        self.maxmags={}

        self.flagmissmax=True
#        self.lc={}
        for b in self.su.bands:
            self.photometry[b]={'mjd':np.zeros((0),float),'mag':np.zeros((0),float),'dmag':np.zeros((0),float),'extmag':np.zeros((0),float), 'camsys':[''], 'natmag':np.zeros((0),float)}
            self.stats[b]=snstats()
            self.polysol[b]=None
            self.snspline[b]=None
            self.templsol[b]=None            
            self.solution[b]={'sol':None,'deg':None, 'pars':None, 'resid':None}
            self.maxmags[b]={'epoch':0.0,'mag':float('NaN'), 'dmag':float('NaN')}

        for c in self.su.cs:
            self.maxcolors[c]={'epoch':0.0,'color':float('NaN'), 'dcolor':float('NaN')}
            self.colors[c]={'mjd':[],'mag':[],'dmag':[]}#np.zeros((0),float),'mag':np.zeros((0),float),'dmag':np.zeros((0),float)}
                            
        self.polyfit= None
        self.setphot()

    def getphot(self, ebmv=0, RIri=False, verbose=False):
        if verbose:
             print self.snnameshort,"E(B-V)",ebmv
             print self.filters
        for b in self.su.bands:
            ##############################setting up band#########################
            if self.filters[b] <=0:
                continue
            indx=np.array(np.where(self.lc['photcode']==self.su.photcodes[b][0])[0])            
            if not self.su.photcodes[b][1]==self.su.photcodes[b][0]:
                if len( indx)==0:
                    indx=np.array(np.where(self.lc['photcode']==self.su.photcodes[b][1])[0])
                else:
                    try:
                        newindx=np.concatenate([np.where(self.lc['photcode']==self.su.photcodes[b][1])[0],indx])
                        indx=newindx
                    except:
                        pass
            self.getphotband(indx, b)
            if verbose: print self.snnameshort,b,self.photometry[b]['mjd']
            self.stats[b].tlim = (min(self.photometry[b]['mjd']), max(self.photometry[b]['mjd']))
            self.stats[b].maglim = (min(self.photometry[b]['mag']), max(self.photometry[b]['mag']))
        if not ebmv == 0:
            self.extcorrect(ebmv)
        try:
            self.Vmax=float(self.Vmax)
            if self.Vmax < 2400000 and not self.Vmax==0: 
                self.Vmax+=2453000.5
        except:
            if self.Vmax.startswith("<0") and len(self.photometry["V"]['mjd'])>0:
                self.Vmax ="<24"+str(self.photometry["V"]['mjd'][0]) 
            if self.Vmax.startswith("<"):
                try:
                    self.Vmax=self.Vmax.replace("<","")
                    
                    self.Vmax=float(self.Vmax)     
#                    if self.Vmax=float(self.Vmax)
                except:
                    pass
            self.nomaxdate=True

        if self.filters['R'] == 0 and self.filters['I'] == 0 and self.filters['r'] > 0 and self.filters['i']>0 and RIri == True :
             self.getonecolor('r-i')
#             print "right here",self.filters
#             self.printsn(color=True,cband='r-i')
             tmptimeline=np.array(self.colors['r-i']['mjd'])+self.Vmax
             if tmptimeline[0]>2300000:
                  tmptimeline-=2400000.5
             tmplin = interp1d(tmptimeline, self.colors['r-i']['mag'],kind='linear', bounds_error=False)
             tmplinerr = interp1d(tmptimeline, self.colors['r-i']['dmag'],kind='linear', bounds_error=False)
             self.photometry['R']['mjd']=self.photometry['r']['mjd']
             self.photometry['R']['mag']=self.photometry['r']['mag']-0.153*tmplin(self.photometry['R']['mjd']) - 0.117
             self.photometry['R']['dmag']=np.sqrt(self.photometry['r']['dmag']**2+tmplinerr(self.photometry['R']['mjd'])**2 + 0.00043**2)             

             self.photometry['I']['mjd']=self.photometry['R']['mjd']
             self.photometry['I']['mag']=self.photometry['R']['mag']-0.930*tmplin(self.photometry['I']['mjd']) - 0.259
             self.photometry['I']['dmag']=np.sqrt(self.photometry['R']['dmag']**2+tmplinerr(self.photometry['I']['mjd'])**2 + 0.00055**2)             
             self.filters['R']=self.filters['r']
             self.filters['I']=self.filters['R']
             
        if self.filters['r'] == 0 and self.filters['i'] == 0 and self.filters['B'] > 0 and self.filters['V']>0 and RIri == True :
             ##Jester et al. (2005) transformations via 
             ##https://www.sdss3.org/dr8/algorithms/sdssUBVRITransform.php
             self.getonecolor('B-V')
             print self.Vmax, np.array(self.colors['B-V']['mjd'])+self.Vmax
             tmptimeline=np.array(self.colors['B-V']['mjd'])+self.Vmax
             if tmptimeline[0]>2300000:
                  tmptimeline-=2400000.5
             tmplin = interp1d(tmptimeline, self.colors['B-V']['mag'],kind='linear', bounds_error=False)             
             tmplinerr = interp1d(tmptimeline, self.colors['B-V']['dmag'],kind='linear', bounds_error=False)             
             self.photometry['r']['mjd']=self.photometry['V']['mjd']
             self.photometry['r']['mag']=self.photometry['V']['mag']- 0.42*tmplin(self.photometry['r']['mjd']) + 0.11
             self.photometry['r']['dmag']=np.sqrt(self.photometry['V']['dmag']**2+tmplinerr(self.photometry['r']['mjd'])**2 + 0.03**2)
             self.filters['r']=self.filters['V']
             
             self.getonecolor('R-I')
             tmptimeline=np.array(self.colors['R-I']['mjd'])+self.Vmax
             if tmptimeline[0]>2300000:
                  tmptimeline-=2400000.5
             tmplin = interp1d(tmptimeline, self.colors['R-I']['mag'],kind='linear', bounds_error=False)
             tmplinerr = interp1d(tmptimeline, self.colors['R-I']['dmag'],kind='linear', bounds_error=False)
             self.photometry['i']['mjd']=self.photometry['r']['mjd']
             self.photometry['i']['mag']=self.photometry['r']['mag']- 0.91*tmplin(self.photometry['i']['mjd']) + 0.20
             self.photometry['i']['dmag']=np.sqrt(self.photometry['r']['dmag']**2+tmplinerr(self.photometry['i']['mjd'])**2 + 0.03**2)             
             self.filters['i']=self.filters['r']


          

    def getphotband(self,indx, b):
        try:
            self.photometry[b]={'mjd':self.lc['mjd'][indx],
                                'mag':self.lc['ccmag'][indx],
                                'dmag':self.lc['dmag'][indx], \
                                'natmag':self.lc['mag'][indx],
            'camsys':[survey[0] if i<survey_dates[0] else survey[1] if i<survey_dates[1] else survey[2][self.pipeline]  if i<survey_dates[2] else survey[3]   for i in self.lc['mjd'][indx]]}
        except:
            print "#############\n\n\n failed to get photometry \n\n\n##############"
            pass
#            self.photometry[b]={'mjd':self.lc['mjd'][indx],'mag':self.lc['mag'][indx],'dmag':self.lc['dmag'][indx], 'camsys':[survey[0] if i<survey_dates[0] else survey[1] if i<survey_dates[1] else survey[2][self.pipeline]  if i<survey_dates[2] else survey[3]   for i in self.lc['mjd'][indx]]}
         
    def extcorrect(self,ebmv, verbose=False):
        if verbose: print "ebmv",ebmv
        for b in self.su.bands:
            R=self.su.AonEBmV[b]*ebmv
            self.photometry[b]['mag']-=R
    
 
    def getmaxcolors(self,band, tol=5):
        (self.maxcolors[band]['epoch'], self.maxcolors[band]['color'], self.maxcolors[band]['dcolor'])=self.getepochcolors(band, tol=tol)
        
        # 
        print self.name, self.maxcolors[band]['epoch'],self.maxcolors[band]['color'],self.maxcolors[band]['dcolor']##
#
#        if len(self.colors[band]['mjd'])<=1:
#            "print no data"
#            return -1
#        else:
#            indx=np.where(abs(np.array(self.colors[band]['mjd']))==min(abs(np.array(self.colors[band]['mjd']))))[0]
#            if len(indx)>0:
#                indx=indx[0]###
#                
#            if abs(self.colors[band]['mjd'][indx]) > tol:
#                print "#########\n\n\n no data within 5 days ",self.name,self.type,"\n\n\n############"
#                self.maxcolors[band]['epoch'] = float('NaN')
#                self.maxcolors[band]['color'] =float('NaN')
#                self.maxcolors[band]['dcolor'] =float('NaN')
#            else:
#                print "#########\n\n\n YES data within 5 days ",self.name,self.type,"\n\n\n############"
#                self.maxcolors[band]['epoch'] = self.colors[band]['mjd'][indx]
#                self.maxcolors[band]['color'] = self.colors[band]['mag'][indx]
#                self.maxcolors[band]['dcolor'] =self.colors[band]['dmag'][indx]
#            print self.maxcolors[band]['epoch'],self.maxcolors[band]['color'],self.maxcolors[band]['dcolor']

    def getepochcolors(self,band, epoch=0.0, tol=5, interpolate=False):
#        print self.name,band,self.colors[band]['mjd'], self.colors[band]['mag'], tol
        
        if len(self.colors[band]['mjd'])<1:
             print "no data"
             return (float('NaN'),float('NaN'),float('NaN'))
        else:
          if not interpolate:
               indx,=np.where(abs(np.array(self.colors[band]['mjd'])-epoch)==min(abs(np.array(self.colors[band]['mjd'])-epoch)))
               #            print self.name,np.array(self.colors[band]['mjd']), epoch,abs((np.array(self.colors[band]['mjd']))-epoch)
               #            print indx
               if len(indx)>0:
                    indx=indx[0]
               if abs(self.colors[band]['mjd'][indx]-epoch) > tol:
                    return (float('NaN'),float('NaN'),float('NaN'))
               return (self.colors[band]['mjd'][indx], self.colors[band]['mag'][indx],self.colors[band]['dmag'][indx])
          else:
               ddate=self.colors[band]['mjd']-epoch
               try:
                    indlft, = np.arange(ddate.size)[ddate==ddate[ddate<0].max()]
                    indrgt, = np.arange(ddate.size)[ddate==ddate[ddate>0].min()]
               except ValueError:
                    print "the datapoints don't surround epoch, im at the edge ",epoch
                    return (float('NaN'),float('NaN'),float('NaN'))
                    
               mag=interp1d(self.colors[band]['mjd'][indlft:indrgt+1], self.colors[band]['mag'][indlft:indrgt+1],kind='linear')(epoch)
               err=np.sqrt( self.colors[band]['dmag'][indlft]**2 + self.colors[band]['dmag'][indrgt]**2)
               return (epoch,float(mag),err)
               

    def getmagmax(self,band,tol=5, verbose=False):
        if is_empty(self.metadata, verbose=verbose):
             return -1
        if verbose:     print "we", self.name, 'cfa'+band.upper()+'max', self.metadata['cfa'+band.upper()+'max']
        try :
           self.maxmags[band]['epoch']=float(self.metadata['CfA '+band.upper()+'JD bootstrap']) 
           self.maxmags[band]['mag']=float(self.metadata['cfa'+band.upper()+'max'])
           self.maxmags[band]['dmag']=float(self.metadata['cfa'+band.upper()+'maxerr'])
           if verbose: print "we have max's" ,self.maxmags[band]['epoch'],self.maxmags[band]['mag'],self.maxmags[band]['dmag']
        except:
             if verbose:  print "no max's"
             if self.Vmax:
                  if verbose: print "self.Vmax:",self.Vmax
                  if not type(self.Vmax)==float:
                       pass
                  if float(self.Vmax) >2000000:
                        Vmax = float(self.Vmax)-2400000.5
                  if verbose: print "Vm:",self.Vmax,Vmax
                  self.maxmags[band]['epoch'],self.maxmags[band]['mag'],self.maxmags[band]['dmag']=self.getmagmax_band(band,epoch=Vmax+coffset[band])
#self.getepochmags(band,epoch=Vmax+coffset[band],tol=tol)

#                  print self.maxmags[band]['epoch'],self.maxmags[band]['mag'],self.maxmags[band]['dmag'],"after getepochmags"
        

    def getmagmax_band(self,band,epoch=None,tol=5):

        if not epoch:
             epoch=Vmax+coffset[band]
             if self.Vmax or not type(self.Vmax)==float:
                  if verbose: print "self.Vmax:",self.Vmax
                  if float(self.Vmax) >2000000:
                       Vmax = float(self.Vmax)-2400000.5
                       if verbose: print "Vm in getmagmax_band:",self.Vmax,Vmax
                  else:         return (0,0,0,0)
        indx,=np.where((self.photometry[band]['mjd']<epoch+15) & (self.photometry[band]['mjd']>self.photometry[band]['mjd']-8))
#        print indx,self.photometry[band]['mjd'],epoch
        x=self.photometry[band]['mjd'][indx]
        y=self.photometry[band]['mag'][indx]
        e=self.photometry[band]['dmag'][indx]
        #       print x,min(x),max(x)
#        pl.figure()
#        pl.plot(x-self.Vmax+2400000.5,y)
#        pl.errorbar(x-self.Vmax+2400000.5,y,yerr=e)
#        try:
        try:
            nodes=splrep(x,y, w=1.0/(self.photometry[band]['dmag'][indx])**2,k=2)
            newx=np.arange(x[0],x[-1],0.1)
            splx = splev(newx, nodes)
            mymax = min(splx)
#            print mymax,
            epmax = newx[np.where(splx==mymax)][0]
#            print epmax
#            pl.plot(newx-self.Vmax+2400000.5,splx)
#            return (epmax,mymax)
        except:
            print "splining to find max mag failed for band ",band
            return(0,0,0)
            
#        pl.errorbar(x-self.Vmax+2400000.5,y,yerr=e)
#        
#        pl.ylim(pl.ylim()[1],pl.ylim()[0])
#        pl.show()
#        accept=int(raw_input("is this spline reasonable? 1 yes 0 no"))
#if accept>0:
#        print epmax,mymax, np.sqrt((e[np.where(x<epmax)[0][-1]])**2+(e[np.where(x>epmax)[0][0]])**2)
        try:
                  return (epmax,mymax, np.sqrt((e[np.where(x<epmax)[0][-1]])**2+(e[np.where(x>epmax)[0][0]])**2))
        except IndexError:
                  return (epmax,mymax, e[np.where(x==epmax)[0][0]])
#        elif accept<0:
#             sys.exit()
#        else:
#             return(0,0,0)

        
#        (self.maxmags[band]['epoch'], self.maxmags[band]['mag'], self.maxmags[band]['dmag'])=self.getepochmags(band, tol=tol, epoch=bandepoch)
        
    def getepochmags(self,band, phase = None,epoch=None, tol=5, interpolate=False, verbose=False, plot=False):
            if verbose: print "getting band", band, "magnitudes"
            if self.filters[band]==0:
                if verbose: print "no data in filter ",band
                return (float('NaN'),float('NaN'),float('NaN'))
            if not epoch:
                 if not phase:
                      epoch = self.Vmax-2400000.5
                 if phase: 
                      epoch = self.Vmax-2400000.5+phase                      
            myc=band.lower()
            if myc=='i':
                 myc='b'
            if myc=='j':
                 myc='m'
            if myc=='h':
                 myc='c'
            if myc=='k':
                 myc='m'
            if plot:
                 if phase:
                      pl.plot(self.photometry[band]['mjd']-self.Vmax+2400000.5,self.photometry[band]['mag'], '%s-'%myc)
                      pl.errorbar(self.photometry[band]['mjd']-self.Vmax+2400000.5,self.photometry[band]['mag'],self.photometry[band]['dmag'],fmt='k.')
                 else:
                      pl.plot(self.photometry[band]['mjd'],self.photometry[band]['mag'], '%s-'%myc)
                      pl.errorbar(self.photometry[band]['mjd'],self.photometry[band]['mag'],self.photometry[band]['dmag'],fmt='k.')
                 pl.title(self.name)

                 #            pl.draw()
            try:
                 ddate=abs(np.array(self.photometry[band]['mjd'])-epoch)
                 indx,=np.where(ddate==min(ddate))
            except ValueError:
                 print "no min "
                 indx=[]
#            print self.name,np.array(self.colors[band]['mjd']), epoch,abs((np.array(self.colors[band]['mjd']))-epoch)
            if verbose: print indx, self.photometry[band]['mjd'][indx], abs(self.photometry[band]['mjd'][indx]-epoch)

            if len(indx)>0:
                 indx=indx[0]
#            print  abs(self.colors[band]['mjd'][indx]-epoch)

            if min(ddate)==0:
                 if verbose: print "observaions at same exact epoch!"
                 print self.photometry[band]['mjd'][indx], self.photometry[band]['mag'][indx],self.photometry[band]['dmag'][indx]
                 return (self.photometry[band]['mjd'][indx], self.photometry[band]['mag'][indx],self.photometry[band]['dmag'][indx])

            if min(ddate) > tol and not interpolate:
                if verbose: print "nodata within ",tol,"days of ",epoch
                return (float('NaN'),float('NaN'),float('NaN'))
            if verbose: print self.photometry[band]['mjd'][indx], self.photometry[band]['mag'][indx],self.photometry[band]['dmag'][indx]
            if not interpolate:
                if plot:
                     if phase:pl.errorbar(self.photometry[band]['mjd'][indx]-thissn.Vmax+2400000.5, self.photometry[band]['mag'][indx],yerr=self.photometry[band]['dmag'][indx],fmt='r')
                     else: pl.errorbar(self.photometry[band]['mjd'][indx], self.photometry[band]['mag'][indx],yerr=self.photometry[band]['dmag'][indx],fmt='r')
                return (self.photometry[band]['mjd'][indx], self.photometry[band]['mag'][indx],self.photometry[band]['dmag'][indx])

            if interpolate:
                 if verbose: print "interpolating"
                 ddate=self.photometry[band]['mjd']-epoch
                 try:
                      indlft, = np.arange(ddate.size)[ddate==ddate[ddate<0].max()]
                      indrgt, = np.arange(ddate.size)[ddate==ddate[ddate>0].min()]
                 except ValueError:
                      print "the datapoint dont surroound epoch, im at the edge ",epoch
                      return (self.photometry[band]['mjd'][indx], self.photometry[band]['mag'][indx],self.photometry[band]['dmag'][indx])

                 mag=interp1d(self.photometry[band]['mjd'][indlft:indrgt+1], self.photometry[band]['mag'][indlft:indrgt+1],kind='linear')(epoch)
                 err=np.sqrt( self.photometry[band]['dmag'][indlft]**2 + self.photometry[band]['dmag'][indrgt]**2)
                 if plot:
                      if phase:pl.errorbar(epoch-self.Vmax+2400000.5,float(mag),yerr=err,fmt='r')
                      else:pl.errorbar(epoch,float(mag),yerr=err,fmt='r')
                 return (epoch,float(mag),err)
                 
#            return (self.photometry[band]['mjd'][indx], self.photometry[band]['mag'][indx],self.photometry[band]['dmag'][indx])
            
        
    def getcolors(self, BmI=False):
     ###setup B-I for bolometric correction as per Lyman 2014
        for ckey in self.su.cs.iterkeys():
##################iterate over the color keys to get the colors cor each object
##############THIS IS LAME AND I MUST FIND A BETTER WAY TO DO IT!!###########
             #print ckey
             self.getonecolor(ckey)
             if len(self.colors['r-i']['mjd'])==0: self.getonecolor('r-i')
             if BmI:
               if self.filters['I'] == 0 and (self.filters['i']>0 and self.filters['r']>0):
                   print self.photometry['r'],self.photometry['i']

                   tmpmjd=[]
                   tmpI=[]
                   tmpIerr=[]
                   for k,mjd in enumerate(self.photometry['r']['mjd']):
                        timediff=np.abs(np.array(self.colors['r-i']['mjd'])+self.Vmax-2400000.5-mjd)
                        print timediff
                        if min(timediff)<1.5:
                             mjdind = np.where(timediff == min(timediff))[0]
                             if len(mjdind)>1:
                                  mjdind=mjdind[0]
                             tmpmjd.append(np.mean([self.colors['r-i']['mjd'][mjdind]+self.Vmax-2400000.5,mjd]))
                             tmpI.append(self.photometry['r']['mag'][k] - 1.2444*(self.colors['r-i']['mag'][mjdind]) - 0.3820)
                             tmpIerr.append(np.sqrt(self.photometry['r']['dmag'][k]**2+self.colors['r-i']['dmag'][mjdind]**2+0.0078**2))
                   self.photometry['I']['mjd']=np.array(tmpmjd)
                   self.photometry['I']['mag']=np.array(tmpI)
                   self.photometry['I']['dmag']=np.array(tmpIerr)
                   self.filters['I']=len(tmpmjd)
                   #        self.printsn(photometry=True)
                   self.getonecolor('B-I')
                   #        self.printsn(color=True)



    def getonecolor(self,ckey, verbose=False):
          if verbose:
               print "ckey",ckey
               print self.colors[ckey]
          if not self.colors[ckey]['mjd'] == [] :
               if not ckey == 'r-i':
                    print "color",ckey,"is already there. clean it first if you want me to redo it"
          else:

               for k,mjd in enumerate(self.photometry[ckey[0]]['mjd']):
                    #check vmax:
                    if not type(self.Vmax)==float or float(self.Vmax) <200000:
                         self.Vmax = float(self.photometry[ckey[0]]['mjd'][0])+2453000.5
                    mjd = float(mjd)
                    try:
                         timediff=min(abs(self.photometry[ckey[2]]['mjd']-mjd))
                         #                        print "timediff ", timediff
                    except:
                         continue
                    if timediff<1.5:
                         indx= np.where(abs(self.photometry[ckey[2]]['mjd']-mjd) == timediff)[0]
                         indx=indx[0]
                         #                    print "mags ",mjd, photometry[ckey[2]]['mag'][indx]
                         self.colors[ckey]['mjd'].append(mjd-float(self.Vmax)+2400000.5)
                         
                         self.colors[ckey]['mag'].append(self.photometry[ckey[0]]['mag'][k]-self.photometry[ckey[2]]['mag'][indx])
                         self.colors[ckey]['dmag'].append(np.sqrt(self.photometry[ckey[0]]['dmag'][k]**2+self.photometry[ckey[2]]['dmag'][indx]**2))         
               self.colors[ckey]['mjd']=np.array(self.colors[ckey]['mjd'])
               self.colors[ckey]['mag']=np.array(self.colors[ckey]['mag'])
               self.colors[ckey]['dmag']=np.array(self.colors[ckey]['dmag'])


    def savecolors(self, band=''):
        if band== '': mybands=[k for k in self.su.cs.iterkeys()]
        else: mybands=[band]
        
        for c  in  mybands:
            fout=open(self.name+"_"+c+".dat","w")
            if len(self.colors[c]['mjd'])>0:
                for i,mjd in enumerate(self.colors[c]['mjd']):
                    print >>fout,self.colors[c]['mjd'][i],self.colors[c]['mag'][i],self.colors[c]['dmag'][i]
        

    def loadsn(self,f, fnir=None, verbose=False, superverbose=False):
        if f.split('/')[-1].startswith('slc'):
            self.pipeline='CfA4'
            if verbose:
                print "lightcurve type CfA4 ",f
            try:
                self.lc=np.loadtxt(f,usecols=(0,1,5,3,7),\
                                  dtype={'names': ('photcode','mjd',\
                                                       'mag','dmag','ccmag'),\
                                             'formats': ('S2', 'f', 'f','f','f')})
                flux = 10**(-self.lc['ccmag']/2.5)*5e10
                dflux = flux*self.lc['dmag']/LN10x2p5
            except:
                try:
                    self.lc=np.loadtxt(f,usecols=(0,1,5,4),\
                                      dtype={'names': ('photcode','mjd',\
                                                           'mag','dmag'),\
                                                 'formats': ('S2', 'f', 'f','f')})
#                    self.lc['ccmag'] = self.lc['mag']
                    flux = 10**(-self.lc['mag']/2.5)*5e10
                    dflux = flux*self.lc['dmag']/LN10x2p5
                except:
                    print "file ",f," failed. wrong file format? moving on "
                    return 0,0,0,0
        #            dflux = 10**(-lc['dmag']/2.5)
        elif f.split('/')[-1].startswith('sn'):
            self.pipeline='CfA3'
            if verbose:
                print "lightcurve type CfA3"
            try:
                self.lc=np.loadtxt(f,usecols=(0,1,6,7,8,),
                              dtype={'names': ('photcode','mjd','mag','dmag',
                                               'ccmag'),\
                                         'formats': ('S2', 'f', 'f','f','f')})
                self.lc['photcode']=['%02d'%int(p) for p in self.lc['photcode']]
                flux = 10**(-self.lc['ccmag']/2.5)*5e10
                dflux = flux*self.lc['dmag']/LN10x2p5
                if superverbose:
                    print self.lc['mjd']
                    print self.lc['ccmag']
                    print self.lc['photcode']
                
            except:
                print "file ",f," failed. wrong file format (2)? moving on"
                return 0,0,0,0
        else:
            print "what kind of file is this??? "+f
            return 0,0,0,0
        if fnir :
            if verbose:
                print "doing nir",self.fnir
            try: 
                if verbose: print self.fnir
                self.nirlc=np.loadtxt(self.fnir,usecols=(0,1,2,3),\
                                  dtype={'names': ('photcode','mjd',\
                                                       'mag','dmag'),\
                                             'formats': ('S1', 'f', 'f','f')})
                
                nirflux = 10**(-self.nirlc['mag']/2.5)*5e10
                nirdflux = nirflux*self.nirlc['dmag']/LN10x2p5
                if len(self.nirlc['mjd']) > 0:
                       self.nir = True
                       lc={}
                       lc['photcode']=np.concatenate([self.lc['photcode'],self.nirlc['photcode']],axis=0)
                       lc['mjd']=np.concatenate([self.lc['mjd'],self.nirlc['mjd']])
                       lc['ccmag']=np.concatenate([self.lc['ccmag'],self.nirlc['mag']])
                       lc['mag']=np.concatenate([self.lc['mag'],self.nirlc['mag']])
                       lc['dmag']=np.concatenate([self.lc['dmag'],self.nirlc['dmag']])
                       self.lc=lc
                       return self.lc,flux,dflux,self.name
            except ValueError:
                if verbose: print "passing Value Error in loadsn nir, no nir data presusmibly"
                pass
        return self.lc,flux,dflux, self.name

    
            
    def getstats(self,b, verbose=False):
    #find max day and dm15
        xp=np.linspace(min(self.photometry[b]['mjd']),max(self.photometry[b]['mjd']),10000)

        maxjd = float(polyroots(self.solution[b]['pars'][::-1])[0].real)
        #                       print root,xp[np.where(solution['sol'](xp)==
        #                                               min(solution['sol'](xp)))[0]]

        if maxjd > min(self.photometry[b]['mjd']) and maxjd < max(self.photometry[b]['mjd']):
            if verbose: print "root found is within data range"
            self.stats[b].maxjd = [maxjd,self.solution[b]['sol'](maxjd)]
            self.stats[b].dm15 = self.stats[b].maxjd[1]-self.solution[b]['sol'](maxjd+15.0)

        else:
            print "root NOT found is within data range"
            mjdindex=np.where(self.solution[b]['sol'](xp)==
                              min(self.solution[b]['sol'](xp)))[0]
            if len(mjdindex)>1:
                mjdindex=[mjdindex[1]]
            try:
                self.stats[b].maxjd[0] = xp[mjdindex]
                self.stats[b].maxjd[1] = self.solution[b]['sol'](self.stats[b].maxjd[0])[0]
                self.stats[b].dm15 = self.stats[b].maxjd[1]-self.solution[b]['sol'](self.stats[b].maxjd[0]+15.0)
                
                
                try:
                    if len(self.stats[b].maxjd[0])>1:
                        self.stats[b].maxjd[0]=np.mean(self.stats[b].maxjd[0])
                        print "WARNING data has multiple points at same epoch"
                    if len(self.stats[b].maxjd[1])>1:
                        self.stats[b].maxjd[1]=np.mean(self.stats[b].maxjd[1])
                        print "WARNING data has multiple points at same epoch"
                except:
                    pass
            except:
                self.stats[b].maxjd[0]=-1000
                self.stats[b].dm15 = -1000
                print "WARNING data does not constraint the max mag"
                self.stats[b].flagmissmax=1

        try:self.stats[b].maxjd[0]=self.stats[b].maxjd[0][0]
        except:pass

        if b == 'V':
            self.Vmax=self.stats[b].maxjd[0]+2453000
            self.Vmaxmag=self.stats[b].maxjd[1]

             
        try:
            self.stats[b].m15data[0]=self.stats[b].maxjd[0]+15.
            self.stats[b].m15data[1]=[self.photometry[b]['mag'][np.where(
                        self.photometry[b]['mjd']==min([tmp for tmp in self.photometry[b]['mjd'] 
                                                        if tmp > self.stats[b].m15data[0]]))[0]],
                                      self.photometry[b]['mag'][np.where(
                        self.photometry[b]['mjd']==max([tmp for tmp in self.photometry[b]['mjd']
                                                        if tmp < self.stats[b].m15data[0]]))[0]]]
            try:
                if len(self.stats[b].m15data[0])>1:
                    self.stats[b].m15data[0]=np.mean(self.stats[b].m15data[0])
                    print "WARNING data has multiple points at same epoch"
            except : pass
            try:
                if len(self.stats[b].m15data[1])>1:
                    self.stats[b].m15data[1]=np.mean(self.stats[b].m15data[1])
            except : pass
            try:
                if len(self.stats[b].dm15)>1:
                    self.stats[b].dm15=np.mean(self.stats[b].dm15)
            except : pass
        except:
            print "WARNING data does not constraint mag at 15 days"
            self.stats[b].flagmiss15=2
#                        print bandlist[bandcounter]
        self.stats[b].success=1
            
    def printlog(self,b,inst,logoutput):
        print >> logoutput, "%-30s %s %-10s %-10s %02d "%( self.name, self.type, b, inst, self.filters[b]) ,
        try:
            print >> logoutput,"%5.3f %02d %5.3f  %-10s %d %-10s %d %-10s %5.3f "%(abs(self.stats[b].polyrchisq), self.stats[b].polydeg, np.median(np.abs(self.stats[b].polyresid))/0.6745, " ", self.stats[b].flagmissmax, " ", self.stats[b].flagmiss15," ",self.stats[b].dm15),
        except:
            print "could not print the log to logoutput"
            print b, self.stats[b].polyrchisq,self.stats[b].polydeg, np.median(np.abs(self.stats[b].polyresid))/0.6745, " ", self.stats[b].flagmissmax, " ", self.stats[b].flagmiss15," ",self.stats[b].dm15

        try:
            l=len(self.stats[b].maxjd[1])
            if l>1: 
                self.stats[b].maxjd[1]=np.mean(self.stats[b].maxjd[1])
                try: 
                    self.stats[b].maxjd[0]=np.mean(self.stats[b].maxjd[0])                        
                except: pass
                print "WARNING: maxjd is an array: something is very wrong with the fit!"
                print >>logoutput, "-1000 -1000"
                self.stats[b].flagmissmax,self.stats[b].flagbadfit=0.0,4
                pl.savefig("%s.%s.%s.png"%(self.name,b,inst), bbox_inches='tight')
                return -1
            else:
                print >>logoutput, "%5.3f %5.3f "%( self.stats[b].dm15lin, self.stats[b].maxjd[0]+2453000.0 )
        except:
            try:
                print >>logoutput, "%5.3f %5.3f " %(self.stats[b].dm15lin, self.stats[b].maxjd[0]+2453000.0)
            except:
                print "could not output to log"
                print self.stats[b].dm15lin, self.stats[b].maxjd[0]+2453000.0         

        return 0

