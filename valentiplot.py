import numpy as np
import glob, pickle
import os,inspect,sys
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]) + "/templates")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
#cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]) + "//../")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)

from snclasses import *
from templutils import *
import optparse
import readinfofile as ri


#pl.ion()
parser = optparse.OptionParser(usage="readlcvV.py snname --yoff yoffset", conflict_handler="resolve")
parser.add_option('--yoff', default=0.5, type="float",
                  help='y offset')
parser.add_option('--offdb', default=False, action="store_true",
                  help='offset from database')
parser.add_option('--locdb', default=False, action="store_true",
                  help='location from database')
parser.add_option('--bin', default=None, type="int",
                  help='bin size for step plot')
parser.add_option('--noylabel', default=False, action="store_true",
                  help='')
options,  args = parser.parse_args()

locs={'04gq':4,'04gt':2,'05az':2,'05bf':2,'05eo':4,'06jc':3, '04gk':2, '07i':4, '07d':4, '05bf':2}
yoffsets={'01gd':1,'04aw':1.0,'04fe':1.3,'04gk':1.7,'04gq':0.1,'04gt':1.2,'05az':0.7,'05bf':1.5,'05eo':1.0, '05hg':0.7,'05mf':0,'06el':0.7,'06f':1.3,'06t':1.0,'06jc':1.7,'07bg':1.0,'06fo':0.1,'07ce':0.1,'07gr':0.1,'07uy':0.2,'07d':1.35,'08cw':1.0, '05nb':1.0,'05kz':0.8, '08d':0, '07i':0.5}
photcodes = {'U':('01','06'),'B':('02','07'),'V':('03','08'),'R':('04','09'),'I':('05','0a'),'r':('13','0b'),'i':('14','0c'), 'H':('H','H'), 'J':('J','J'),'K':('K','K'), 'u':('15','15')}
boffsets={'U':2,'u':2,'B':1,'V':0,'R':-1,'I':-2,'r':-1,'i':-2, 'J':-3,'H':-4,'K':-5}


su=setupvars()
typecount={}
for k in su.mytypecolors.keys():
     typecount[k]=0 

try:
     os.environ['SESNPATH']
except KeyError:
     print "must set environmental variable SESNPATH"
     sys.exit()
if len(args)>0:
     fall = glob.glob(os.environ['SESNPATH']+"/finalphot/*"+args[0]+".*[cf]")
else: fall=glob.glob(os.environ['SESNPATH']+"/finalphot/s*[cf]")

Rs={'R':[],'DR15':[], 'dR':[],'dDR15':[], 'type':[]}
Vmax,sntype=ri.readinfofile()
pl.figure()

for i,f in enumerate(fall):
     myebmv=0
     legendloc=1
     thissn=mysn(f, verbose="True")
     fnir = True

     thisyoff=options.yoff
     noylabel=options.noylabel

     lc,flux,dflux, snname = thissn.loadsn(f,fnir, verbose=True)
     if not snname.lower() in Vmax.keys():
          if '05eo' not in snname.lower():
               print "skipping object ", snname.lower()
               continue
          else:
               sntype[snname.lower()]='Ia'
               Vmax[snname.lower()]='<0.000'

#     thissn.readinfofileall(verbose=False, earliest=False, loose=False)
     for k in thissn.metadata.keys():
            print "here",k,"\t",thissn.metadata[k]
     thissn.setsn(sntype[snname.lower()],Vmax[snname.lower()])
     thissn.setphot()
     thissn.getphot()
     pl.clf()
     print "here1"
     if 1:
#     try:
          thissn.setsnabsR()
          if not is_empty(thissn.Rmax) and thissn.Rmax['mjd']>0:
               print "magsanddm15s", thissn.name,thissn.Rmax['absmag'],thissn.Rmax['dmag'],thissn.Rmax['dm15'], thissn.Rmax['ddm15'] , thissn.type
               pl.errorbar(thissn.Rmax['mjd'],thissn.Rmax['mag'],yerr=thissn.Rmax['dmag'],fmt='r*')
               pl.errorbar(thissn.Rmax['mjd']+15,thissn.Rmax['mag']-thissn.Rmax['dm15'], thissn.Rmax['ddm15'],fmt='r*')
               try:
                    if thissn.filters['R']==0:
                         pl.ylim(max(pl.ylim()[0],max(thissn.photometry['r']['mag']+0.2))+0.2,min(pl.ylim()[1],min(thissn.photometry['r']['mag']-0.2)))
                    else:
                         pl.ylim(max(pl.ylim[0],max(thissn.photometry['R']['mag']+0.2))+0.2,min(pl.ylim()[1],min(thissn.photometry['r']['mag']-0.2)))
               except: pass
               pl.show()
               raw_input('does this look ok? (0 no, 1 yes)')
               try:
                    mynp=int(raw_input('Input:'))
               except ValueError:
                    print "Not a number"
                    continue
#                    sys.exit()

               if mynp==0:
                    continue
               Rs['type'].append(thissn.type)
               Rs['R'].append(thissn.Rmax['absmag'])
               Rs['dR'].append(thissn.Rmax['dmag'])
               Rs['DR15'].append(thissn.Rmax['dm15'])
               Rs['dDR15'].append(thissn.Rmax['ddm15'])
               #     except: pass
     print thissn.printtype()
     
Rs['DR15']=np.array(Rs['DR15'])
Rs['dDR15']=np.array(Rs['dDR15'])
Rs['R']=np.array(Rs['R'])
Rs['dR']=np.array(Rs['dR'])

for i in range(len(Rs['R'])):
     print Rs['DR15'][i],
     print Rs['R'][i],
     print Rs['dR'][i],
     print Rs['dDR15'][i],
     print Rs['type'][i], Rs['type'][i]=='Ib', Rs['type'][i]=='Ic'
pl.figure()

print 
Ibs,=np.where(np.array(Rs['type'])=='Ib')#[Rs['type']=='Ib']
Ics,=np.where(np.array(Rs['type'])=='Ic')#[Rs['type']=='Ib']
print Ibs,Ics
pl.errorbar(-Rs['DR15'][Ibs],Rs['R'][Ibs],yerr=Rs['dR'][Ibs],xerr=Rs['dDR15'][Ibs], fmt='bs')
pl.errorbar(-Rs['DR15'][Ics],Rs['R'][Ics],yerr=Rs['dR'][Ics],xerr=Rs['dDR15'][Ics], fmt='rs')
pl.ylim(-15,-21)
#pl.xlim(0,2)
pl.show()


     
     
     

