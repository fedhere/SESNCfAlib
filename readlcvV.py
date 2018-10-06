import numpy as np
import glob, pickle
import os,inspect,sys
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( ins
                                                                             pect.currentframe() ))[0]) + "/templates")
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

PHOTTABLE=False
#PHOTTABLE=True
COLOR=False
#COLOR=True
ALLCOLOR=False     
#ALLCOLOR=True
try:
     os.environ['SESNPATH']
except KeyError:
     print "must set environmental variable SESNPATH"
     sys.exit()
if len(args)>0:
     fall = glob.glob(os.environ['SESNPATH']+"/finalphot/*"+args[0]+".*[cf]")
else: fall=glob.glob(os.environ['SESNPATH']+"/finalphot/s*[cf]")

#print fall,os.environ['SESNPATH']+"/finalphot/*"+args[0]+"*.[cf]"

#sys.exit()
#fall=fall[:3]
Vmax,sntype=ri.readinfofile()
if COLOR: 
     colorfigs={}
     print "remove files if exists"
#     for ckey in su.cs.iterkeys():
#          cf=open(ckey+".dat","w")##.upper()+".dat", "w")
#          cf.close()
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

     fig = pl.figure()
     pl.errorbars(thissn.photometry['B']['phase'],
                  thissn.photometry['B']['mag'],
                  yerr = thissn.photometry['B']['dmag'],
                  color = 'k', fmt = '.')
     gpphot(self, 'B', fig)
     
#     if thissn.nomaxdate:
#          continue
     if thissn.filters['K']>0:
          print "cheap hack"
          thisyoff+=0.7

     if PHOTTABLE:
          thissn.printsn_fitstable()
#          thissn.printsn_textable(photometry=True, fout=snname+".phot.tex")        
          continue
     try:
          thissn.setsnabsR()
          print "magsanddm15s", thissn.name,thissn.Rmax['absmag'],thissn.Rmax['dmag'],thissn.Rmax['dm15'], thissn.Rmax['ddm15'] , thissn.sntype
          #lcall.append(thissn.photometry['V'])
          #lcallnames.append(thissn.name)
     except: pass
     

     #continue
     
     maxmag = -25
     minmag = 200
     mint=25555555.0
     maxt=-2555555.0
     lcall.apend(thissn.photometry['V'])
x     ylabel=""
#     thissn.getphot(0)
     params = {'legend.fontsize': 12,
               'legend.linewidth': 1,
               'legend.numpoints':1,
               'legend.handletextpad':0.001,
}
     pl.rcParams.update(params)

     if COLOR:
          for snoff in ebmvs.iterkeys():
               if thissn.name.endswith((snoff.strip()).lower()):
                    myebmv=ebmvs[snoff]

          thissn.getphot(myebmv)
          thissn.getcolors()
          for ci,ckey in enumerate(su.cs.iterkeys()):
               cf=open(ckey+".dat","a")#ckey.upper()+".dat", "a")
               if thissn.getmaxcolors(ckey) == -1:
                    continue

               print >>cf,snname, ckey, thissn.maxcolors[ckey]['epoch'],thissn.maxcolors[ckey]['color'],thissn.maxcolors[ckey]['dcolor'] ,
               tmp=thissn.getepochcolors(ckey,50)
               print >>cf, tmp[0],tmp[1],tmp[2],
               tmp=thissn.getepochcolors(ckey,100)
               print >>cf, tmp[0],tmp[1],tmp[2]
               cf.close()

               fout = f.split("/")[-1].replace("slc.","").split('.')[0]+"."+ckey+".dat"
               fileout = open(fout,"w")          
               thissn.printsn(photometry=False, cband = ckey,color=True, fout=fout)
               fig=figure(su.cs[ckey]+1000)
               pl.ylabel(ckey)
               pl.errorbar(thissn.colors[ckey]['mjd'],thissn.colors[ckey]['mag'],fmt='o',yerr=thissn.colors[ckey]['dmag'], label=snname)
               pl.xlabel("phase (days)")               

          if '06jc' in thissn.name:
               thissn.plotsn(photometry=False,color=True, save=True,fig=i, ylim=(maxmag,minmag), xlim=(mint-10,maxt+10), relim=False, offsets=True, ylabel=ylabel,  aspect=0.5, nbins=options.bin, singleplot=True, noylabel=noylabel)
          else:
               thissn.plotsn(photometry=False,color=True, save=True,fig=i, ylim=(maxmag,minmag), xlim=(mint-10,maxt+10), relim=False, offsets=True, ylabel=ylabel,  aspect=0.5, nbins=options.bin, singleplot=False,noylabel=noylabel)
          pl.savefig(f.split("/")[-1].replace("slc.","").split('.')[0]+"_color.png", bbox_inches="tight", dpi=350)
          continue

     else:
          thissn.printsn(photometry=True)
          for b in photcodes.iterkeys():
               myphotcode=su.photcodes[b]
               fout = f.split("/")[-1].replace("slc.","").split('.')[0]+"."+b+".dat"
               fout = fout.replace('.i.','.ip.').replace(".r.",".rp.")
               fileout = open(fout,"w")
               thissn.printsn(photometry=True, band=b, fout=fout)
               
               if not thissn.stats[b].maglim[1] == 0:
                    maxmag = max(maxmag, thissn.stats[b].maglim[1]+boffsets[b]+1)
               if not thissn.stats[b].tlim[0] == 0:
                    mint = min(thissn.stats[b].tlim[0],mint)     
               maxt = max(thissn.stats[b].tlim[1],maxt)     
          if options.offdb :
               for snoff in yoffsets.iterkeys():
                    if thissn.name.endswith(snoff.strip()):
                         thisyoff=yoffsets[snoff]
          if options.locdb :
               for snoff in locs.iterkeys():
                    if thissn.name.endswith(snoff.strip()):
                         legendloc=locs[snoff]

          if thissn.stats['V'].maglim[0] >0:
               minmag= thissn.stats['V'].maglim[0]-thisyoff
          nopt=0
          for b in ['U','u','B','V','R','r','I','i']:
               nopt+=thissn.filters[b]         
      
          for b in ['U','u','B','V','R','r','I','i','J','H','K']:
               if not thissn.stats[b].maglim[0] == 0:
                    minmag= min(minmag, thissn.stats[b].maglim[0]+boffsets[b]-thisyoff)
               if  thissn.filters[b]>0:
                    ylabel+=b+"+%d"%boffsets[b]+", "
          if nopt == 0:
               maxmag +=1
          ylabel=ylabel[:-2].replace("+-","-").replace('r','r\'').replace('i','i\'').replace('u','u\'')+" [mag]"
          print "maxs:",mint,maxt, maxmag, minmag
          if minmag ==0 or minmag < 0:
               minmag = 7
          fig=pl.figure(i,figsize=(8,16))
          if '06ep' in thissn.name:
               thissn.plotsn(photometry=True,show=False, fig=i, ylim=(maxmag,minmag), xlim=(mint-10,maxt+3), relim=False, offsets=True, ylabel=ylabel, aspect=0.5, Vmax=False, legendloc=legendloc,noylabel=noylabel, save=True)
          elif '07ke' in thissn.name:
               thissn.plotsn(photometry=True,show=False, fig=i, ylim=(maxmag,minmag), xlim=(mint-8,maxt+10), relim=False, offsets=True, ylabel=ylabel, aspect=0.5, Vmax=False, legendloc=legendloc,noylabel=noylabel, save=True)
          else:
               thissn.plotsn(photometry=True,show=False, fig=i, ylim=(maxmag,minmag), xlim=(mint-10,maxt+10), relim=False, offsets=True, ylabel=ylabel, aspect=0.5, Vmax=False, legendloc=legendloc,noylabel=noylabel, save=True)
          #     extent=fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
          #pl.savefig(f.split("/")[-1].replace("slc.","").split('.')[0]+"_UBVRIriHJK.pdf")#, bbox_inches="tight")
          os.system("convert -resize 50\% "+f.split("/")[-1].replace("slc.","").split('.')[0]+"_UBVRIriHJK.png " +f.split("/")[-1].replace("slc.","").split('.')[0]+"_thumb.png")
     
if ALLCOLOR:

     for ci,ckey in enumerate(su.cs.keys()[:-1]):
          fig=figure(su.cs[ckey]+1000)
          
          pl.savefig(ckey+"_allsn.png")#, bbox_inches='tight', dpi=150)


     '''
     minmag= thissn.stats['V'].maglim[0]
     if not minmag ==0: minmag=minmag-3.2

     else:
          for b in ['J','H','K']:
               if minmag ==0:
                    minmag= thissn.stats[b].maglim[0]
                    if not minmag == 0:
                         minmag+=boffets[b]-1.4

     for b in ['J','H','K']:
          if  thissn.filters[b]>0:
               ylabel+=b+"+%d"%boffsets[b]+", "
          print ylabel               
     ylabel=ylabel[:-2].replace("+-","-")+" [mag]"
     print mint,maxt, maxmag, minmag

     thissn.plotsn(photometry=True,show=True, fig=i, ylim=(maxmag,minmag), xlim=(mint-5,maxt+5), relim=False, offsets=True, ylabel=ylabel, save=False, nir = True,aspect=0.5)
     pl.savefig(thissn.name+".HJK.png")
     '''

          
