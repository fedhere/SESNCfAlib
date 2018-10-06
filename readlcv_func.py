import numpy as np
import glob, pickle
import os,inspect,sys
try:
     os.environ['SESNPATH']
     os.environ['SESNCFAlib']

except KeyError:
     print "must set environmental variable SESNPATH and SESNCfAlib"
     sys.exit()

RIri = False

cmd_folder = os.getenv("SESNCFAlib")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
cmd_folder = os.getenv("SESNCFAlib")+"/templates"
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)



from snclasses import *
from templutils import *
import optparse
import readinfofile as ri


def doit(args,options):
     
     locs={'04gq':4,'04gt':2,'05az':2,'05bf':2,'05eo':4,'06jc':3, '04gk':3, '07i':4, '07d':4, '05bf':2}
     yoffsets={'01gd':1,'04aw':1.0,'04fe':1.3,'04gk':1.3,'04gq':0.8,'04gt':1.7,'05az':1.8,'05bf':1.5,'05eo':1.0, '05hg':0.7,'05mf':0.5,'06el':0.7,'06f':1.3,'06t':1.0,'06jc':2.1,'07bg':1.0,'06fo':1.4,'07ce':0.8,'07gr':1.1,'07uy':1.2,'07d':1.35,'08cw':1.0, '05nb':1.0,'05kz':0.8, '08d':0, '07i':1.5, '09er':1.5, '09iz':1.5,'07ke':0.7}
     photcodes = {'U':('01','06'),'B':('02','07'),'V':('03','08'),'R':('04','09'),'I':('05','0a'),'r':('13','0b'),'i':('14','0c'), 'H':('H','H'), 'J':('J','J'),'K':('K','K'), 'u':('15','15')}
     boffsets={'U':2,'u':2,'B':1,'V':0,'R':-1,'I':-2,'r':-1,'i':-2, 'J':-3,'H':-4,'K':-5}

     #mybands = ['V']
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
     lit=False
     #lit=True
     addlit = True
     addlitphot = True
     #Vmax,sntype=ri.readinfofile(verbose=False)

     if COLOR: 
          colorfigs={}
          #     for ckey in su.cs.iterkeys():
          #          cf=open(ckey+".dat","w")##.upper()+".dat", "w")
          #          cf.close()



     ticklabelsz = options['ticklabelsz']
     thisyoff=options['yoff']
     noylabel=options['noylabel']



     for i,f in enumerate(args):
          if '05eo' in f: continue
          myebmv=0
          legendloc=1
          if isinstance (f, basestring):
               thissn=mysn(f, addlit=addlit, quiet=True)
          else:
               thissn=mysn(f[0], addlit=addlit, quiet=True)
          fnir = True
          #print "optical files: ", thissn.optfiles
          lc,flux,dflux, snname = thissn.loadsn2(verbose=False)
          #     lc,flux,dflux, snname = thissn.loadsn(thissn.optfiles, fnir, lit=lit,
          #                                           verbose=True, addlit=addlit,
          #                                           photlit=True)


          #else:
          #     sntype[snname.lower()]='Ia'
          #     Vmax[snname.lower()]b='<0.000'
          Vmax={snname.lower():'<0.000'}
          #sntype={snname.lower():}
          #print "printing sn info"
          #thissn.printsn()

          thissn.readinfofileall(bigfile=False, verbose=False, earliest=False, loose=True) 
          
          #thissn.setsn(sntype[snname.lower()],Vmax[snname.lower()])
          if np.isnan(thissn.Vmax):
               return -1
          thissn.setphot()
          thissn.getphot(RIri=RIri)
          thissn.setphase()
          thissn.printsn()

          if options['hostebmv']:
               if thissn.snnameshort not in su.ebmvhost and\
                  thissn.snnameshort not in su.ebmvcfa:
                    continue
               myebmv = su.ebmvs[thissn.snnameshort]

               try:
                    thisebmv = su.ebmvs[thissn.snnameshort] + \
                               su.ebmvhost[thissn.snnameshort]    
               except KeyError:
                    thisebmv = su.ebmvs[thissn.snnameshort] +  \
                               su.ebmvcfa[thissn.snnameshort]    

               thissn.cleanphot()
               thissn.printsn()
               thissn.getphot(myebmv,RIri=RIri)
               thissn.setphase()


               if options['abs']:
                    try:
                         distpc=float(thissn.metadata['distance Mpc'])*1e6
                    except:
                         print "failed on distance:", snname#, thissn.metadata['distance Mpc']
                         continue

                    dm= 5.0*(np.log10(distpc)-1)
                    thissn.photometry['V']['mag']-=dm

          if thissn.filters['K']>0:
               #print "cheap hack"
               thisyoff+=0.7

          if PHOTTABLE:
               thissn.printsn_fitstable()
               #          thissn.printsn_textable(photometry=True, fout=snname+".phot.tex")        
               continue
          maxmag = -25
          minmag = 200
          mint=25555555.0
          maxt=-2555555.0
          ylabel=""

          params = {'legend.fontsize': 12,
                    #'legend.linewidth': 1,
                    'legend.numpoints':1,
                    'legend.handletextpad':0.001,
                    'xtick.labelsize'  : ticklabelsz,
                    'ytick.labelsize'  : ticklabelsz               
          }
          pl.rcParams.update(params)

          if COLOR:
               #          for snoff in ebmvs.iterkeys():
               #               if thissn.name.endswith((snoff.strip()).lower()):
               #                    myebmv=ebmvs[snoff]
               if not options['hostebmv']:
                    myebmv=su.ebmvs[thissn.snnameshort]
                    thissn.cleanphot()
                    thissn.getphot(myebmv,RIri=RIri)
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
                    if isinstance (f, basestring):
                         fout = f.split("/")[-1].replace("slc.","").split('.')[0]+"."+ckey+".dat"
                    else:
                         fout = f[0].split("/")[-1].replace("slc.","").split('.')[0]+"."+ckey+".dat"
                         
                    fileout = open(fout,"w")          
                    thissn.printsn(photometry=False, cband = ckey,color=True, fout=fout)
                    fig=figure(su.cs[ckey]+1000)
                    pl.ylabel(ckey)
                    pl.errorbar(thissn.colors[ckey]['mjd'],thissn.colors[ckey]['mag'],fmt='o',yerr=thissn.colors[ckey]['dmag'], label=snname)
                    pl.xlabel("phase (days)")               

               if '06jc' in thissn.name:
                    #               thissn.plotsn(photometry=False,color=True, save=True,fig=i, ylim=(maxmag,minmag), xlim=(mint-10,maxt+10), relim=False, offsets=True, ylabel=ylabel,  aspect=0.5, nbins=options.bin, singleplot=True, noylabel=noylabel)
                    _ = thissn.plotsn(photometry=False,color=True, save=True,fig=i, ylim=(maxmag,minmag), xlim=(mint-10,maxt+10+(maxt-mint+20)*0.2), relim=False, offsets=True, ylabel=ylabel,  aspect=0.5, nbins=options['bin'], singleplot=False, noylabel=noylabel, ticklabelsz=ticklabelsz)
               else:
                    _ = thissn.plotsn(photometry=False,color=True, save=True,fig=i, ylim=(maxmag,minmag), xlim=(mint-10,maxt+10+(maxt-mint+20)*0.2), relim=False, offsets=True, ylabel=ylabel,  aspect=0.5, nbins=options['bin'], singleplot=False,noylabel=noylabel, ticklabelsz=ticklabelsz)
               if options['showme'] : pl.show()
               else : pl.savefig(f.split("/")[-1].replace("slc.","").split('.')[0]+"_color.png", bbox_inches="tight", dpi=350)
               continue

          else:
               #thissn.printsn(photometry=True)
               for b in photcodes.iterkeys():
                    if thissn.filters[b] == 0: continue
                    myphotcode=su.photcodes[b]
                    if isinstance (f, basestring):
                         fout = f.split("/")[-1].replace("slc.","").split('.')[0]+"."+b+".dat"
                    else:
                         if not isinstance (f[0], basestring):
                              f = f[0]
                         fout = f[0].split("/")[-1].replace("slc.","").split('.')[0]+"."+b+".dat"                    
                    fout = fout.replace('.i.','.ip.').replace(".r.",".rp.").replace(".u.",".up.")
                    fileout = open(fout,"w")
                    #thissn.printsn(photometry=True, band=b, fout=fout)
                    #if b=='U':
                         #thissn.printsn(photometry=True, band = b)

                    if not thissn.stats[b].maglim[1] == 0:
                         maxmag = max(maxmag, thissn.stats[b].maglim[1]+boffsets[b]+1)
                    if not thissn.stats[b].tlim[0] == 0:
                         mint = min(thissn.stats[b].tlim[0], mint)     
                    maxt = max(thissn.stats[b].tlim[1], maxt)
                    #print "maxt", maxt
               if options['offdb'] :
                    for snoff in yoffsets.iterkeys():
                         if thissn.name.endswith(snoff.strip()):
                              thisyoff=yoffsets[snoff]
                              #print yoffsets[snoff]
               if options['locdb'] :
                    for snoff in locs.iterkeys():
                         if thissn.name.endswith(snoff.strip()):
                              legendloc=locs[snoff]

               if thissn.stats['V'].maglim[0] >0:
                    minmag= thissn.stats['V'].maglim[0]-thisyoff
               nopt=0
               for b in ['U','u','B','V','R','r','I','i']:
                    nopt+=thissn.filters[b]         

               for b in ['U','u','B','V','R','r','I','i','J','H','K']:
                    #print "####\n\n####", thissn.name, b, thissn.filters[b], "####"
                    if not thissn.stats[b].maglim[0] == 0:
                         minmag= min(minmag, thissn.stats[b].maglim[0]+boffsets[b]-thisyoff)
                    if  thissn.filters[b]>0:
                         ylabel+=b+"+%d"%boffsets[b]+", "
                         '''
                         ax = pl.figure().add_subplot(111)
                         
                         ax.errorbar(thissn.photometry[b]['phase'],
                                thissn.photometry[b]['mag'],
                                yerr = thissn.photometry[b]['dmag'],
                                color = 'k', fmt = '.')
                         ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])
                         thissn.gpphot(b, ax=ax)
                         '''          
               if nopt == 0:
                    maxmag +=1
               ylabel=ylabel[:-2].replace("+-","-").replace('r','r\'').\
                       replace('i','i\'').replace('u','u\'') + " [mag]"
               #print "maxs:", mint, maxt, maxmag, minmag
               if minmag == 0 or minmag < 0:
                    minmag = 7
               fig=pl.figure(i, figsize=(8,16))
               
               # individual SN plot setups
               if '06ep' in thissn.name:
                    thissn.plotsn(photometry=True, show=False, fig=i,
                                  ylim=(maxmag,minmag), xlim=(mint-10,maxt+3),  
                                  relim=False, offsets=True, ylabel=ylabel,
                                  aspect=0.5, Vmax=False, legendloc=legendloc,
                                  noylabel=noylabel, save=True, ticklabelsz=ticklabelsz)
               elif '07ke' in thissn.name:
                   thissn.plotsn(photometry=True, show=False, fig=i,
                                  ylim=(maxmag,minmag), xlim=(mint-8,maxt+10), 
                                  relim=False, offsets=True, ylabel=ylabel,
                                  aspect=0.5, Vmax=False, legendloc=legendloc,
                                  noylabel=noylabel, save=True, ticklabelsz=ticklabelsz)
               elif '05az' in thissn.name:
                    thissn.plotsn(photometry=True, show=False, fig=i,
                                  ylim=(maxmag,minmag), xlim=(mint-10,maxt), 
                                  relim=False, offsets=True, ylabel=ylabel,
                                  aspect=0.5, Vmax=False, legendloc=legendloc,
                                  noylabel=noylabel, save=True, ticklabelsz=ticklabelsz)
                    #all other SN plots
               else:
                    #print thissn.fnir
                    thissn.plotsn(photometry=True, fig=i,
                                  ylim=(maxmag,minmag), show=True,
                                  #xlim=(mint-10,maxt+10+(maxt-mint+20)*0.4),
                                  xlim=(thissn.Vmax-22-2400000, thissn.Vmax+17-2400000), 
                                  relim=False, offsets=True, ylabel=ylabel,
                                  aspect=0.5, Vmax=False, legendloc=legendloc,
                                  noylabel=noylabel, save=True, ticklabelsz=ticklabelsz)
                    #     extent=fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                    #pl.savefig(f.split("/")[-1].replace("slc.","").split('.')[0]+"_UBVRIriHJK.pdf")#, bbox_inches="tight")
               try:
                    os.system("convert -resize 50\% "+fout +fout.replace(".png","_thumb.png"))
               except UnboundLocalError: pass
               if options['showme']: pl.show()

     if ALLCOLOR:
          
          for ci,ckey in enumerate(su.cs.keys()[:-1]):
               fig=figure(su.cs[ckey]+1000)
               if options['showme']: pl.show()
               else: pl.savefig(ckey+"_allsn.png")#, bbox_inches='tight', dpi=150)


     return thissn

               
#pl.ion()
if __name__ == '__main__':
     parser = optparse.OptionParser(usage="readlcvV.py snname --yoff yoffset", conflict_handler="resolve")
     parser.add_option('--yoff', default=0.5, type="float",
                       help='y offset')
     parser.add_option('--offdb', default=False, action="store_true",
                       help='offset from database')
     parser.add_option('--locdb', default=False, action="store_true",
                       help='location from database')
     parser.add_option('--bin', default=None, type="int",
                       help='bin size for step plot')
     parser.add_option('--hostebmv', default=False, action="store_true", 
                       help='host ebmv correction using cfa ebmv values')
     parser.add_option('--abs', default=False, action="store_true", 
                       help='abs mag')
     parser.add_option('--noylabel', default=False, action="store_true",
                       help='')
     parser.add_option('--ticklabelsz', default=13, type="float",
                       help='tick size')
     parser.add_option('--showme', default=False, action="store_true",
                       help='')
     
     options,  args = parser.parse_args()
     doit (args, options)
