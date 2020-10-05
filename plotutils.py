import pylab as pl
import numpy as np
import os
import json

def adjustFigAspect(fig,aspect=1):
    '''
    Adjust the subplot parameters so that the figure has the correct
    aspect ratio.
    '''
    xsize,ysize = fig.get_size_inches()
    minsize = min(xsize,ysize)
    xlim = .4*minsize/xsize
    ylim = .4*minsize/ysize
    if aspect < 1:
        xlim *= aspect
    else:
        ylim /= aspect
    fig.subplots_adjust(left=.5-xlim,
                        right=.5+xlim,
                        bottom=.5-ylim,
                        top=.5+ylim)

def myopenfig(fignumber, xlim, ylim=(23,10)):
      figcounter=1
      fig = pl.figure(fignumber)
      pl.ylim(ylim)
      return fig

def myplot_txtcolumn(x,y,dy,labels,myfig):
    for label in labels:
#          print label
          pl.text(x,y, label, ha='left', fontsize=10,transform=myfig.transFigure) 
          y=y-dy
        

def myplot_setlabel(xlabel=None, ylabel=None, title=None,
                    label=None, xy=(0,0), ax=None,
                    labsize=15, rightticks=False, verbose=False):
    import matplotlib as mpl
    mpl.rcParams['font.size'] = labsize+0.
#    mpl.rcParams['font.family'] = 'serif'# New Roman'
    mpl.rcParams['font.serif'] = 'Bitstream Vera Serif'

    mpl.rcParams['axes.labelsize'] = labsize+1.
    mpl.rcParams['xtick.labelsize'] = labsize+0.
    mpl.rcParams['ytick.labelsize'] = labsize+0.

    if label:
        if verbose:
            print ("######################## LABELS HERE ##########################")
        
        if xy==(0, 0):
            xy=(0.22, 0.20)
        if not ax:
            print ("WARNING: no axix, cannot place label")
            pl.figtext(xy[0], xy[1], label, fontsize=labsize)
            pl.xlabel(xlabel, fontsize=labsize+1)
            pl.ylabel(ylabel, fontsize=labsize+1)            
        else:
            pl.text(xy[0], xy[1], label, transform=ax.transAxes, fontsize=labsize)
            if rightticks:
                ax.yaxis.tick_right()
            ax.set_xlabel(xlabel, fontsize=labsize+1)
            ax.set_ylabel(ylabel, fontsize=labsize+1)
    if title:
        _ = pl.title(title)

def myplot_err(x, y, yerr=None, xerr=None, xlim=None, ylim=None,
               symbol=None, alpha=0.5, offset=0, fig = None, fcolor=None,
               ms=7, settopx=False, markeredgewidth=1, litsn=None, ax=None):

#    print "litsn", litsn
    import matplotlib as mpl
    mpl.rcParams['figure.dpi'] = 150 
    if symbol:
        color=symbol[:-1]
        marker=symbol[-1]
    else:
        color = 'blue'
        marker='o'
        fcolor=color
    if not fig:
        fig = pl.figure()
        ax = pl.add_subplot(111)
    #print fig
#    print "from myplot_err:",symbol,xlim,ylim
    if xlim :
        ax.set_xlim(xlim)
    if ylim :
        ax.set_ylim(ylim)
    #print xlim,ylim
#,x
    if not isinstance(litsn, np.ndarray):
        litsn=[]
    #print "litsn", litsn

    
    if not symbol :
        symbol='ko'
    if yerr is not None:
          #print color
          ax.errorbar(x, np.asarray(y) + offset, yerr=yerr, xerr=xerr,
                      fmt=None, color=color,
                      ecolor=color, alpha=alpha, markeredgewidth=1.0)

    elif xerr is not None:
          ax.errorbar(x, np.asarray(y) + offset, yerr=yerr, xerr=xerr,
                      fmt=None, color=color,
                      ecolor=color, alpha=alpha, markeredgewidth=1.0)

    thisp = ax.plot(x, np.asarray(y) + offset, ".", marker=marker,
                    alpha=alpha, markersize=ms,
                    markerfacecolor=fcolor, mec=color,
                    markeredgewidth=markeredgewidth)
#    print litsn

    ax.plot(x[litsn], np.asarray(y[litsn]) + offset, ".",
            marker=marker, alpha=alpha, markersize=ms,
            markerfacecolor=fcolor, mec='slategrey',
            markeredgewidth=1)
    return thisp
    #return #pl.plot(x,y+offset,symbol,alpha=alpha, markersize=8)

def myplot_hist(x, y, xlim=None, ylim=None, symbol=None, alpha=1, offset=0,
                fig = None, fcolor=None, ax=None, nbins=None):

    if symbol:
        color=symbol[:-1]
        marker=symbol[-1]
        if not fcolor:
            fcolor=color
    else:
        color = 'blue'
        marker='o'
        fcolor=color
    if fig:
        pl.figure(fig)
#    print "from myplot_err:",symbol,xlim,ylim
    if xlim :
        pl.xlim(xlim)
    if ylim :
        pl.ylim(ylim)
    #print xlim,ylim,x,y
    if not symbol :
        symbol='ko'
    #print nbins
    if nbins is None:
        nbins=int((xlim[1]-xlim[0])/10)
    else:
        nbins=int((xlim[1]-xlim[0])/nbins)
    print (nbins)
    print (xlim[0],xlim[1], np.asarray(x),np.asarray(y),nbins,ax)
    print ("\n\n\n")
    
    X,Y,Ystd=binMeanStdPlot(np.asarray(x),np.asarray(y),numBins=nbins,xmin=xlim[0],xmax=xlim[1], ax=ax)
    
    pl.errorbar(X,Y+offset, yerr=Ystd, fmt=None,color=color, ecolor=color,alpha=alpha)
    pl.errorbar(X,Y+offset, yerr=Ystd, fmt='.',color=color, ecolor=color,alpha=alpha)
    try:
        binsz=X[1]-X[0]
    except:
        binsz=0
    newx=[]
    newy=[]
    newx=[newx+[x,x] for x in X-binsz/2]
    newx =np.asarray(newx).flatten()[1:]

    newy=[newy+[y,y] for y in (Y)]
    newy=np.asarray(newy).flatten()
    newx=np.insert(newx,len(newx),newx[-1]+binsz)
    return pl.step(newx,np.asarray(newy)+offset,"",alpha=alpha,color=color)

    #return #pl.plot(x,y+offset,symbol,alpha=alpha, markersize=8)

def myplotarrow(x,y,label, dx=0,dy=+0.3, color='k'):
#    ax.annotate(label, xy=(x, y), xytext=(x+dx, y+dy),
#                arrowprops=dict(facecolor=color, shrink=0.05),)
    pl.arrow(x,y, dx ,dy, length_includes_head=True, color=color)
    pl.text(x,y+(-2.0*dy), label, ha='center', fontsize=10, color=color)#,transform=myfig.transFigure) 
#    pl.arrow(maxjd+15, maxflux-1, 0, +0.5, length_includes_head=True)


def binMeanPlot(X,Y,numBins=8,xmin=None,xmax=None, binsize=None):
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    if not binsize is None:
        numBins=int((X.max()-X.min())/binsize)
    bins = np.linspace(xmin,xmax,numBins+1)
#    print bins,Y

    YY = np.array([nanmean(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    YYmedian = np.array([nanmedian(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    YYstd = np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    return bins[:-1]+(bins[1]-bins[0])*0.5,YY,YYmedian,YYstd
 
def binMeanStdPlot(X,Y,ax=None,numBins=8,xmin=None,xmax=None, binsize=None):
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    if binsize:
        numBins=int((xmax-xmin)/binsize)

    bins = np.linspace(xmin,xmax,numBins+1)
    XX = np.array([np.mean((bins[binInd], bins[binInd+1])) for binInd in range(numBins)])
    YY = np.array([np.mean(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    #XX[np.isnan(YY)]=np.nan
    YYstd = np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    return XX, YY, YYstd

def binWMeanStdPlot(X,Y,std,ax=None,numBins=8,xmin=None,xmax=None, binsize=None):
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    if binsize:
        numBins=int(((xmax-xmin)/binsize)+0.5)
    else:
        binsize=(xmin-xmax)/numBins
    bins = np.arange(xmin,xmax+binsize,binsize)
    
    XX=[]
    YY=[]
    for binInd in range(numBins):
        XX.append(np.mean((bins[binInd], bins[binInd+1])) )
        thisY=Y[(X > bins[binInd]) & (X <= bins[binInd+1])]
        print ("thislen",XX[-1],len(thisY))
        if len(thisY)>0:            
            YY.append(np.average(thisY, weights=1.0/(np.array(std[(X > bins[binInd]) & (X <= bins[binInd+1])]))**2))
            print (thisY,std[(X > bins[binInd]) & (X <= bins[binInd+1])],
                   1.0/(np.array(std[(X > bins[binInd]) & (X <= bins[binInd+1])]))**2,
                   np.average(thisY, weights=1.0/np.array(std[(X > bins[binInd]) & (X <= bins[binInd+1])])**2),
                   np.average(thisY))

        else:
            YY.append(float('nan'))
        print ("\n\n")

    print ("\n\n\n",YY[-1],"\n\n\n")

    YYstd = np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    return XX, np.array(YY), YYstd

def binMedianStdPlot(X,Y,ax=None,numBins=8,xmin=None,xmax=None,binsize=None):
    if xmin is None:
        xmin = X.min()
    if xmax is None:
        xmax = X.max()
    if binsize:
        numBins=int(((xmax-xmin)/binsize)+0.5)
    else:
        binsize=(xmin-xmax)/numBins
    bins = np.arange(xmin,xmax+binsize,binsize)
    print (xmin,xmax,bins, numBins)
    XX = np.array([np.mean((bins[binInd], bins[binInd+1])) for binInd in range(numBins)])
    YY = np.array([np.median(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])
    #XX[np.isnan(YY)]=np.nan
    YYstd = np.array([np.std(Y[(X > bins[binInd]) & (X <= bins[binInd+1])]) for binInd in range(numBins)])

    return XX, YY, YYstd




'''
    if ax is None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)

    lineHandles = ax.plot(XX,YY)
    return lineHandles[0], XX, YY


    if ax is None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)
    lineHandles = ax.plot(XX,YY)
patchHandle = ax.fill_between(XX[~np.isnan(YY)],YY[~npmyplo.isnan(YY)]-YYstd[~np.isnan(YY)],YY[~np.isnan(YY)]+YYstd[~np.isnan(YY)])
    patchHandle.set_facecolor([.8, .8, .8])
    patchHandle.set_edgecolor('none')
    return lineHandles[0], patchHandle, '''


def plotUberTemplates(sne=None, bs=None):
    import pickle as pkl
    import matplotlib.gridspec as gridspec
    import os
    from snclasses import setupvars as stp
    import pylabsetup

    from matplotlib.ticker import MultipleLocator, FormatStrFormatter    
    #pl.rcParams['font.size']=20
    
    su = stp()
    d11V = np.loadtxt(os.environ['SESNCFAlib'] + "/D11templates/V_template.txt",
                      skiprows=4, unpack=True)# names=["t", "mag"])
    #print (d11V)

    d11R = np.loadtxt(os.environ['SESNCFAlib'] + "/D11templates/R_template.txt",
                      skiprows=4, unpack=True)# names=["t", "mag"])


    T15 = np.loadtxt(os.environ['SESNCFAlib'] + "/T15templates.csv",
                      skiprows=1, unpack=True)# names=["t", "mag"])
     
    T15[0][T15[0] == -99] = np.nan
    T15[1][T15[1] == -99] = np.nan
    T15[2][T15[1] == -99] = np.nan

    for i in [1,4,7,10,13]:
        T15[i] = np.nanmax(T15[i]) - T15[i]
    
    fig = pl.figure(figsize=(30,30))
    if not bs:
        bs = ['U','u','B','V','R','r','g','I','i','J','H','K','w1','w2','m2']
    gs = gridspec.GridSpec(len(bs)/3 + 1, 3)
    gs.update(wspace=0.0)
    gs.update(hspace=0.0)    
    print (gs[-1,0])
    vinfile =  os.environ['SESNCFAlib'] + "/templatesout/UberTemplate_V.pkl" 

    vut = pkl.load(open(vinfile, 'rb'))
    vinfile =  os.environ['SESNCFAlib'] + "/templatesout/UberTemplate_V40.pkl" 

    vut40 = pkl.load(open(vinfile, 'rb'))    
    rinfile =  os.environ['SESNCFAlib'] + "/templatesout/UberTemplate_R40.pkl" 

    rut40 = pkl.load(open(rinfile, 'rb'))
    
    for i,b in enumerate(bs):
        print (b)
        if b == '': continue
        ax = fig.add_subplot(gs[i/3,int(i%3)])
        infile = os.environ['SESNCFAlib'] + "/templatesout/UberTemplate_%s.pkl" %\
                   (b + 'p' if b in ['u', 'r', 'i']
                                            else b)
        ut = pkl.load(open(infile, 'rb'))
        ind = (ut['std']>0)
        
        ax.plot(ut['epochs'], ut['med'], 'k-', alpha=0.7, lw=1)
        ax.plot(ut['epochs'][ind], ut['mu'][ind], '-',
                color = su.mycolors[b],lw=2)
        ax.text(100, 0, b,va='center', ha='right')

        ax.fill_between(ut['epochs'][ind],
                        ut['pc25'][:-1][ind], ut['pc75'][:-1][ind],
                        #ut['mu'][ind]-ut['std'][ind],
                        #ut['mu'][ind]+ut['std'][ind],
                        color = su.mycolors[b], alpha=0.3)
        ax.fill_between(ut['epochs'][ind], ut['mu'][ind]-ut['wstd'][:-1][ind],
                        ut['mu'][ind]+ut['wstd'][:-1][ind],
                        color = su.mycolors[b], alpha=0.6)        
        ax.plot(vut['epochs'], vut['mu'], '--', color = 'DarkGreen', lw=1)
        ax.legend(loc=1, frameon=False, fontsize=10)
        ax.tick_params(axis="both", which="both", bottom="on", top="on",  
                       left="on", right="on")
        ax.set_xlim(-27,110)
        ax.set_ylim(2.9,-1)
        #ax.grid(True)

        ax.minorticks_on()
        ax.tick_params('both', length=10, width=1, which='major')
        ax.tick_params('both', length=5, width=1, which='minor')
        ax.set_yticks([0,2])
        majorFormatter = FormatStrFormatter('%d')
        minorLocator = MultipleLocator(0.5)
        #ax.yaxis.set_minor_locator(minorLocator)

        if int(i%3) > 0:
            ax.set_yticks([0, 2])            
            ax.set_yticklabels([' ', ' '])
        else:
            ax.set_yticks([0, 2], ['0', '2'])
            ax.yaxis.set_minor_locator(minorLocator)            
            
        if i / 3 < (len(bs) - 1) / 3:
            ax.set_xticks([0, 50, 100])
            ax.set_xticklabels([' ',  ' ', ' '])
        else:
            ax.set_xticks([0, 50, 100])
            ax.set_xticklabels(['0', '50', '100'])

    pl.text(-325, -10, "relative magnitude", rotation=90, fontsize=20)
    pl.text(-140, 4.5, "phase (days)",fontsize=20)
    pl.savefig(os.environ['SESNCFAlib'] +
               "/templatesout/ubertemplates_compare.pdf")
    os.system("pdfcrop " + 
              os.environ['SESNCFAlib'] +
              "/templatesout/ubertemplates_compare.pdf " +
              os.environ['DB'] + "/papers/SESNtemplates.working/figs/ubertemplates_compare.pdf")
    
    pl.close(fig)
    #return 0
    def setticks(ax):
        ax.minorticks_on()
        majorLocator = MultipleLocator(2)
        majorFormatter = FormatStrFormatter('%d')
        minorLocator = MultipleLocator(0.5)
        ax.yaxis.set_major_locator(majorLocator)            
        ax.yaxis.set_minor_locator(minorLocator)            
       
        ax.tick_params('both', length=10, width=1, which='major')
        ax.tick_params('both', length=5, width=1, which='minor')
    #comparing w D11 V and R band
    fig = pl.figure(figsize=(15,15))
    gs = gridspec.GridSpec(2,1)
    gs.update(hspace=0.0)    

    ax = fig.add_subplot(gs[0,0])
    ind = vut40['epochs']<40 
    ax.plot(vut40['epochs'][ind], vut40['mu'][ind], '-',
            color = su.mycolors['V'],lw=2, label="V")
    ax.plot(vut40['epochs'][ind], vut40['med'][ind], 'k-',
            alpha=0.5, lw=1, label="median")
    ax.fill_between(vut40['epochs'][ind],
                    vut40['pc25'][:-1][ind], vut40['pc75'][:-1][ind],
                    #(vut40['mu']-vut40['std'])[ind],
                    #(vut40['mu']+vut40['std'])[ind],
                    color = su.mycolors['V'], alpha=0.3)
    ax.fill_between(vut40['epochs'][ind], (vut40['mu']-vut40['wstd'][:-1])[ind],
                    (vut40['mu']+vut40['wstd'][:-1])[ind],
                    color = su.mycolors['V'], alpha=0.6)
    ax.plot(d11V[0],d11V[1], '-', color="SteelBlue", label='D11 V')
    ax.set_ylim([3.2, -1.2])
    
    ax.set_xlabel("phase (days)")
    ax.set_ylabel("relative magnitude")    
    ax.tick_params(axis="both", which="both", bottom="on", top="on",  
                   left="on", right="on")
    setticks(ax)
    ax.legend()

    
    ax = fig.add_subplot(gs[1,0])
    ax.plot(rut40['epochs'][ind], rut40['mu'][ind], '-',
            color = su.mycolors['R'],lw=2, label="R")
    ax.plot(rut40['epochs'][ind], rut40['med'][ind], 'k-',
            alpha=0.5, lw=1, label="median")
    ax.fill_between(rut40['epochs'][ind],
                    rut40['pc25'][:-1][ind], rut40['pc75'][:-1][ind],
                    #(rut40['mu']-rut40['std'])[ind],
                    #(rut40['mu']+rut40['std'])[ind],
                    color = su.mycolors['R'], alpha=0.3)
    ax.fill_between(rut40['epochs'][ind],
                    (rut40['mu']-rut40['wstd'][:-1])[ind],
                    (rut40['mu']+rut40['wstd'][:-1])[ind],
                    color = su.mycolors['R'], alpha=0.6)    

    ax.plot(d11R[0],d11R[1], '-', color="SteelBlue", label='D11 R')
    ax.set_xlabel("phase (days)", fontsize=21)
    ax.set_ylabel("relative magnitude", fontsize=21)        
    ax.set_ylim([4.2, -1.2])
    ax.tick_params(axis="both", which="both", bottom="on", top="on",  
                   left="on", right="on")
    setticks(ax)
    ax.legend()
    pl.savefig(os.environ['SESNCFAlib'] +
               "/templatesout/ubertemplates_compareD11.pdf")

    os.system("pdfcrop " + 
              os.environ['SESNCFAlib'] +
              "/templatesout/ubertemplates_compareD11.pdf " +
              os.environ['DB'] +
              "/papers/SESNtemplates.working/figs/ubertemplates_compareD11.pdf")
    
    #comparing w T15 u, r and iband
    uinfile =  os.environ['SESNCFAlib'] + "/templatesout/UberTemplate_up.pkl" 

    uut50 = pkl.load(open(uinfile, 'rb'))
    rinfile =  os.environ['SESNCFAlib'] + "/templatesout/UberTemplate_rp.pkl" 

    rut50 = pkl.load(open(rinfile, 'rb'))    
    iinfile =  os.environ['SESNCFAlib'] + "/templatesout/UberTemplate_ip.pkl" 

    iut50 = pkl.load(open(iinfile, 'rb'))

    fig = pl.figure(figsize=(9,16))
    gs = gridspec.GridSpec(3,1)
    gs.update(hspace=0.0)    

    ax = fig.add_subplot(gs[0,0])
    ind = uut50['epochs']<50
    
    indu = ind * (uut50['epochs'] < 35) * (uut50['epochs'] > -15)


    ax.plot(uut50['epochs'][indu], uut50['mu'][indu], '-',
            color = su.mycolors['u'],lw=2, label="u'")
    ax.plot(uut50['epochs'][indu], uut50['med'][indu], 'k-',
            alpha=0.5, lw=1, label="median")
    ax.fill_between(uut50['epochs'][ind],
                    uut50['pc25'][:-1][ind], uut50['pc75'][:-1][ind],
                    #(uut50['mu']-uut50['std'])[ind],
                    #(uut50['mu']+uut50['std'])[ind],
                    color = su.mycolors['u'], alpha=0.3)
    ax.fill_between(uut50['epochs'][indu],
                    (uut50['mu']-uut50['wstd'][:-1])[indu],
                    (uut50['mu']+uut50['wstd'][:-1])[indu],
                    color = su.mycolors['u'], alpha=0.6)    
    ax.plot(T15[0],T15[1], '-', color="SteelBlue", label="T15 u'")
    ax.fill_between(T15[0], T15[1]-T15[2], T15[1]+T15[2],
                    color="SteelBlue", alpha=0.5)
    ax.set_ylim([3.2, -1.2])
    ax.set_xlim([-22.6, 53])    
    setticks(ax)
    ax.tick_params(axis="both", which="both", bottom="on", top="on",  
                   left="on", right="on")    
    ax.legend()

    
    ax = fig.add_subplot(gs[1,0])
    ax.plot(rut50['epochs'][ind], rut50['mu'][ind], '-',
            color = su.mycolors['R'],lw=2, label="r'")
    ax.plot(rut50['epochs'][ind], rut50['med'][ind], 'k-',
            alpha=0.5, lw=1, label="median")
    ax.fill_between(rut50['epochs'][ind],
                    rut50['pc25'][:-1][ind], rut50['pc75'][:-1][ind],
                    #(rut50['mu']-rut50['std'])[ind],
                    #(rut50['mu']+rut50['std'])[ind],
                    color = su.mycolors['r'], alpha=0.3)
    ax.fill_between(rut50['epochs'][ind],
                    (rut50['mu']-rut50['wstd'][:-1])[ind],
                    (rut50['mu']+rut50['wstd'][:-1])[ind],
                    color = su.mycolors['r'], alpha=0.6)    

    ax.plot(T15[6],T15[7], '-', color="SteelBlue", label="T15 r'")
    ax.fill_between(T15[6], T15[7]-T15[8], T15[7]+T15[8],
                     color="SteelBlue", alpha=0.5)

    ax.set_ylabel("relative magnitude", fontsize=21)        
    ax.set_ylim([3.2, -1.2])
    ax.set_xlim([-22.6, 53])        
    setticks(ax)
    ax.tick_params(axis="both", which="both", bottom="on", top="on",  
                   left="on", right="on")    
    ax.legend()

    ax = fig.add_subplot(gs[2,0])
    ax.plot(iut50['epochs'][ind], iut50['mu'][ind], '-',
            color = su.mycolors['R'],lw=2, label="i'")
    ax.plot(iut50['epochs'][ind], iut50['med'][ind], 'k-',
            alpha=0.5, lw=1, label="median")
    ax.fill_between(iut50['epochs'][ind],
                    iut50['pc25'][:-1][ind], iut50['pc75'][:-1][ind],
                    #(iut50['mu']-iut50['std'])[ind],
                    #(iut50['mu']+iut50['std'])[ind],
                    color = su.mycolors['i'], alpha=0.3)
    ax.fill_between(iut50['epochs'][ind],
                    (iut50['mu']-iut50['wstd'][:-1])[ind],
                    (iut50['mu']+iut50['wstd'][:-1])[ind],
                    color = su.mycolors['i'], alpha=0.6)    

    ax.plot(T15[9],T15[10], '-', color="SteelBlue", label="T15 i'")
    ax.fill_between(T15[9], T15[10]-T15[11], T15[10]+T15[11],
                      color="SteelBlue", alpha=0.5)

    ax.set_xlabel("phase (days)", fontsize=21)
    ax.set_ylim([3.2, -1.2])
    ax.set_xlim([-22.6, 53])
    setticks(ax)
    ax.tick_params(axis="both", which="both", bottom="on", top="on",  
                   left="on", right="on")
    ax.legend(loc=8)
    pl.savefig(os.environ['SESNCFAlib'] +
               "/templatesout/ubertemplates_compareT15.pdf")

    os.system("pdfcrop " + 
              os.environ['SESNCFAlib'] +
              "/templatesout/ubertemplates_compareT15.pdf " +
              os.environ['DB'] +
              "/papers/SESNtemplates.working/figs/ubertemplates_compareT15.pdf")
    
    #comparing w 03dh 03lw V and I band
    if not sne:
        return ax
    
    snecolors = pkl.load(open('colorSNe.pkl', "rb"))
                        
    iinfile =  os.environ['SESNCFAlib'] + "/templatesout/UberTemplate_I.pkl" 

    iut = pkl.load(open(iinfile, 'rb'))
    binfile =  os.environ['SESNCFAlib'] + "/templatesout/UberTemplate_B.pkl" 

    but = pkl.load(open(binfile, 'rb'))
    rinfile =  os.environ['SESNCFAlib'] + "/templatesout/UberTemplate_R.pkl" 

    rut = pkl.load(open(rinfile, 'rb'))

    fig = pl.figure(figsize=(24,21))
    gs = gridspec.GridSpec(2,2)


    def plottempllcv(ax, ut, b, su, sne, right=False):

        ax.plot(ut['epochs'], ut['mu'], '-',
                color = 'k', lw=2, label=b)
        ax.plot(ut['epochs'], ut['med'], 'k-',
                alpha=0.5, lw=1, label="median")
        ax.fill_between(ut['epochs'],
                        ut['pc25'][:-1], ut['pc75'][:-1],
                        #(ut['mu']-ut['std']),
                        #(ut['mu']+ut['std']),
                        color = su.mycolors[b], alpha=0.3)
        ax.fill_between(ut['epochs'],
                        (ut['mu']-ut['wstd'][:-1]),
                        (ut['mu']+ut['wstd'][:-1]),
                        color = su.mycolors[b], alpha=0.6)

        for sn in sne:
            name = sn.snnameshort
            tp = sn.sntype
            sn = sn.photometry[b]
            if len(sn['phase'])==0:
                continue
            if '03dh' in name:
                ind = (sn['phase']>-1) * (sn['phase']<1)
            else:
                ind = (sn['phase'] < 50)
                if ind.sum() == 0:
                    continue

                phasemin = np.where(sn['mag'] == sn['mag'][ind].min())[0]
                phasemin = phasemin[0]        
                ax.errorbar(sn['phase'] - sn['phase'][phasemin],
                            sn['mag'] - sn['mag'][ind].min(),
                            yerr=sn['dmag'], fmt='-',
                            color = snecolors[name],
                            label=name + " "+ tp)
        
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlabel("phase (days)", fontsize=20)
        ax.set_ylabel("relative magnitude", fontsize=20)
        setticks(ax)
        if right:
            ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position('both')
            ax.yaxis.set_label_position("right")
            
        ax.legend(fontsize=10, loc=4)
        #ax.grid(True)
    ax = fig.add_subplot(gs[0,1])
    plottempllcv(ax, vut, 'V', su, sne, right=True)
    ax.set_xlim(-25,115)
    ax.set_ylim(6.2,-0.7)
    ax = fig.add_subplot(gs[1,1])
    plottempllcv(ax, iut, 'I', su, sne, right=True)
    ax.set_xlim(-25,115)
    ax.set_ylim(6.2,-0.7)    
    ax = fig.add_subplot(gs[0,0])
    plottempllcv(ax, but, 'B', su, sne)
    ax.set_xlim(-25,115)
    ax.set_ylim(6.2,-0.7)    
    ax = fig.add_subplot(gs[1,0])
    plottempllcv(ax, rut, 'R', su, sne)
    ax.set_xlim(-25,115)
    ax.set_ylim(6.2,-0.7)    
    gs.update(hspace=0.02)    
    gs.update(wspace=0.02)    

    pl.savefig(os.environ['SESNCFAlib'] +
               "/templatesout/ubertemplates_compareSNe.pdf")
    os.system("pdfcrop " + 
              os.environ['SESNCFAlib'] +
              "/templatesout/ubertemplates_compareSNe.pdf  " +
              os.environ['DB'] + "/papers/SESNtemplates.working/figs/ubertemplates_compareSNe.pdf")
    return ax

