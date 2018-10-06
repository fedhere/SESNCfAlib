import sys
import os
import glob
import inspect
import pylab as pl
from numpy import *
from scipy import optimize
import pickle
import time
import copy

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]) + "/templates")
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
from templutils import *
import pylabsetup
pl.ion()

#fits the vacca leibundgut model to data:
# a linear decay, with a gaussian peak on top, an exponential rise, and possibly a second gaussian (typically the Ia second bump around phase=25 days

def minfunc(p, y, x, e, secondg, plot=False):
    '''
    p is the parameter list
    if secondg=1: secondgaussian added
    if secondg=0: secondgaussian not    
    parameters are: 
    p[0]=first gaussian normalization (negative if fitting mag)
    p[1]=first gaussian mean
    p[2]=first gaussian sigma
    p[3]=linear decay offset
    p[4]=linear decay slope
    p[5]=exponxential rise slope
    p[6]=exponential zero point
    p[7]=second gaussian normalization (negative if fitting mag)
    p[8]=second gaussian mean
    p[9]=second gaussian sigma
    '''
    if plot:
        pl.figure(3)
        pl.errorbar(x, y, yerr=e, color='k')
    import time
    #    time.sleep(1)
    #    print sum(((y-mycavvaccaleib(x,p,secondg=True))**2))
    if secondg > 0:
        return sum(((y - mycavvaccaleib(x, p, secondg=True)) ** 2) / e ** 2)
    else:
        return sum(((y - mycavvaccaleib(x, p, secondg=False)) ** 2) / e ** 2)

import scipy.optimize

if __name__ == '__main__':
    lcv = np.loadtxt(sys.argv[1], unpack=True)
    secondg = False
    try:
        if int(sys.argv[2]) > 0:
            secondg = True
    except:
        pass
    x = lcv[1]
    y = lcv[2]
    e = lcv[3]
    mjd = lcv[0]
    ax = pl.figure(0, figsize=(10,5)).add_subplot(111)
    #pl.errorbar(x, y, yerr=e, color="#47b56c", label="data")
    p0 = [0] * 10
    p0[0] = -4
    peakdate = x[np.where(y == min(y))[0]]
    if len(peakdate) > 1:
        peakdate = peakdate[0]
    p0[1] = peakdate + 5
    p0[2] = 10  # sigma
    #pl.draw()
    lintail = np.where(x > peakdate + 50)[0]
    if len(lintail) < 1:
        print "no tail data"
        linfit = np.polyfit(x[-2:], y[-2:], 1)
        p0[3] = linfit[1]
        p0[4] = linfit[0]
    else:
        linfit = np.polyfit(x[lintail], y[lintail], 1)
        p0[3] = linfit[1]
        p0[4] = linfit[0]

    p0[5] = 0.1
    p0[6] = peakdate - 20
    p0[7] = -1
    p0[8] = peakdate + 25
    p0[9] = 10

    
    pl.figure(3)
    pl.clf()
    #    pf= scipy.optimize.minimize(minfunc,p0,args=(y,x,1), method='Powell')#,options={'maxiter':5})
    if secondg:
        p0[0] += 1.5
        p0[1] *= 2
        pl.plot(x[10:], mycavvaccaleib(x[10:], p0, secondg=True), 'm')

        pf = scipy.optimize.minimize(minfunc, p0, args=(y[10:], x[10:], e[10:], 1), method='Powell')  # ,options={'maxiter':5})

    else:
        pl.plot(x[10:], mycavvaccaleib(x[10:], p0, secondg=False), 'k')

        pf = scipy.optimize.minimize(minfunc, p0, args=(y[10:], x[10:], e[10:], 0), method='Powell')  # ,options={'maxiter':5})
    #pl.figure(4)
    pl.figure(0)
    ax.errorbar(mjd+0.5-53000, y, yerr=e, fmt=None, ms=7,
                alpha = 0.5, color='k', markersize=10,)
    ax.plot(mjd+0.5-53000, y, '.', ms=7,
            alpha = 0.5, color='#47b56c', markersize=10,
            label = "SN 19"+sys.argv[1].split('/')[-1].\
            replace('.dat', '').replace('.', ' '))
    #    mycavvaccaleib(x,pf.x, secondg=True)
    mycavvaccaleib(x, pf.x, secondg=secondg)
    
    ax.plot(mjd[10:]+0.5-53000, mycavvaccaleib(x[10:], pf.x, secondg=secondg), 'k',
            linewidth=2, label="vacca leibundgut fit")  # , alpha=0.5)
    #    pl.plot(x,mycavvaccaleib(x,pf.x, secondg=True), 'k',linewidth=2, label="fit")
    xlen = mjd.max() - mjd.min()
    ax.set_xlim(mjd.min()-xlen*0.02+0.5-53000, mjd.max()+xlen*0.02+0.5-53000)
    ax.set_ylim(max(y + 0.1), min(y - 0.1))
    ax2 = ax.twiny()
    Vmax = 2449095.23-2453000
    ax2.tick_params('both', length=10, width=1, which='major')
    ax2.tick_params('both', length=5, width=1, which='minor')
    ax2.set_xlabel("phase (days)")
    ax2.set_xlim((ax.get_xlim()[0] - Vmax, ax.get_xlim()[1] - Vmax))
    #    pl.ylim(10,21)
    pl.draw()
    pl.legend()
    ax.set_xlabel("JD - 24530000")
    ax.set_ylabel("magnitude")
    #pl.title(sys.argv[1].split('/')[-1].replace('.dat', '').replace('.', ' '))
    #pl.show()
    pl.tight_layout()
    
    pl.savefig("../fits/" + sys.argv[1].split('/')[-1].replace('.dat', '.vdfit.pdf'))
    cmd = "pdfcrop " + "../fits/" + sys.argv[1].split('/')[-1].replace('.dat', '.vdfit.pdf')
    print cmd
    os.system(cmd)
