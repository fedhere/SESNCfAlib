import matplotlib as mpl
import pylab as pl
from pylab import rc
rc('axes', linewidth=1.2)
#from matplotlib.font_manager import FontProperties##


#FontProperties.set_weight('normal')
mpl.rcParams['font.size'] = 18.
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times']#Computer Modern Roman']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['text.usetex'] = True
#print mpl.rcParams['font.serif']
#mpl.rcParams['font.serif'] = 'Times New Roman'#Bitstream Vera Serif'
#print mpl.rcParams['font.serif']
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 16.
mpl.rcParams['ytick.labelsize'] = 16.
params = {'legend.fontsize': 18,
          'legend.linewidth': 1,
          'legend.numpoints':1,
          'legend.handletextpad':0.01
      }
pl.rcParams.update(params)    
