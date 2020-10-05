import matplotlib as mpl
import pylab as pl
from pylab import rc
rc('axes', linewidth=1.2)
#from matplotlib.font_manager import FontProperties##


params = {'legend.fontsize': 20,
          #'legend.linewidth': 1,
          'legend.numpoints':1,
          'legend.handletextpad':1
      }

pl.rcParams.update(params)    

print ("updating ticks")

params = {'xtick.top': True,
          'xtick.minor.bottom': True,
          'xtick.minor.top': True,
          'ytick.right': "True",
          'ytick.minor.right': "True",
          'ytick.minor.right': "True"
      }

pl.rcParams.update(params)    

#FontProperties.set_weight('normal')
mpl.rcParams['font.size'] = 18.
#mpl.rcParams['font.size'] = 22.
mpl.rcParams['font.family'] = 'Times New Roman'#serif'
mpl.rcParams['font.serif'] = ['Times']#Computer Modern Roman']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['text.usetex'] = True
#print mpl.rcParams['font.serif']
#mpl.rcParams['font.serif'] = 'Times New Roman'#Bitstream Vera Serif'
#print mpl.rcParams['font.serif']
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 20.
mpl.rcParams['ytick.labelsize'] = 20.
#mpl.rocParams['axes.labelsize'] = 22
#mpl.rcParams['xtick.labelsize'] = 20.
#mpl.rcParams['ytick.labelsize'] = 20.
mpl.rcParams['xtick.major.size']= 10.
mpl.rcParams['xtick.minor.size']= 5.
mpl.rcParams['ytick.major.size']= 10.
mpl.rcParams['ytick.minor.size']= 5.


print ("pylab set up, rcParams")
