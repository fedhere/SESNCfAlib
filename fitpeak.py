import pandas as pd
import pylab as pl
import numpy as np
from effwavelengths import *
import statsmodels.formula.api as smf

def plotmodel(modl, df, color='r'):
    y = np.zeros_like(df.effw)
    for p in range(len(modl.params)):
        y = y + modl.params[p] * (df.effw ** p)
        
    pl.plot(df.effw, y, color=color)

pks = pd.read_csv("peakdates.csv",  header=None,
                  names=["band","mean","median","std"])
pks['effw'] = [effws[b][0] for b in pks.band.values]

model1md = smf.gls(formula = 'median ~ effw', data=pks).fit()
model1 = smf.gls(formula = 'mean ~ effw', data=pks).fit()
model2 = smf.gls(formula = 'mean ~ effw + I(effw**2)', data=pks).fit()
model2md = smf.gls(formula = 'median ~ effw + I(effw**2)', data=pks).fit()

pl.errorbar(pks.effw, pks['median'], yerr = pks['std'], fmt='.')
pl.errorbar(pks.effw, pks['median'], yerr = pks['std'], fmt='.')
pl.plot(pks.effw,pks.effw*1.42e-3 - 8.32,'k')

plotmodel(model1, pks, color='r')
plotmodel(model2, pks, color='orange')
plotmodel(model1md, pks, color='b')
plotmodel(model2md, pks, color='violet')

print model2md.params
print model2md.conf_int()
