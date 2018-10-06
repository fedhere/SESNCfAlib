
def readin(f):
    thissn=snstuff.mysn(f, addlit=True)

    ################read supernova data and check metadata

    lc, flux, dflux, snname = thissn.loadsn(f, fnir=True, verbose=True,
                                            lit=True, addlit=True)

    #thissn.printsn()
    #raw_input()
    thissn.readinfofileall(verbose=False, earliest=False, loose=True)
    #thissn.printsn()


    Dl = float(thissn.metadata['luminosity distance Mpc'])
    su = templutils.setupvars()
    thissn.setsn(thissn.metadata['Type'], thissn.Vmax)
    myebmv=su.ebmvs[thissn.snnameshort]
    print ("E(B-V)", myebmv)
    myebmv+=hostebmv#su.ebmvcfa[thissn.snnameshort]
    print ("E(B-V) total", myebmv)
    
        
    Vmax = thissn.Vmax
    thissn.setphot()
    thissn.getphot(myebmv)
    thissn.setphase()
    
    thissn.printsn(photometry=False)

    #thissn.printsn()

    fig = pl.figure(figsize=(5,3))
    thissn.plotsn(photometry=True, show=True, fig = fig)
    return thissn



def skgp (x, y, yerr, phases, t0):
    from sklearn.gaussian_process import GaussianProcess
    XX = np.atleast_2d(np.log(x-min(x)+1)).T
    #XX = np.atleast_2d(x).T
    
    gphere = GaussianProcess(corr='squared_exponential',
                                  theta0=t0,
                                  thetaL=t0*0.1,
                                  thetaU=t0*10,
                                  nugget=(yerr / y) ** 2,
                                  random_start=100)
    gphere.fit(XX, y)
    
    xx = np.atleast_2d(np.log(phases-min(X)+1)).T
    #xx = np.atleast_2d(phases).T
    y_pred, MSE = gphere.predict(xx, eval_MSE=True)
    sigma = np.sqrt(MSE)
    return (y_pred, sigma)


    
def georgegp (x, y, yerr, phases, kc, kc1):
    import george
   
    # Set up the Gaussian process.
    kernel = kc1 * 10 * kernelfct(kc)#ExpSquaredKernel(1.0)
    gp = george.GP(kernel)
    #print ("wtf", gp.kernel)
    
    # adding  a small random offset to the phase so that i never have
    # 2 measurements at the same time which would break the GP
    
    # Pre-compute the factorization of the matrix.
    XX = x
    XX = np.log(XX-XX.min()+1)

    # You need to compute the GP once before starting the optimization.
    gp.compute(XX, yerr)

    # Print the initial ln-likelihood.
    #print("here", gp.lnlikelihood(y))
    #print("here", gp.grad_lnlikelihood(y))

    # Run the optimization routine.
    if OPT:
        p0 = gp.kernel.vector
    
        results = op.minimize(nll, p0, jac=grad_nll, args=(gp))
        print results.x
        # Update the kernel and print the final log-likelihood.
        gp.kernel[:] = results.x
    #print(gp.lnlikelihood(y))

    
#    gp.compute(XX, yerr)
    
    # Compute the log likelihood.
    #print(gp.lnlikelihood(y))

    #t = np.linspace(0, 10, 500)
    ##xx = np.log(phases-min(X)+1)
    xx = phases
    xx = np.log(xx-x.min()+1)
    mu, cov = gp.predict(y, xx)
    std = np.sqrt(np.diag(cov))
    return (mu, std)

import scipy.optimize as op

# Define the objective function (negative log-likelihood in this case).
def nll(p, gp):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    ll = gp.lnlikelihood(y, quiet=True)

    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p, gp):
    # Update the kernel parameters and compute the likelihood.
    print ("wtf2", gp.kernel)
    gp.kernel[:] = p
    print (gp.kernel[:])
    print -gp.grad_lnlikelihood(y, quiet=True)
    return -gp.grad_lnlikelihood(y, quiet=True)


def getskgpreds(ts, x, y, yerr, phases, fig = None):
    t0, t1 = ts
    if t0 ==0 or t1==0:
        return 1e9
    #print (t0,t1)
    gp1, gp2 = georgegp(x, y, yerr, x, t0, t1)
    s1= sum(((gp1-y)/yerr)**2)/len(y)
    #pl.figure(figsize=(1,3))

    #pl.plot(x, gp1,'*')
    
    gp1, gp2 = georgegp(x, y, yerr, phases, t0, t1)
    s2= sum(np.abs((gp1[2:]+gp1[:-2]-2*gp1[1:-1])/\
              (diff(phases)[1:]+diff(phases)[:-1])))
    print ("%.3f"%t0, "%.3f"%t1, "%.1f"%s1, "%.3f"%s2, s1*s2)
    if fig:
        pl.errorbar(x,y,yerr=yerr,fmt='.')
        pl.plot(phases, gp1,'-')
        pl.fill_between(phases, gp1-gp2, gp1+gp2, color='k')
        pl.title("%.3f %.3f %.3f"%(t0, t1, (s1*s2)), fontsize=15)
        pl.ylim(pl.ylim()[1], pl.ylim()[0])
    if isfinite(s1*s2) and not np.isnan(s1*s2):
        return s1*s2
    return 1e9

def kernelfct(kc):
    from george.kernels import ExpSquaredKernel, WhiteKernel, ExpKernel, Matern32Kernel
    return ExpSquaredKernel(kc)# Matern32Kernel(kc)

from scipy import stats

sn = '08D'
b='V'


def findgp(sn, b):
    fall = glob.glob(os.getenv('SESNPATH')+'/finalphot/*'+sn+'*[cf]')
    if len(fall)>0:
        fall[-1] = [fall[-1]] + \
                   [ff for ff in glob.glob(os.environ['SESNPATH']+\
                                           "/literaturedata/phot/*"+sn+".*[cf]")]
    else: fall =  [[ff for ff in glob.glob(os.environ['SESNPATH']+"/literaturedata/phot/*"+sn+".*[cf]")]]
    f = fall[0]
    if not isinstance (f, basestring):
        f=f[0]
    
    x= thissn.photometry[b]['phase']
    x+=0.01*np.random.randn(len(x))
    y= thissn.photometry[b]['mag']
    yerr= thissn.photometry[b]['dmag']
    phases = np.arange(x.min(),x.max(),0.1)
    if x.max()<=30:
        if x.min()<=-15:
            x15 = np.where(np.abs(x+15)==np.abs(x+15).min())[0]
            print x15, y[x15[0]]+0.5
            x = np.concatenate([x,[30]])
            y = np.concatenate([y,[y[x15[0]]+0.5]])
            yerr = np.concatenate([yerr,[0.5]])
            print (x,y,yerr)
        elif (x>=15).sum()>1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x[x>=15],y[x>=15])
            x = np.concatenate([x,[30]])
            y = np.concatenate([y,[slope*30.+intercept]])
            yerr = np.concatenate([yerr,[yerr.max()*2]])
            print (x,y,yerr)
        else:
            return -1
    
    #fig = pl.figure(figsize=(10,3))
    results = op.minimize(getskgpreds, (0.4,1.0), args = (x,y,yerr,phases), bounds=((0,None),(0,None)), tol=1e-8)
    print (results.x)
    
    #t1s = np.exp(np.arange(-2,2,0.5))
    #for tt in np.exp(np.arange(-2,2,0.5)):
    #    fig = pl.figure(figsize=(10,3))
    #    for i,ttt in enumerate(t1s):
    #        ax = fig.add_subplot(len(t1s),1,i+1)
    #        getskgpreds(x,y,yerr,phases,tt,ttt)
    fig = pl.figure(figsize=(10,3))
    getskgpreds(results.x,x,y,yerr,phases, fig)
    pl.ylabel(b+" magnitude")
