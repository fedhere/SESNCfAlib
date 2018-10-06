import sys,os,glob, inspect
import pylab as pl
from numpy import *
from scipy import optimize
import pickle
import time
import copy
cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]) + "/templates")
if cmd_folder not in sys.path:
     sys.path.insert(0, cmd_folder)
from templutils import *

filters=sys.argv[1:]
start=0



def mygauss(x,p):
    global start
    start = 0.1
#    start+=0.05
    PENALTY=0
    GAUSSPLOT=False#True
    g=p[0]*exp(-(x-p[3])**2/p[1]**2)+p[4]
# add linear decay after 50 days from peak f(x)=0.02*x+1    
    f=p[8]*((x))
    g+=f
    if GAUSSPLOT:            
        print "plotting gaussian, ",start
        pl.plot(x,g,'r-', alpha=start)
        pl.draw()
#        pl.ylim(3.5,-1.0)
    return g

'''
    if PENALTY:
        #if p[0]<0:
        #    print "penalty 3!!"
        #    g=g*abs(p[0])
        #    print max(g)
        if min(g)<9.8:
            g=g-(9.5-min(g))
            print "penalty 4!!"
            print 9.8-min(g)
        else :
            print "here", min(g)
#add second gaussian
    if 'BOSr' in filter or 'BOSi' in filter or 'UKIRTH' in filter or 'UKIRTJ' in filter or 'UKIRTK' in filter or 'WIRCH' in filter or 'WIRCJ' in filter or 'WIRCK' in filter or 'FTNi' in filter or 'FTNr' in filter:
        g=g+p[5]*exp(-(x-p[6])**2/p[7]**2)
        if PENALTY:
            if p[5]<0:
                print "penalty 2!!"
                g=g*abs(p[5])
            if abs(p[6]-50)>20.0:
                print "penalty 1!!"
                g=g*abs(p[6]-50)

'''


def exprise(x,g,p):
    global start
    EXPPLOT=False
    start+=10
#    g[where(x<-2)]*=x[where(x<-2)]**(p[11]*2)
    tmp=p[9]
#    pl.plot (x,g,'r-',linewidth=3)
    newg=copy.deepcopy(g)
    newg=newg+1
#    newg[where(x<-tmp)]
    newg*=((exp(-x/p[10])/exp(tmp/p[10]))+1)#[where(x<-tmp)]
    print "newg: ",p[10],p[9]
    
#(p[9]*exp((abs(x+tmp)**p[11])/p[10]**p[11]))[where(x<-tmp)]
#    pl.plot (x,g,'y-')
#    pl.draw()
#    print p[8],p[9]
#    plot (x,p[11]*exp(-(x)/p[2])/min(exp(-(x)/p[2]))+1,'r',alpha=start)
#    print "now the rise is plotted ",min(g),max(g), p[11]
    newg=newg-1
    if EXPPLOT:
         pl.plot (x,newg,'y-',alpha = 0.3)#,linewidth=3)
    #    pl.ylim(3.5,-0.5)
         pl.draw()
#    pl.show()
#    time.sleep(10)
#    ylim(11,9)
    return newg

#errfunc = lambda p, x, y, err: (y - mygauss(x,p,filter))/ err 
if __name__=='__main__':
  pl.ion()
  errfunc = lambda p, x, y,myfilter: (y - mygauss(x,p))
  errfuncrise = lambda p, x, g, y: (y - exprise(x,g,p))
#errfuncrise = lambda p, x, y,filter: (y - mygauss2(x,p,filter))


#ion()


  template=Mytempclass()
  template.loadtemplatefile()
  pl.figure()    
  
  for b in 'V','R':
     
     pl.plot(template.template[b].x, template.template[b].median, 'b-')
     pl.fill_between(template.template[b].x,template.template[b].median-template.template[b].std,template.template[b].median+template.template[b].std, alpha=0.1, color='#0000ff')
     pl.ylim(3.5,-1)
     pl.xlim(-10,50)
     
  pl.draw()
  q=[0.03,30,12]
  
  pinit=zeros(12,float)
  
  pinit[ 0 ]= -2.23712689035
  pinit[ 1 ]= 23.771414014
  pinit[ 2 ]= 30.0
  pinit[ 3 ]= 0.44643855512
  pinit[ 4 ]= 2.3090049932
  pinit[ 5 ]= -2.0
  pinit[ 6 ]= 40.0
  pinit[ 7 ]= 20.0
  pinit[ 8 ]= 0.01
  pinit[ 9 ]= 5.49366
  pinit[ 10 ]= 3.5207
  pinit[ 11 ]= 0.01
  
  newx=arange(-10,150)
  err = ones(len(template.template[b].x),float)#+20.0
#err[where(abs(template.template[b].x-10)==min(abs(template.template[b].x-10)))]=1.0
  
  for b in ['V','R']:
       for repeat in [0,1]:

            pl.figure()    
            
            pl.plot(template.template[b].x, template.template[b].median, 'b-')
            pl.fill_between(template.template[b].x,template.template[b].median-template.template[b].std,template.template[b].median+template.template[b].std, alpha=0.1, color='#0000ff')
            pl.ylim(3.5,-1)
            pl.xlim(-10,50)
#pl.show()
            pl.draw()
            for i,p in enumerate(pinit):
                 print "repeat: ",repeat," pinit[",i,"]=",p
            myfilter=b
        
        #plot initial guess
            pl.plot(newx,mygauss(newx,pinit)*(exp(-(newx-0)/2.0)/max(exp(-(newx-0)/2.0))+1), 'k--')
#        pl.draw()
            minx=0
            
            out = optimize.leastsq(errfunc, pinit,args=(template.template[b].x[where(template.template[b].x>minx)],template.template[b].median[where(template.template[b].x>minx)],myfilter),full_output=1)#,maxfev=50)#, err[where(sn[0]>25)]), full_output=1)
            pfinal=out[0]
            covar=out[1]
            for i,p in enumerate(pfinal):
                 print "renew pinit[",i,"]=",p
           
#            pl.plot(template.template[b].x, template.template[b].median)
#            pl.fill_between(template.template[b].x,template.template[b].median-template.template[b].std,template.template[b].median+template.template[b].std, alpha=0.1)        
            pl.plot(newx,mygauss(newx,pfinal),'c-')
            pl.draw()
            pinit=pfinal
            start=0
            
            print "now for the rise"
            out = optimize.leastsq(errfuncrise, pinit,args=(template.template[b].x,mygauss(template.template[b].x,pfinal), template.template[b].median)
                             ,full_output=1)#, err[where(sn[0]>-5)]), full_output=1)

#      time.sleep(20)
      
        #    show()
            pfinal=out[0]
            print pfinal
            for i,p in enumerate(pfinal):
                 print "repeat ", repeat," renew pinit[",i,"]=",p
#            covar=out[1]
            pl.plot(template.template[b].x,exprise(template.template[b].x,mygauss(template.template[b].x,pfinal),pfinal),'y-',linewidth=2)
            pl.draw()
            pinit=pfinal 
      
       pl.ylabel(myfilter)
       pl.xlabel("epoch")       
       pl.savefig("templates/empiricalmodel_"+b+".png")
       pickle.dump(pfinal,open("templates/empiricalmodel_"+b+".pkl", "wb"))
#       time.sleep(3)


    #savefig("empiricalmodel_"+filter+".png")


#    time.sleep(3)
#show()
#savefig("allempiricalmodels.png")
