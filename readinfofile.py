from __future__ import print_function
import csv, os, sys
import numpy as np

def readinfofile(verbose=False, earliest=False, loose=False):
    infofile=open(os.getenv("SESNCFAlib")+"/CfA.SNIbc.BIGINFO.csv",'r')
    reader = csv.reader(infofile)
    rownum = 0
    Vmax={}
    Rmax={}
    Bmax={}
    Imax={}
    sntype={}
    earliestV={}
    Vmaxflag={}
    for row in reader:
        # Save header row.
        if rownum == 0:
            header = row
        else:
            colnum = 0
            for col in row:
               if len(header) <= colnum:
                   continue
               #print header[colnum]
               if 'lcvquality' == header[colnum].lower():
                   lcvqual=col.lower().replace(' ','')
                   if verbose: print (name,'lcvquality', lcvqual)
               elif 'SNname'.lower() == header[colnum].lower():
                    name=col.lower().replace(' ','')
                    if verbose: print (name,'cfa ')
               elif 'Type' in header[colnum]:
                    thissntype=col
                    
               elif  header[colnum] =='CfA VJD bootstrap' :
                    cfavjdmax=col
                    if verbose: print (name,'cfa V',col)
               elif 'MaxVJD' in header[colnum]:
                    vjdmax=col
                    if verbose: print (name,'not cfa',col)
               elif  header[colnum] =='CfA BJD bootstrap' :
                    cfabjdmax=col
                    if verbose: print (name,'cfa B',col)
               elif  header[colnum] =='CfA IJD bootstrap' :
                    cfaijdmax=col
                    if verbose: print (name,'cfa I',col)
               elif  header[colnum] =='CfA RJD bootstrap' :
                    cfarjdmax=col
                    if verbose: print (name,'cfa R',col)
               elif 'earliestV' in  header[colnum]:
                    if verbose: print (name,'earliest',col)
                    earliestv=col
               elif header[colnum] == 'finalmaxVjd':
                    print ("here",  name, col)
                    if verbose: print (name,finalmaxVjd,col)
                    finalmaxVjd=col
                    print ("here" , finalmaxVjd)
               colnum += 1
            if verbose: print ("from readinfofile: ", name, thissntype, finalmaxVjd,
                               cfavjdmax, vjdmax,
                               earliestv)
            if '0' in lcvqual:
                if verbose: print ("bad quality, moving on")
                continue
            else:
                if verbose: print ("lets see what we have for ", name)

            sntype[name]=thissntype
            Vmaxflag[name]=False
            try:
              Vmax[name]=float(finalmaxVjd)
            except:
              try:
                  Vmax[name]=float(cfavjdmax)
              except:
                try:
                    Vmax[name]=float(vjdmax)
                except:
                    try:
                        Rmax[name]=float(cfarjdmax)
                        if verbose: print ("here Rmax", Rmax, end="")
                        Rmaxflag=True
                    except:
                        Rmaxflag=False
                        pass
                    try:
                        Bmax[name]=float(cfabjdmax)
                        if verbose: print ("here Bmax", Bmax, end="")
                        Bmaxflag=True
                    except:
                        Bmaxflag=False
                        pass
                    try:
                        Imax[name]=float(cfaijdmax)
                        if verbose: print ("here Imax", Imax, end="")
                        Imaxflag=True
                    except:
                        Imaxflag=False
                        pass
                    if  Rmaxflag+Bmaxflag+Imaxflag>=2:
                        if Bmaxflag and Rmaxflag:
                            Vmax[name]=np.mean([Rmax[name]-1.5+2400000.5,Bmax[name]+2.3+2400000.5])
                        elif Bmaxflag and Imaxflag:
                            Vmax[name]=np.mean([Imax[name]-3.1+2400000.5,Bmax[name]+2.3+2400000.5])
                        elif Rmaxflag and Imaxflag:
                            Vmax[name]=np.mean([Imax[name]-3.1+2400000.5,Rmax[name]-1.5+2400000.5])
                    elif Rmaxflag+Bmaxflag+Imaxflag>=1 and loose:
                        if Imaxflag:
                            Vmax[name]=Imax[name]-3.1+2400000.5
                        if Rmaxflag:
                            Vmax[name]=Rmax[name]-1.5+2400000.5
                        if Bmaxflag:
                            Vmax[name]=Bmax[name]+2.3+2400000.5
                    else:
                        if earliest:
                            Vmax[name]=earliestv
                            Vmaxflag[name]=True
                        else:
                            Vmax[name]=vjdmax
                            Vmaxflag[name]=True


            if verbose: print (Vmax[name])
        rownum+=1


    infofile.close()
    if earliest:
        return Vmax, sntype, Vmaxflag
    else:
        return Vmax, sntype


def readinfofileVmin(verbose=False):
    infofile=open("CfA.SNIbc.BIGINFO.csv",'r')
    reader = csv.reader(infofile)
    rownum = 0
    Vmax={}
    Vmaxflag={}
    sntype={}
    
    for row in reader:
        # Save header row.
        if rownum == 0:
            header = row
        else:
            colnum = 0
            for col in row:
               if 'lcvquality' == header[colnum].lower():
                   lcvqual=col.lower().replace(' ','')
               if 'SNname'.lower() == header[colnum].lower():
                    name=col.lower().replace(' ','')
                    print ("here is the name",col)
               if 'Type' in header[colnum]:
                    thissntype=col
               if  header[colnum] =='CfA VJD bootstrap' :
                    cfavjdmax=col
               if 'earliestV' in  header[colnum]:
                    earliestv=col
               elif 'MaxVJD' in header[colnum]:
                    vjdmax=col
               colnum += 1
            if verbose:
                print ("from readinfofile: ",name, thissntype, cfavjdmax, vjdmax, earliestv)
            if '0' in lcvqual:
                continue
            Vmaxflag[name]=False
            sntype[name]=thissntype
            try:
                Vmax[name]=float(cfavjdmax)
            except:
                try:
                    Vmax[name]=float(vjdmax)
                except:
                    Vmax[name]=earliestv
                    Vmaxflag[name]=True
                    
            rownum += 1
        print (Vmax[name])

    infofile.close()
    return Vmax, sntype, Vmaxflag
