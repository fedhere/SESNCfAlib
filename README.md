Library to ingest, process, analyze, and manipulate and the Stripped Envelope Supernova time series produced by the CfA Supernova Group 
http://www.cfa.harvard.edu/supernova/


requires numpy, scipy, mpmath, and a few more standard python modules. 
in addition it requires my astro utilities which are in this github repository: https://github.com/fedhere/fedsastroutils


first, set the following environmental variables (assuming bash syntax)

export SESNPATH="_path_to_data_directory_"
export SESNCFAlib="_path_to_library_/SESNCFAlib"
export UTILPATH="_path_to_randomutils_/fedastroutils/"

you then need the data... the CfA SN data can be found on the CfA Supernova Group website and upon request (to me) it can be provided in the format assumed by the library. 

the data must include directories with the format described below:



the directory:

$SESNPATH/finalphot/

must contain the optical photometry files. these have the CfA3 or CfA formats:

the CfA4 format: this is the head of a CfA4 SN file:

02 53984.47089   3 0.0400 0.0490   18.2740 0.0633 18.3089

02 53990.29343   3 0.0470 0.0533   18.5950 0.0711 18.6364

02 53991.38819   1 0.0750 0.0911   18.8470 0.1180 18.8941

02 53992.20581   3 0.0500 0.0550   18.7480 0.0743 18.7901

02 53994.28382   3 0.0610 0.0443   18.9810 0.0754 19.0258

02 54000.33464   2 0.0940 0.0431   19.6035 0.1034 19.6610

02 54002.35464   3 0.1210 0.0703   19.7940 0.1399 19.8537

02 54006.40358   2 0.1155 0.0269   19.8820 0.1186 19.9337


CfA4 columns: 

column 1: PHOTCODE (filter and phot system identifier) 

column 2: MJD 

column 3: number of templates (?)

column 4: ERROR (from subtraction alone) 

column 5: ? 

column 6: MAG (Natural system) 

column 7: ERROR (includes multiple templates) 

column 8: MAG (Standard system)


the CfA3 format: this is the head of a CfA3 SN file:


  2          54043.49609  512.46  513.90        686.00       113.70   19.572   0.170  19.642

  2          54050.50944  512.46  513.90        590.50        85.70   19.804   0
.151  19.881

  2          54051.53480  512.46  513.90        443.20        78.20   20.046   0
.180  20.143

  2          54053.52676  512.46  513.90        463.20        46.90   20.054   0
.209  20.149

  2          54055.46838  512.46  513.90        543.40        48.20   19.878   0
.197  19.955

  2          54056.50316  512.46  513.90        492.10        33.20   19.932   0
.176  20.013

  2          54070.50659  512.46  513.90        364.80        29.90   20.257   0
.190  20.352

  2          54072.45896  512.46  513.90        571.30        60.80   19.770   0
.214  19.813

  2          54075.53192  512.46  513.90        433.00        60.30   20.071   0
.245  20.135

  2          54083.55186  512.46  513.90        430.00        37.10   20.079   0.194  20.138

CfA3 columns:

column 1: PHOTCODE (filter and phot system identifier)

column 2: MJD

column 3: ?

column 4: ?

column 5: ?

column 6: ?

column 7: MAG (Natural system)

column 8: ERROR (includes multiple templates)

column 9: MAG (Standard system)



the directory for the NIR data:

$SESNPATH/nirphot/PAIRITEL_Ibc/Ibc/lcs/mag/


must contain the NIR photometry files. the following format is assumed:



\#  SN               lII         bII          MW_E(B-V)   z_(helio)      z_(CMB)
        RA(2000.0)   DEC(2000.0)

\#..... a bunch more comment lines here all starting with \#

\#passband   MJD        mag      dmag    Telescope/Instrument                   
            

J           53456.29   16.403   0.175   PTEL_F12_PTEL_mosjSN.4.5-2005Mar27_p3.di
ff.fits    

J           53461.33   16.279   0.250   PTEL_F12_PTEL_mosjSN.4.8-2005Apr01_p3.di
ff.fits    

J           53462.30   16.443   0.046   PTEL_F12_PTEL_mosjSN.4.9-2005Apr02_p3.di
ff.fits    

J           53463.24   16.442   0.175   PTEL_F12_PTEL_mosjSN.4.10-2005Apr03_p3.d
iff.fits   

J           53464.32   16.356   0.175   PTEL_F12_PTEL_mosjSN.4.11-2005Apr04_p3.d
iff.fits   

J           53465.23   16.669   0.083   PTEL_F12_PTEL_mosjSN.4.12-2005Apr05_p3.d
iff.fits   

J           53466.24   16.599   0.060   PTEL_F12_PTEL_mosjSN.4.13-2005Apr06_p3.d
iff.fits   

J           53467.43   16.478   0.053   PTEL_F12_PTEL_mosjSN.4.14-2005Apr07_p3.d
iff.fits   





CfA NIR columns: 

column 1:filter

column 2: MJD

column 3: mag (2MASS system which is the CfA PAIRITEL natural system)

column 4: ERROR

column 5: image name





then you can read, process, plot the data.



you should be able to run it by typing 

python testcode.py

and testcode can be anywhere on your computer as long as your paths are set correctly


