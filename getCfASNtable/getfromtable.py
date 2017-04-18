#!/usr/bin/python
doc_name        = 'CfA.SNIbc.BIGINFO.list'
#doc_name        = 'test'
USR='fedhere'
PWD='erail1959'
import sys,os
sys.path.append('mypython')
import pylab as pl
import gdata.docs
import gdata.docs.service
import gdata.spreadsheet.service
import re
import optparse
from numpy import array, ma, mean, std, median, where,zeros


def getstats(vals,data, sntype=None):
        mydic=makevalsintodictionary((vals,data))
	if sntype== None:
		sntype = ['all']                  
	else:

		sntype=sntype.split(',')
        for key in mydic.iterkeys():
            mydic[key]=array(mydic[key])

	    for t in sntype:
		    if t == 'all':
			    mydata=mydic[key]
		    else:
			    chindex=where(mydic['type']==t)[0]
			    mydata=mydic[key][chindex]
		    if len(mydata)==0:
			    continue
		    mymask=zeros(len(mydata),float)
		    for i in where(mydata == '--')[0]:
			    mymask[i]=1

		    mx =ma.masked_array(mydata, mask=mymask)
		    thislist=[]
		    failedcount = 0
		    for f in mx:
			    try:
				    thislist.append(float(f))
			    except:
				    failedcount +=1
				    pass
		    print "type:", t, "number of elements: ",len(thislist), "(",failedcount,"failed)"
		    if len(thislist)>0:
			    print "\t",key,": mean=",mean(thislist),"median=",median(thislist),"std=",std(thislist)
                





def putstuff(user,pwd,vals,sn, value,cohort_key=None ):
    # Connect to Google
    gd_client = gdata.spreadsheet.service.SpreadsheetsService()
    gd_client.email = USR#user
    gd_client.password = PWD#pwd
    gd_client.source = 'payne.org-example-1'
    gd_client.ProgrammaticLogin()
#Now that we're connected, we query the spreadsheet by name, and extract the unique spreadsheet and worksheet IDs.

    q = gdata.spreadsheet.service.DocumentQuery()
    q['title'] = doc_name
    q['title-exact'] = 'true'
    feed = gd_client.GetSpreadsheetsFeed(query=q)
    spreadsheet_id = feed.entry[0].id.text.rsplit('/',1)[1]
    feed = gd_client.GetWorksheetsFeed(spreadsheet_id)
    worksheet_id = feed.entry[0].id.text.rsplit('/',1)[1]
    
    rows = gd_client.GetCellsFeed(spreadsheet_id, worksheet_id).entry
    cells=gd_client.GetCellsFeed(spreadsheet_id, worksheet_id)
    batchRequest = gdata.spreadsheet.SpreadsheetsCellsFeed()
    cohort = GetCohort(cohort_key)
    for myrow in cohort:
        colcursor = 0
    #print ("row ::"+str(myrow))
        for mycell in myrow:
      #print ("cell::"+str(mycell))
            found = 0
      #print ("try and place"+str(rowcursor)+','+str(colcursor))
            for myentry in cells.entry:
                if ((int(myentry.cell.row) == int(rowcursor+1)) and (int(myentry.cell.col) == int(colcursor+1))):
                    print "updating  "+myentry.cell.text+" to "+str(mycell)
                    myentry.cell.inputValue = str(mycell)
                    batchRequest.AddUpdate(myentry)
                    found = 1
 
            if not found:
                print "inserting "+str(mycell)+" at Cell "+ str(rowcursor+1)+'_'+str(colcursor+1)
                newCell = gdata.spreadsheet.SpreadsheetsCell()
                newCell.cell = gdata.spreadsheet.Cell(inputValue=str(mycell), text=None, row=str(rowcursor+1), col=str(colcursor+1))
                print newCEll.inpu
                batchRequest.AddInsert(newCell)# the broken part
            colcursor = colcursor + 1
        rowcursor = rowcursor + 1
 
    updated = gd_client.ExecuteBatch(batchRequest, cells.GetBatchLink().href)
 
    if updated:
        print "Sucessfull!"+str(updated)
    else:
        print "Failed!"

'''    if len(sn) == 7:
        sn=sn.upper().replace('SN','sn')
    for i in rows:
        help(cells.entry[0])
        print cells.entry[0]#[i]#.cells.inputValue
        print cells.entry[1]#[i]#.cells.inputValue
        sys.exit()
        if  sn == cells.entry[1].cells.inputValue:
            i.custom[v].cell =value
    return value
    return 0
'''
        

def makevalsintodictionary(toutput):
    thisdic={}
    for i,f in enumerate(toutput[0][1:]):
        #toutput[0][1:] is the variables list
        thisdic[f]=toutput[1][i+1]
#        print f,thisdic[f]
#        thisdic[f][p] = 
    return thisdic

def makeintodictionary(toutput):
    thisdic={}
    for i,f in enumerate(toutput[1][0]):
        #toutput[1][0] is the sn name
        f=f.lower()
        thisdic[f]={}
        for j,p in enumerate(toutput[0][1:]):
            #toutput[0][1:] is the variables list
            thisdic[f][p]={}
            thisdic[f][p] = toutput[1][j+1][i]

    return thisdic

def grepstuff(user,pwd,vals,sn, types='all',showvals=False,po=False):
    # Connect to Google
    gd_client = gdata.spreadsheet.service.SpreadsheetsService()
    gd_client.email = USR#user
    gd_client.password = PWD#pwd
    gd_client.source = 'payne.org-example-1'
    gd_client.ProgrammaticLogin()
#Now that we're connected, we query the spreadsheet by name, and extract the unique spreadsheet and worksheet IDs.

    q = gdata.spreadsheet.service.DocumentQuery()
    q['title'] = doc_name
    q['title-exact'] = 'true'
    feed = gd_client.GetSpreadsheetsFeed(query=q)
    spreadsheet_id = feed.entry[0].id.text.rsplit('/',1)[1]
    feed = gd_client.GetWorksheetsFeed(spreadsheet_id)
    worksheet_id = feed.entry[0].id.text.rsplit('/',1)[1]
    
    rows = gd_client.GetListFeed(spreadsheet_id, worksheet_id).entry
    

#At this point, you have a row iterator which will yield rows for the spreadsheet. This example will print everything out, keyed by column names:
    
    if showvals:
        for r in rows[0].custom.iterkeys():
            if r.startswith("#"):
                continue
            else:
                print r

        return None, None 


    if vals == 'all':
        vals=[]
        for r in rows[0].custom.iterkeys():
            if r.startswith("#"):
                continue
            vals.append(r)

    else:
            vals=vals.split(',')

    types = types.split(',')
    if not types == ['all']:
	    vals.append('type')
    try:
        vals.remove('comment')
    except:
#        print "no comment column to remove"
        pass
    data=[]
    if 'snname' not in vals:
        vals.insert(0,'snname')
    else:
        a=vals.index('snname')
        vals[0],vals[a]=vals[a],vals[0]
    if 'po' and 'lcvquality' not in vals:
        vals.insert(1,'lcvquality')

#    print "here", vals
    if sn:
        if ',' in sn:
            sn=sn.split(',')
        else:
            sn=[sn]

        for s in sn:
            datai=[]
            if len(s) == 7:
                s=s.upper().replace('SN','sn')

            if s not in [i.custom['snname'].text for i in rows if not i.custom['snname'].text.startswith('#')]:
		    print "cannot find supernova ",s#,". available objects are: ",", ".join([i.custom['snname'].text for i in rows if not i.custom['snname'].text.startswith('#')])
		    pass
            for v in vals:
			if po:
				try:
					datai.append(array([i.custom[v].text for i in rows if not i.custom[v].text == None and not i.custom[v].text.startswith('#') and s in i.custom['snname'].text and not i.custom['lcvquality'].text == '0' ]))
				except: 
					print "cannot find column '"+v+"'. available keywords are: ",", ".join([r for r in rows[0].custom.iterkeys() if not r.startswith('#')])
				pass
			else:
				try:
					datai.append(array([i.custom[v].text for i in rows if not i.custom[v].text == None and  not i.custom[v].text.startswith('#') and s in i.custom['snname'].text ]))                    
				except: 
					print "cannot find column '"+v+"'. available keywords are: ",", ".join([r for r in rows[0].custom.iterkeys() if not r.startswith('#')])
					pass
	    if len(datai)>0:
		    data.append(datai)
    else:
        if types == ['all']:
		datai=[]
		for v in vals:
			if po:
				try:
					datai.append(array([i.custom[v].text for i in rows if not i.custom[v].text == None and not i.custom[v].text.startswith('#') and not i.custom['lcvquality'].text == '0' ]))
				except: 
					print "cannot find column '"+v+"'. available keywords are: ",", ".join([r for r in rows[0].custom.iterkeys() if not r.startswith('#')])
				pass
			else:
				try:
					datai.append(array([i.custom[v].text for i in rows if not i.custom[v].text == None and not i.custom[v].text.startswith('#')]))
				except: 
					print "cannot find column '"+v+"'. available keywords are: ",", ".join([r for r in rows[0].custom.iterkeys() if not r.startswith('#')])
					pass
		if len(datai)>0:
			data.append(datai)
	else:
		datai=[]
		for v in vals:
			if po:
				try:
					datai.append(array([i.custom[v].text for i in rows if not i.custom[v].text == None and not i.custom[v].text.startswith('#') and not i.custom['lcvquality'].text == '0' and i.custom['type'].text in types]))				
				except: 
					print "cannot find column '"+v+"'. available keywords are: ",", ".join([r for r in rows[0].custom.iterkeys() if not r.startswith('#')])
					pass
			else:
				try:
					datai.append(array([i.custom[v].text for i in rows if not i.custom[v].text == None and not i.custom[v].text.startswith('#') and i.custom['type'].text in types]))
				except: 
					print "cannot find column '"+v+"'. available keywords are: ",", ".join([r for r in rows[0].custom.iterkeys() if not r.startswith('#')])
					pass
		if len(datai)>0:
			data.append(datai)
			
			
#        print data


    return vals,data


#################################################################

if __name__ == "__main__":

    parser = optparse.OptionParser(usage="getfromtable -u username -p password -v <value1>,<value2>,<value3> -s <snname>", conflict_handler="resolve")
    parser.add_option('-u','--user', default=None , type="string",
                      help='google user name')
    parser.add_option('-p','--password', default=None , type="string",
                      help='google password')
    parser.add_option('-s','--sn', default=None , type="string",
                      help='sn name')
    parser.add_option('--showvals', default=False , action="store_true",
                      help="show only")
    parser.add_option('--onlyphot', default=False , action="store_true",
                      help="only phot")
    parser.add_option('--stats', default=False , action="store_true",
                      help='''get mean, std and median for the desired values.''')
    parser.add_option('--type', default='all' , type="string",
                      help="only extracting given type (or types)")
    parser.add_option('-v','--values', default='all' , type="string",
                      help='''comma separated list of values to extract.

acceptable values are:
snname : SN name (its printed either way) |
type:    SN type |
ra : RA |
dec : Dec |
hostgal : h ost galaxy  |
vz : redshift velocity | 
z : redshift |
zref : redshift |
maxvdate : date of Vmax (julian date) from literature |
maxvjd : JD date of Vmax from literature |
maxvmag : V flux at maxVJD from literature) |
maxref : reference for literature Vmax |
cfavjdpolyfit : Vmax JD derived from CfA data - polynomial fitting |
cfavjdcovarerror : error on CfA data derived Vmad JD from polynomial fit covariance matrix |
dvjdwithcfapolyfit : difference b|w CfA and literature Vmax JD (days) - polynomial fit |
cfavjdbootstrap : Vmax JD from CfA data derived with bootstrap  |
cfavjdbootstraperror : error on CfA data derived Vmad JD from bootstrapped polynimial fit |
dvjd : difference b/w CfA and literature Vmax JD (days) - bootstrap |
cfadm15 : CfA dm15 from polynomial fit |
cfadm15linear : dm15 from CfA data from linear interpolation |
''')
    options,  args = parser.parse_args()
    if len(args)>0:
        sys.argv.append('--help')
        options,  args = parser.parse_args()
        sys.exit(0)
        
    if options.user == None:
        print "no username"
        sys.argv.append('--help')
        options,  args = parser.parse_args()
        sys.exit(0)


    if options.password == None:
        print "no username"
        sys.argv.append('--help')
        options,  args = parser.parse_args()
        sys.exit(0)

    if options.showvals == True:
        print "available values to extract"
        vals,alldata=grepstuff(options.user,options.password,options.values, options.sn, showvals=True)
        sys.exit()


    vals,alldata=grepstuff(options.user,options.password,options.values, options.sn, options.type, options.showvals,options.onlyphot)
    if options.stats:
        
        getstats(vals,alldata, options.type)
    
    else:
        maxlen=0
        for data in alldata:
            maxlen=max(maxlen,len(data))
        for data in alldata:
            l=len(data)
	print '#',
        for i in vals:
		print i,"\t\t",
        print ""
	for alld in alldata:
           for i in range(len(alld[0])):
               for data in alld:
		    try:
			    print data[i],"\t",
		    except IndexError:
			    print "N/A",
	       print ""
        
