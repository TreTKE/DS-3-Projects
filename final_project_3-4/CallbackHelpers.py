
def encodeIncident(stuff):

	typeDict = dict({

		'Callback': 'cb',
		'Safety Inspection': 'si',
		'Preventive Maintenance': 'pm',
		'PM': 'pm',
		'TFR Repair': 'rpr',

		})

	for key in typeDict.keys():
		
		if key in stuff:
			newstuff = typeDict[key]

	return newstuff



def combineTix(dfRaw):
	import pandas as pd

	uniques = dfRaw.ticketKey.unique() #list of unique ticketKeys
	finaldf = pd.DataFrame()

	for n in uniques:
		# create new df only containing rows with the current ticketKey
		newdf = dfRaw[(dfRaw.ticketKey==n)==True] 

		#list of index numbers from dfRaw
		inds = newdf.index.tolist()

		# proceed only if we have duplicates in this key
		if len(newdf) > 1:

			mechList = []
			descList = []

			# do the thing for mechNotes
			for note in newdf.mechNotes:

				# if not a duplicate note, add to the list
				if not (note in mechList):
					mechList.append(note)
					mechList.append('///') #seperator

			# do the thing for descriptions
			for note in newdf.desc:

				if not (note in descList):
					descList.append(note)
					descList.append('///') #seperator

			newdf = newdf[newdf.index.tolist()==min(inds)]
			#print newdf
			newdf.mechNotes.iloc[0] = mechList
			newdf.desc.iloc[0] = descList

		finaldf = pd.concat([finaldf,newdf])

		del(newdf)

	return finaldf



# Find the number of each incident on a given day
# optoinal input to import weather columns as well
def incidentByDate(df,weatherCols=None):
	import pandas as pd

	uniques = df.Reported.unique()

	cols = ['cb','pm','rpr','si']

	if weatherCols is not None:

		colsF = cols + weatherCols

	dfOut = pd.DataFrame(index=uniques,columns=colsF)

	for day in (uniques):

		dfTemp = df[df.Reported==day] #isolate current day
		
		numCB = dfTemp.cb.sum() #number of callbacks
		numPM = dfTemp.pm.sum() #number of prev. maint
		numRPR = dfTemp.rpr.sum() #number repairs
		numSI = dfTemp.si.sum() #num safety inspections

		dfOut.loc[day,cols] = [numCB,numPM,numRPR,numSI]

		if weatherCols is not None:

			dfOut.loc[day,weatherCols] = dfTemp.loc[min(dfTemp.index),weatherCols].astype('float')

	return dfOut


def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.
	    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    import matplotlib.pyplot as plt
    df2=df.astype(float)
    df2.index=range(0,len(df))
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');
    plt.yticks(range(len(corr.columns)), corr.columns);
    heatmap = ax.pcolor(corr)
    
    plt.colorbar(heatmap)


# Cross Correlation plotting function 
def MyXcorr(x, y, ax=None):
	import matplotlib.pyplot as plt
	import numpy as np

	if ax is None:
		ax = plt.gca()

	x = x - x.mean()

	autocorr = np.correlate(x, y, mode='full')
	autocorr = autocorr[x.size:]
	autocorr /= np.abs(autocorr).max()

	return ax.plot(autocorr)

# fig, ax = plt.subplots()
# acorr(data)
# plt.show()
    



def MyXcorr2(x, y, ax=None):
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()

    x = x - x.mean()
    y = y - y.mean()
    autocorr = np.correlate(x, y, mode='full')
    autocorr = autocorr[x.size:]
    ff = (x**2).sum()
    gg = (y**2).sum()
    den = (ff*gg)**.5
    autocorr /= den
    #autocorr /= np.abs(autocorr).max()
    
    return ax.plot(autocorr)


def box_dow(dfInCt):
	import matplotlib.pyplot as plt 
	import pandas as pd
	import numpy as np 
	dfMon = dfInCt[dfInCt.index.weekday==0]
	dfTue = dfInCt[dfInCt.index.weekday==1]
	dfWed = dfInCt[dfInCt.index.weekday==2]
	dfThu = dfInCt[dfInCt.index.weekday==3]
	dfFri = dfInCt[dfInCt.index.weekday==4]
	dfSat = dfInCt[dfInCt.index.weekday==5]
	dfSun = dfInCt[dfInCt.index.weekday==6]
	n=1
	fig = plt.figure(figsize=(18,5))
	ax1 = fig.add_subplot(1,7,n)
	n+=1
	ax2 = fig.add_subplot(1,7,n)
	n+=1
	ax3 = fig.add_subplot(1,7,n)
	n+=1
	ax4 = fig.add_subplot(1,7,n)
	n+=1
	ax5 = fig.add_subplot(1,7,n)
	n+=1
	ax6 = fig.add_subplot(1,7,n)
	n+=1
	ax7 = fig.add_subplot(1,7,n)
	ax1.boxplot(dfMon.cb)
	ax2.boxplot(dfTue.cb)
	ax3.boxplot(dfWed.cb)
	ax4.boxplot(dfThu.cb)
	ax5.boxplot(dfFri.cb)
	ax6.boxplot(dfSat.cb)
	ax7.boxplot(dfSun.cb)
	ax1.set_xlabel('Mon')
	ax2.set_xlabel('Tues')
	ax3.set_xlabel('Wed')
	ax4.set_xlabel('Thu')
	ax5.set_xlabel('Fri')
	ax6.set_xlabel('Sat')
	ax7.set_xlabel('Sun')
 