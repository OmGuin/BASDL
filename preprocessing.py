import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags, medfilt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

#PART 1
#FUNCTIONS

#Rolling window filter to remove baseline wander but retain overall mean of timestream
def basedrift_data(I, driftwin, channel):
    """
    % Removes a smoothed baseline from timestream but retains overall mean of
    % timestream; Use a 50 sample median filter (slow and memory intensive but largely insensitive to spikes) 
    % followed by a 12000 sample averaging filter on that result (fast and
    % roughly 6 second time constant with nominal 0.5 ms sampling
    """

    medianI = np.median(I)
    qq = I - medianI #removes median to roughly center baseline on zero
    ksize = np.fix(driftwin*50)
    if ksize % 2 == 0:  # Ensure odd
        ksize += 1
    MF = medfilt(qq, kernel_size=ksize) #sets window size and filters
    basedrift = MF+medianI #restores original amplitude
    basedrift = gaussian_filter1d(basedrift, sigma = 12000/6)

    cI = I - basedrift
    cI += medianI
    cI[cI < 0] = 0

    plt.figure()
    plt.plot(I, 'b', label='Original')
    plt.plot(basedrift, 'r', label='Baseline Drift')
    plt.xlim([0, 10000])
    plt.ylim([0, medianI * 5])
    plt.title(f'Original data with drift overplotted : {channel}')
    plt.legend()
    plt.show()
    
    return cI, basedrift

#Offset removal
#In bd_175.m this is also called "Cut threshold". This routine can be used to subtract a background before finding spikes and spike amps.
#This subtracts a smoothed drifting function from the raw data and sets any values below zero to zero. The idea is to remove an
#unresolved, ubiquitous, luminous background that is adding to the event amplitudes.
def basethresh_data(I, basethresh, channel):
    """
    % Used same median filter as used in Basedrift to find a baseline but is then removed. Zeroes out any data 
    % below zero value in each channel. 
    % Use a 50 sample median filter (slow and memory intensive but largely insensitive to spikes) 
    % followed by a 12000 sample averaging filter on that result (fast and
    % roughly 6 second time constant with nominal 0.5 ms sampling
    """
    medianI = np.median(I)
    qq = I - medianI #Removes median to roughly center baseline on zero
    ksize = np.fix(basethresh*50).astype(int)
    if ksize%2==0:
        ksize+=1
    
    MF = medfilt(qq, kernel_size=ksize) #sets window size and filters
    baseoffset = MF+medianI #restores original amplitude
    baseoffset = gaussian_filter1d(baseoffset, sigma = 12000/6) #SLIGHTLY DIFFERENT FROM MATLAB VERSION
 
    cI = I - baseoffset #Remove drift but now there are negative values
    cI[cI < 0] = 0 #zero any negative values

    #Overplot data and baseline in zoomed in function
    plt.figure()
    plt.plot(I, 'b', label='Original')
    plt.plot(baseoffset, 'r', label='Baseline Offset')
    plt.xlim([0, 10000])
    plt.ylim([0, max(medianI * 5, 10)]) #Handles cases with zero median value
    plt.title(f'Original presubtracted data with filtered baseline offset to be SUBTRACTED : {channel}')
    plt.legend()
    plt.show()

    return cI, baseoffset

#Save processed results to .npz --------- example unpacking in view_results.py
def save_preprocessed(pre_path, pre_filename, datastatA, datastatB, datastatC, handlesA, handlesB, handlesC, peaklocA, peaklocB, peaklocC, readme_pre, versionnum): 
    #Get information about this data run and create a director for this data store
    filebase, tstr = Information(versionnum)
    newdir = os.path.join(pre_path, f"{filebase}_{tstr}")
    os.makedirs(newdir, exist_ok=True)

    save_file_path = os.path.join(newdir, f"{filebase}{pre_filename}{tstr}.npz")
    
    # print(f"pre_path: {type(pre_path)}")
    # print(f"pre_filename: {type(pre_filename)}")
    # print(f"filebase: {type(filebase)}")
    # print(f"newdir: {type(newdir)}")
    # print(f"tstr: {type(tstr)}")
    # print(f"datastatA: {type(datastatA)}")
    # print(f"datastatB: {type(datastatB)}")
    # print(f"datastatC: {type(datastatC)}")
    # print(f"handlesA: {type(handlesA)}")
    # print(f"handlesB: {type(handlesB)}")
    # print(f"handlesC: {type(handlesC)}")
    # print(f"peaklocA: {type(peaklocA)}")
    # print(f"peaklocB: {type(peaklocB)}")
    # print(f"peaklocC: {type(peaklocC)}")
    # print(f"readmePRE: {type(readme_pre)}")
    # print(f"versionnum: {type(versionnum)}")
    np.savez_compressed(
        save_file_path,
        datastatA=datastatA,
        datastatB=datastatB,
        datastatC=datastatC,
        filebase=filebase,
        handlesA=handlesA,
        handlesB=handlesB,
        handlesC=handlesC,
        newdir=newdir,
        peaklocA=peaklocA,
        peaklocB=peaklocB,
        peaklocC=peaklocC,
        prefilenamesave=save_file_path,
        pre_filename=pre_filename,
        pre_path=pre_path,
        readmePRE=readme_pre,
        tstr=tstr,
        versionnum=versionnum
    )
    print(f"Preprocessed data saved to: {save_file_path}")
    return True
    # saved_data = {
    #     "PrePathName": pre_path,
    #     "PreFileName": pre_filename,
    #     "filebase": filebase,
    #     "newdir":newdir,
    #     "tstr":tstr,
    #     "datastatA": datastatA,
    #     "datastatB": datastatB,
    #     "datastatC": datastatC,
    #     "handlesA": handlesA,
    #     "handlesB": handlesB,
    #     "handlesC": handlesC,
    #     "peaklocA": peaklocA,
    #     "peaklocB": peaklocB,
    #     "peaklocC": peaklocC,
    #     "readmePRE": readme_pre,
    #     "versionnum": versionnum
    # }


#Find burst location times and amplitudes (preprocess)
handles = {}
def burst_data(d, tbin, corrt, dofit, channel, handles):
    """
    %This is original 2013 preprocessing code used in MegaMan_V1.m and bd_175
    %Some of the commented code is left to show comparison to what was - if
    %commented it was not doing anything even in old code
    %Channel data are mapped to appropriate old variables like handles
    %Input variable channel is string to label channel being processed
    %handles.medwin=5;    Not used in bd_175 either
    """
    handles['d'] = np.zeros((2, len(d)))
    handles['d'][0, :] = np.arange(len(d)) * tbin #make time vector in seconds
    handles['d'][1, :] = d #time data filtered by drift and basethresh is read in
    #handles['base'] =d     Original time data unfiltered is read in

    print(f"***** Processing channel : {channel}")

    #Show original data
    plt.figure()
    plt.plot(handles['d'][0, :], handles['base'], label='Original') #show original time stream data
    plt.plot(handles['d'][0, :], handles['d'][1, :], '.r', label = "Optionally filtered")
    plt.title(f"Optionally drift/baseline filtered timestream and original {channel}", fontsize=12)
    plt.xlabel("Total Concatenated Time (seconds)")
    plt.ylabel("Photon counts per time bin")
    plt.legend(loc = "upper left")
    plt.show()

    """
    Consider using a threshold level to define when burst boundary happens
    rather than always using zero

    Set an amplitude threshold for base aof spike to 0, mean, median, rms, or
    dynamic range (1/100) of large spike
    """
    #Determine upper range of spike activity ; 30 is a heuristic parameter
    sortbase = np.sort(handles['base'])
    UB = np.median(sortbase[-30:])

    #Determine lower bound (threshold) of trustable spike activity
    #Default to mean for most cases
    handles['spthreshtype'] = 'mean'
    dynrange = 100

    if handles['spthreshtype'] == 'zero':
        spikethresh = 0
        print("Chosen spike finding threshold is zero")
    elif handles['spthreshtype'] == 'mean':
        spikethresh = 2* handles['mean']
        print("Chosen spike finding threshold setting is the 2*mean of 5-sigma filtered data set")
    elif handles['spthrestype'] == 'median':
        spikethresh = 2 * handles['median']
        print("Chosen spike finding threshold setting is the 2*median of 5-sigma filtered data set")
    elif handles['spthrestype'] == 'rms':
        spikethresh = 2 * handles['rms']
        print("Chosen spike finding threshold setting is the 2*rms of 5-sigma filtered data set")
    elif handles['spthrestype'] == 'dynamicrange':
        #Estimate upper bound on events in a channel
        spikethresh = roudn(UB/dynrange)
        print(f"Applying a 1/100 dynamic range threshold : {spikethresh}")
        print("Chosen spike finding threshold setting is UB/100 of dataset")

    #Bound lower threshold has some constraints
    if spikethresh < round(UB / dynrange): #the dynamic range is minimum possible
        spikethresh = round(UB / dynrange)
        print('**** OVERRIDE: Restricted to Upper Bound / 100 as lower limit ****')
    if spikethresh < 10: # set 10 as a lower limit in all cases due to shot noise
        spikethresh = 10
        print('**** OVERRIDE: Restricted to 10 cnts as lower limit - arbitrary Poisson noise limit ****')

    #Store parameters in structure for later use
    handles['spthresh'] = spikethresh
    handles['UB'] = UB

    strike = np.array(d) #data imported to function (not baseline)

    #Set a lower bound for spike finding below the threshold as a way of estimating the subthreshold
    #activity later
    print(f"Setting lower spike acceptance threshold to: {spikethresh}")
    print(f"Retaining subthreshold spikes between {spikethresh/2} and {spikethresh} in Handles structure")
    strike[strike < spikethresh / 2] = 0 #Set lower amplitude to find the edge of spikes

    #Guarantee zeroed end points
    strike[0:2] = 0
    strike[-2:] = 0

    while True:
        t = np.max(np.where(strike == np.max(strike))[0]) #This could be 2nd pnt, 2nd ot last pnt or in the middle of data
        strike[t] = -1 #tip of peak in single event set to -1
        i = 1
        rightend = leftend = True
        while rightend or leftend:
            if rightend:
                ti_index = max(min(t + i, len(strike) - 1), 1) #because nonzero index needed
                if abs(strike[ti_index]) > 0: #see comment above about threshold
                    strike[ti_index] = 0
                else:
                    rightend = False
            if leftend:
                ti_index = max(min(t - i, len(strike) - 1), 1)
                if abs(strike[ti_index]) > 0: #See comment above about threshold
                    strike[ti_index] = 0
                else:
                    leftend = False
            i += 1
        if np.max(strike) <= 0:
            break

    handles['strike'] = -strike
    handles['strikeamp'] = -strike * handles['d'][1, :]

    # Plot spike amplitudes
    plt.figure()
    plt.plot(handles['d'][0, :], handles['strikeamp'], 'r.', label='Located peaks')
    plt.plot(handles['d'][0, :], handles['d'][1, :], 'b')
    plt.axhline(y=UB, color='k', linestyle='-')
    plt.title(f'Threshold & Upperbound with located peaks {channel}', fontsize=12)
    plt.xlabel('Total Concatenated Time (seconds)')
    plt.ylabel('Photon counts per time bin')
    plt.legend()
    plt.show()

    # Store some of hte spike data below the threshold for later noise and 
    # signle event prob analysis; zero these data in main propagated data
    # variables
    submask = handles['strikeamp'] < spikethresh
    handles['subthreshstrike'] = handles['strike'] * submask
    handles['subthreshstrikeamp'] = handles['strikeamp'] * submask
    strike_mask = handles['strikeamp'] > spikethresh
    handles['strike'] *= strike_mask
    handles['strikeamp'] *= strike_mask

    #Consider peaks away from very edge of data set; 15 bin buffer is
    #sized for later Gaussian model fit to events
    peakloc = np.where(handles['strikeamp'] > 0)[0]
    peakloc = peakloc[(peakloc > 15) & (peakloc < len(handles['strikeamp']) - 15)]
    npeaks = len(peakloc)

    # Check interspike timing statistics and remove events too closely spaced
    print("Checking interspike timing distribution and removing events closer than corrt")
    ISI_check(handles,tbin,corrt)
    
    #Do fit of Gaussian model to each event if dofit flag set in parameter
    if dofit: #Hook to turn off Gausian fitting - remove datastat=1 too
        params = {'mean': [], 'sig': [], 'A': []}
        for i in range(npeaks):
            #start with general spike zone
            center = peakloc[i]
            fitme = handles['d'][1, center - 10:center + 11]
            dzone = int(2 * np.sum(fitme) / np.max(fitme)) #refine zone assuming Gaussian peak
            x = np.arange(center - dzone, center + dzone + 1)
            fitme = handles['d'][1, center - dzone:center + dzone + 1]
            A, mu, sigma = fitgauss(fitme, x, np.max(fitme), center, 3)
            params['mean'].append(mu)
            params['sig'].append(sigma)
            params['A'].append(A / np.sqrt(2 * np.pi * sigma ** 2)) #A is now amplitude

        q = [(0 < sig < 10 and 0 < A < 2 * np.max(handles['strikeamp'])) for sig, A in zip(params['sig'], params['A'])]
        datastat = {
            'mean': [params['mean'][i] for i in range(len(q)) if q[i]],
            'sig': [params['sig'][i] for i in range(len(q)) if q[i]],
            'A': [params['A'][i] for i in range(len(q)) if q[i]]
        }
    else:
        datastat = 1

    return datastat, handles, peakloc


#Check intervals between events (ISI)
def ISI_check(handles, tbin, corrt):
    """
    % Uses events locations in peakloc and timestream stored in handles structure
    % to generate interspike interval plots compared to simple single-rate
    % poisson model. The sampling time tbin and the correlation time (~width of a
    % burst) corrt are used to exclude events that are too close in time.
    """
    nsamples = len(handles['strikeamp'].flatten()) #number of sample bins in concatenated data
    nhistbins = 100 #consider sampletime*nhistbins seconds of event spacing (typically 100ms)
    #Check interspike interval using generated time stream and a single
    #population model prediction based on random times and same avg rate
    
    striketimesm = np.sort(np.random.randint(1, nsamples, int(np.sum(handles['strike'])))) #chronological order of event bins
    peakshm = np.roll(striketimesm, 1)
    peakdiffm = striketimesm - peakshm #This spacing (ISI) gives statistical measure of

    #For reference, model of fsingle population distribution with same avg event rate
    #If multiple species, events shoud still be uncorrelated like 1 species
    #of higher rate
    xoutm = np.arange(1, nhistbins + 1)
    nm, _ = np.histogram(peakdiffm[1:-1], bins=xoutm) #hist of random samples for control
    plt.figure()
    plt.semilogy(xoutm[:-1] * tbin, nm, '-o')
    rate = len(striketimesm) / (tbin * nsamples)
    subtitle = f"Random event model w/ single event rate: {rate:.4f} events/sec"
    plt.title(f"Random Interspike Interval Dist Compared to Data\n{subtitle}")
    plt.xlabel('Time between events')
    plt.ylabel('Number of events')

    # Do again with real data shifted for comparison; should yield same powerlaw
    #slope; Overplot distributions
    qq = handles['strikeamp'].flatten() > 0
    striketimesd = np.arange(1, nsamples + 1)[qq] #need transpose to switch index order; indices where strikes occured
    rtd = np.sort(striketimesd)
    peakshd = np.roll(rtd, 1)
    peakdiffd = rtd - peakshd
    xoutd = np.arange(1, nhistbins + 1)
    nd, _ = np.histogram(peakdiffd[1:-1], bins=xoutd) #hist of random samples for control
    plt.semilogy(xoutd[:-1] * tbin, nd, '-ro')
    plt.legend(['Noiseless Model', 'Simulated Data'], loc='upper right')

    """
    %Find real data interspike intervals that don't violate correlation time
    %parameter corrt; keep first event of two that are too close; arbitrary
    %Would be better to keep larger amplitude event
    """
    dontkeep = np.abs(peakdiffd) < round(corrt / tbin) #logical indexing
    rr = rtd[dontkeep]
    handles['strikeamp'][rr - 1] = 0 #zero timing violation amplitude
    handles['strike'][rr - 1] = 0 #zero the strike flag as well

    #Now show with sub correlation time events removed
    print('-----------------------------------------')
    ntotevents = int(np.sum(handles['strike']))
    print(f"Removing {len(rr)} intertime violation events from {ntotevents} total events")
    qq = handles['strikeamp'].flatten() > 0
    striketimesd = np.arange(1, nsamples + 1)[qq] #need transpose to switch index order; indices where strikes occured
    rtd = np.sort(striketimesd)
    peakshd = np.roll(rtd, 1)
    peakdiffd = rtd - peakshd
    nd, _ = np.histogram(peakdiffd[1:-1], bins=xoutd) #hist of random samples for control
    plt.semilogy(xoutd[:-1] * tbin, nd, '-go')

    # Fit exponential model to distribution
    def exp_model(x, a, b):
        return a * np.exp(b * -1 * x)

    xdata = xoutd[:-1] * tbin
    ydata = nd
    popt, _ = curve_fit(exp_model, xdata, ydata, p0=[1, 1])
    plt.semilogy(xdata, exp_model(xdata, *popt))
    plt.legend(['random control', 'Input data', 'Remove timing violations'], loc='upper right')
    plt.show()

#Outputs code version information
def Information(versionnum):
    #Display version and hold introductory information
    prepfile = f"Pre_{versionnum}"
    tstr = datetime.now().strftime("%d%b%y-%H%M%S")
    print(f"version number: {versionnum}")
    return prepfile, tstr

#PART 2

#Initialize
versionnum = "v1p1"
print("FOR TIME STAMP DATA FILES ONLY. Version information and details of modeling at bottom of mlx code ' ...' below the Information function")

#Choose subdirectory to process
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("Parameter files", "*.par")], title="Select preprocessing parameter file")
PrePathName, PreFileName = os.path.split(file_path)
print(f"Using preprocessing parameter file: {PrePathName}{PreFileName}")


#Read Parameter File

#The parameter file is assumed in same directory as data
#Parameter file should have 5 header lines at top 

paramfile = os.path.join(PrePathName, PreFileName)
#parse header for info
with open(paramfile, "r") as f:
    lines = f.readlines()
lines = [line for line in lines if line!="\n"]
readmePRE = {}
textdata = lines[0:len(lines)-1]
textdata[len(textdata)-1] = textdata[len(textdata)-1].split()
colheaders = textdata[-1]
data = [float(x) for x in lines[-1].split()]
readmePRE["data"] = data
readmePRE["textdata"] = textdata
readmePRE["colheaders"] = colheaders

preproparams = readmePRE["data"]
catfilenames = readmePRE["textdata"][5:len(readmePRE["textdata"])-1]
colheader = readmePRE["colheaders"]

#parameters: datatype sampletime driftwinA driftwinB basethreshA basethreshB corrt dofit colA colB colC
#some of these are obsolete. Datatype is used in case there is an update to
#formatting. Default is 1 for current TAMU format. Note, data files
#typically have txt header with words (say about first 13 lines).
datatype = readmePRE["data"][0] #Flag for possible changes in data formatting
Tottime = readmePRE["data"][1] #Total recording time
MTclock = readmePRE["data"][2] #TAMU photon counting board clock
tbin = readmePRE["data"][3] #Binning zine in seconds to form time series
driftwinA = readmePRE["data"][4] #integer bin factor multiplied by 2000
driftwinB = readmePRE["data"][5] #Slow drift removal
driftwinC = readmePRE["data"][6]
basethreshA = readmePRE["data"][7] #Low amplitude spike hash removal (integer cnts)
basethreshB = readmePRE["data"][8] #Does change amplitude of remaining data
basethreshC = readmePRE["data"][9]
corrt = readmePRE["data"][10] #Minimal time acceptable between events
dofit = readmePRE["data"][11] #Flag for Gaussian fit to events (usually 0)
colA = int(readmePRE["data"][12]) #Column number where A data located; 0 if no data
colB = int(readmePRE["data"][13]) #Column number where B data located; 0 if no data
colC = int(readmePRE["data"][14]) #Column number where C data located; 0 if no data

numcatfiles = len(catfilenames) #photon counts in time bins

I_A = [0]
I_B = [0]
I_C = [0]

#Set up time series using tbin and total time
print(f"Estimated number of samples per file at MT clock rate: {Tottime / MTclock}")
print(f"Estimated timestream dimension per file after requested binning time: {Tottime / tbin}")

#Determine edges of time bins for photon count per bin histogram to follow
Nafterbin = round(Tottime / tbin)
tedge = np.arange(0, Nafterbin + 1) * tbin


#Concatenate data files into 3 color channel vectors

print('Concatenating files using single baseline offset defined in params file')
print('Confirm mean and rms of all files similar!!')


for i in range(numcatfiles):
    filename = os.path.join(PrePathName, catfilenames[i].strip("\n"))

    with open(filename, "r") as f:
        lines = f.readlines()
    data_start_index = None
    for i, line in enumerate(lines):
        if "***end header***" in line:
            data_start_index = i + 2
            break
    if data_start_index is None:
        raise ValueError("Bad file... couldn't find header")
    
    temp = pd.read_csv(filename, delimiter='\t', skiprows=data_start_index+1, header=None) #Tab delimited
    #Handle possible NaN values - should be very infrequent
    ss = temp.mean(axis = 1) #use to find rows with NaN's then drop those rows
    allnans = np.where(~np.isfinite(ss))[0]
    numnans = len(allnans)
    print(f"Number of NaN containing rows removed: {numnans}")
    temp = temp.drop(index = allnans)

    #Form up to 3 color channels
    if colA != 0:
        print("Add a data file for A Channel")
        tempA = temp.iloc[:, colA - 1].to_numpy() * MTclock * 2 #Arrival times in seconds for one file
        #factor of 2 is a feature of the BH board
        tempA = tempA[tempA != 0] #Remove 0's that pad the data columns
        #Apply binning set by valiable tbin=readmePRE["data"][3]
        histA, _ = np.histogram(tempA, bins=tedge)
        I_A.extend(histA.tolist())
        plt.clf() #Clear current figure
        plt.hist(tempA, bins=tedge)
        plt.pause(1)
    if colB != 0:
        print("Add a data file for B Channel")
        tempB = temp.iloc[:, colB - 1].to_numpy() * MTclock * 2 #Arrival times in seconds for one file
        #factor of 2 is a feature of the BH board
        tempB = tempB[tempB != 0] #Remove 0's that pad the data columns
        #Apply binning set by valiable tbin=readmePRE["data"][3]
        histB, _ = np.histogram(tempA, bins=tedge)
        I_A.extend(histB.tolist())
        plt.clf() #Clear current figure
        plt.hist(tempB, bins=tedge)
        plt.pause(1)
    if colC != 0:
        print("Add a data file for C Channel")
        tempC = temp.iloc[:, colC - 1].to_numpy() * MTclock * 2 #Arrival times in seconds for one file
        #factor of 2 is a feature of the BH board
        tempC = tempC[tempC != 0] #Remove 0's that pad the data columns
        #Apply binning set by valiable tbin=readmePRE["data"][3]
        histB, _ = np.histogram(tempC, bins=tedge)
        I_A.extend(histA.tolist())
        plt.clf() #Clear current figure
        plt.hist(tempA, bins=tedge)
        plt.pause(1)

#Remove first null data point and initialize original timestream output and
#find spike-free rms, mean and median for possible offset correction in BAS code
handlesA = {}

if colA!=0:
    I_A = I_A[1:]
    handlesA['base'] = I_A
    rms1 = np.std(I_A, ddof=1)
    qrms1 = I_A < 5 * rms1 # Data that isn't part of major spike activity
    handlesA["rms"] = np.std(np.array(I_A)[qrms1], ddof=1)
    handlesA["median"] = np.median(np.array(I_A)[qrms1])
    handlesA["mean"] = np.mean(np.array(I_A)[qrms1])
else:
    print("No A Channel data specified")
    handlesA['base'] = 0
    datastatA = 0
    peaklocA = 0

handlesB = {}
if colB!=0:
    I_B = I_B[1:]
    handlesB['base'] = I_B
    rms1 = np.std(I_B, ddof=1)
    qrms1 = I_B < 5 * rms1 # Data that isn't part of major spike activity
    handlesB["rms"] = np.std(np.array(I_B)[qrms1], ddof=1)
    handlesB["median"] = np.median(np.array(I_B)[qrms1])
    handlesB["mean"] = np.mean(np.array(I_B)[qrms1])
else:
    print("No B Channel data specified")
    handlesB['base'] = 0
    datastatB = 0
    peaklocB = 0

handlesC = {}
if colC!=0:
    I_C = I_C[1:]
    handlesC['base'] = I_C
    rms1 = np.std(I_C, ddof=1)
    qrms1 = I_C < 5 * rms1 # Data that isn't part of major spike activity
    handlesC["rms"] = np.std(np.array(I_C)[qrms1], ddof=1)
    handlesC["median"] = np.median(np.array(I_C)[qrms1])
    handlesC["mean"] = np.mean(np.array(I_C)[qrms1])
else:
    print("No C Channel data specified")
    handlesC['base'] = 0
    datastatC = 0
    peaklocC = 0

#Apply slow drift correction to baseline (paramsfile parameter 5&6)
if driftwinA != 0 and colA != 0:
    I_A, basedrift = basedrift_data(I_A, driftwinA, "A")
    handlesA['basedriftA'] = basedrift
if driftwinB != 0 and colB != 0:
    I_B, basedrift = basedrift_data(I_B, driftwinB, "B")
    handlesB['basedriftB'] = basedrift
if driftwinC != 0 and colC != 0:
    I_C, basedrift = basedrift_data(I_C, driftwinC, 'C')
    handlesC['basedriftC'] = basedrift

#Apply baseline threshold filter (paramsfile parameters 7&8) to subtract offset
if basethreshA != 0 and colA != 0:
    I_A, baseoffset = basethresh_data(I_A, basethreshA, 'A')
    handlesA['baseoffsetA'] = baseoffset
if basethreshB != 0 and colB != 0:
    I_B, baseoffset = basethresh_data(I_B, basethreshB, 'B')
    handlesB['baseoffsetB'] = baseoffset
if basethreshC != 0 and colC != 0:
    I_C, baseoffset = basethresh_data(I_C, basethreshC, 'C')
    handlesC['baseoffsetC'] = baseoffset

#Determine location and amplitudes of spikes in concatenated dataset
if colA != 0:
    datastatA, handlesA, peaklocA = burst_data(I_A, tbin, corrt, dofit, 'A', handlesA)
if colB != 0:
    datastatB, handlesB, peaklocB = burst_data(I_B, tbin, corrt, dofit, 'B', handlesB)
if colC != 0:
    datastatC, handlesC, peaklocC = burst_data(I_C, tbin, corrt, dofit, 'C', handlesC)

#Check rate of strikes and single particle limit
#Use a few stats to determine if single particle limit holds for this dataset

do_strikestats = True

if do_strikestats:
    #Determine some statistics about detecting spikes
    if colA != 0:
        print(">>>>>>>Channel A")
        numbins = len(handlesA["strike"])
        numspA = np.sum(handlesA["strike"])
        print(f"Number events above A threshold: {numspA}")
        #If random, prob of spikes in singnle bin and multiple adjacent bins
        p0bin1A = (numbins - numspA) / numbins
        p1bin1A = numspA/numbins
        print(f"Prob of 0 event in a bin: {p0bin1A}")
        print(f"Prob of 1 event in a bin: {p1bin1A}")
        print(f"Prob of 2 events in a bin based on P(n=1)^2: {p1bin1A**2}")
        print(" ")
        print(f"Fraction of events that are double strikes {(p1bin1A**2 * numbins)/numspA}")

        #average value and prob of a subthreshold event in subthresh band
        avgsubampA = np.mean(handlesA['subthreshstrikeamp'])
        probsubampA = np.sum(handlesA['subthreshstrikeamp']) / numbins
        print('avg A channel contribution of subthresh events is')
        overampsubA = probsubampA * avgsubampA
        print(overampsubA)
    
    #Prob of subthreshold events contributing to more than 10%
    if colB != 0:
        print('>>>>>>>Channel B')
        numbins = len(handlesB['strike'])
        numspB = np.sum(handlesB['strike'])
        print(f'Number events above B threshold: {numspB}')

        #If random, prob of spikes in singnle bin and multiple adjacent bins
        p0bin1B = (numbins - numspB) / numbins
        p1bin1B = numspB/numbins


        print(f'Prob of 0 event in a bin: {p0bin1B}')
        print(f'Prob of 1 event in a bin: {p1bin1B}')
        print(f'Prob of 2 events in a bin based on P(n=1)^2: {p1bin1B ** 2}')
        print('')
        print(f'Fraction of events that are double strikes: {(p1bin1B ** 2 * numbins) / numspB}')

        #average value and prob of a subthreshold event in subthresh band
        avgsubampB = np.mean(handlesB['subthreshstrikeamp'])
        probsubampB = np.sum(handlesB['subthreshstrike']) / numbins
        print('avg B channel contribution of subthresh events is')
        overampsubB = probsubampB * avgsubampB
        print(overampsubB)

    if colC != 0:
        print('>>>>>>>Channel C')
        numbins = len(handlesC['strike'])
        numspC = np.sum(handlesC['strike'])
        print(f'Number events above C threshold: {numspC}')
        #If random, prob of spikes in singnle bin and multiple adjacent bins
        p0bin1C = (numbins - numspC) / numbins
        p1bin1C = numspC/numbins

        print(f'Prob of 0 event in a bin: {p0bin1C}')
        print(f'Prob of 1 event in a bin: {p1bin1C}')
        print(f'Prob of 2 events in a bin based on P(n=1)^2: {p1bin1C ** 2}')
        print('')
        print(f'Fraction of events that are double strikes: {(p1bin1C ** 2 * numbins) / numspC}')

        #If random, prob of spikes in singnle bin and multiple adjacent bins
        avgsubampC = np.mean(handlesC['subthreshstrikeamp'])
        probsubampC = np.sum(handlesC['subthreshstrike']) / numbins
        print('avg C channel contribution of subthresh events is')
        overampsubC = probsubampC * avgsubampC
        print(overampsubC)


#Cross-correlation of datasets
#Determine fraction of events common to both channels using raw signal
do_xcorr = True

def normalized_xcorr(x, y):
    x = np.array(x)
    y = np.array(y)

    x = x - np.mean(x)
    y = y - np.mean(y)

    corr = correlate(x, y, mode="full")
    lags = correlation_lags(len(x), len(y), mode="full")

    norm_factor = np.std(x, ddof=1) * np.std(y, ddof=1) * len(x)
    if norm_factor == 0:
        return lags, np.zeros_like(corr)  # Avoid divide by zero
    corr /= norm_factor

    return lags, corr

if colA * colB * do_xcorr != 0:
    print("Both channels active: finding fractional overlap of events ...")
    #Find typical upper bound using 10 largest strikes
    lags, c = normalized_xcorr(handlesA['base'], handlesB['base'], 5)
    plt.figure()
    plt.stem(lags, c)
    plt.title("Normalized Cross-correlation of RAW SIGNAL vs bin lag")

    lags, c = normalized_xcorr(handlesA["strike"], handlesB["strike"], 5)
    plt.figure()
    plt.stem(lags, c)
    plt.title("Normalized Cross-correlation of EVENT TIMES vs bin lag")

    sortA = np.sort(handlesA['strikeamp'])[::-1] #sort descending
    sortB = np.sort(handlesB['strikeamp'])[::-1]
    UBA = np.median(sortA[:10])
    UBB = np.median(sortB[:10])
    AQ1 = handlesA['strikeamp'] > 0.5 * UBA
    BQ1 = handlesB['strikeamp'] > 0.5 * UBB

    Atemp1 = handlesA['strike'] * AQ1
    Btemp1 = handlesB['strike'] * AQ1
    plt.figure()
    lags, c = normalized_xcorr(Atemp1, Btemp1, 5)
    plt.stem(lags, c)

    Atemp2 = handlesA['strike'] * BQ1
    Btemp2 = handlesB['strike'] * BQ1
    plt.figure()
    lags, c = normalized_xcorr(Atemp2, Btemp2, 5)
    plt.stem(lags, c)
    plt.legend(['All events', 'Top 50%; A-triggered', 'Top 50%; B-triggered'])

    numberA = np.sum(handlesA['strike'][AQ1])
    numberB = np.sum(handlesB['strike'][BQ1])

    print('Event zero lag A-triggered, B-triggered, and auto-correlation AA, BB = 1 check')
    fracoverlapAB = np.sum(handlesA['strike'] * handlesB['strike'] * AQ1) / numberA
    fracoverlapBA = np.sum(handlesB['strike'] * handlesA['strike'] * BQ1) / numberB
    fracoverlapAA = np.sum(handlesA['strike'] * handlesA['strike'] * AQ1) / numberA
    fracoverlapBB = np.sum(handlesB['strike'] * handlesB['strike'] * BQ1) / numberB

    handlesA['overlapAB'] = fracoverlapAB
    handlesB['overlapAB'] = fracoverlapAB

if colC * colB * do_xcorr != 0:
    print("Both channels active: finding fractional overlap of events ...")
    #Find typical upper bound using 10 largest strikes
    lags, c = normalized_xcorr(handlesC['base'], handlesB['base'], 5)
    plt.figure(); plt.stem(lags, c); plt.title("X-Corr RAW: C vs B")

    lags, c = normalized_xcorr(handlesC['strike'], handlesB['strike'], 5)
    plt.figure(); plt.stem(lags, c); plt.title("X-Corr EVENTS: C vs B")

    sortC = np.sort(handlesC['strikeamp'])[::-1] #sort descending
    sortB = np.sort(handlesB['strikeamp'])[::-1]
    UBC = np.median(sortC[:10])
    UBB = np.median(sortB[:10])
    CQ1 = handlesC['strikeamp'] > 0.5 * UBC
    BQ1 = handlesB['strikeamp'] > 0.5 * UBB

    Ctemp1 = handlesC['strike'] * AQ1
    Btemp1 = handlesB['strike'] * AQ1
    plt.figure(); lags, c = normalized_xcorr(Ctemp1, Btemp1, 5); plt.stem(lags, c)

    Ctemp2 = handlesC['strike'] * BQ1
    Btemp2 = handlesB['strike'] * BQ1
    plt.figure(); lags, c = normalized_xcorr(Ctemp2, Btemp2, 5); plt.stem(lags, c)

    plt.legend(['All events', 'Top 50%; C-triggered', 'Top 50%; B-triggered'])

    numberC = np.sum(handlesC['strike'][CQ1])
    numberB = np.sum(handlesB['strike'][BQ1])

    fracoverlapCB = np.sum(handlesC['strike'] * handlesB['strike'] * CQ1) / numberC
    fracoverlapBC = np.sum(handlesB['strike'] * handlesC['strike'] * BQ1) / numberB
    handlesC['overlapCB'] = fracoverlapCB
    handlesB['overlapCB'] = fracoverlapCB

if colA * colC * do_xcorr != 0:
    print("Both channels active: finding fractional overlap of events ...")
    #Find typical upper bound using 10 largest strikes
    lags, c = normalized_xcorr(handlesA['base'], handlesC['base'], 5)
    plt.figure(); plt.stem(lags, c); plt.title("X-Corr RAW: A vs C")

    lags, c = normalized_xcorr(handlesA['strike'], handlesC['strike'], 5)
    plt.figure(); plt.stem(lags, c); plt.title("X-Corr EVENTS: A vs C")

    sortA = np.sort(handlesA['strikeamp'])[::-1] #sort descending
    sortC = np.sort(handlesC['strikeamp'])[::-1]
    UBA = np.median(sortA[:10])
    UBC = np.median(sortC[:10])
    AQ1 = handlesA['strikeamp'] > 0.5 * UBA
    CQ1 = handlesC['strikeamp'] > 0.5 * UBC

    Atemp1 = handlesA['strike'] * AQ1
    Ctemp1 = handlesC['strike'] * AQ1
    plt.figure(); lags, c = normalized_xcorr(Atemp1, Ctemp1, 5); plt.stem(lags, c)

    Atemp2 = handlesA['strike'] * CQ1
    Ctemp2 = handlesC['strike'] * CQ1
    plt.figure(); lags, c = normalized_xcorr(Atemp2, Ctemp2, 5); plt.stem(lags, c)

    plt.legend(['All events', 'Top 50%; A-triggered', 'Top 50%; C-triggered'])

    numberA = np.sum(handlesA['strike'][AQ1])
    numberC = np.sum(handlesC['strike'][CQ1])

    fracoverlapAC = np.sum(handlesA['strike'] * handlesC['strike'] * AQ1) / numberA
    fracoverlapCA = np.sum(handlesC['strike'] * handlesA['strike'] * CQ1) / numberC
    handlesA['overlapAC'] = fracoverlapAC
    handlesC['overlapAC'] = fracoverlapAC

#Save preprocessing    
savebasresults = True

if savebasresults:
    savepredone = save_preprocessed(
        PrePathName, 
        PreFileName,
        datastatA,
        datastatB,
        datastatC,
        handlesA,
        handlesB,
        handlesC,
        peaklocA,
        peaklocB,
        peaklocC,
        readmePRE,
        versionnum
    )
