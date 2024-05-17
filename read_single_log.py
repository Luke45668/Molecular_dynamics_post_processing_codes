#%%
import os
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import regex as re
import pandas as pd
import sigfig
# plt.rcParams.update(plt.rcParamsDefault)
# plt.rcParams['text.usetex'] = True
#from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import time 
import scipy.stats
from datetime import datetime
import h5py as h5 
import multiprocessing as mp
from log2numpy import *
from mom2numpy import *
from velP2numpy import *
from dump2numpy import * 


import glob 
from post_MPCD_MP_processing_module import *
colour = [
 'black',
 'blueviolet',
 'cadetblue',
 'chartreuse',
 'coral',
 'cornflowerblue',
 'crimson',
 'darkblue',
 'darkcyan',
 'darkgoldenrod',
 'darkgray']


from scipy.optimize import curve_fit
def linearfunc(x,a):
    return (a*x)

filepath="/Users/luke_dev/Documents/simulation_test_folder/langevin_test"
log_name="log.langevin_val_no1_single_p_valid_1256_0_23_1000_1000_1000000000_0.2"
thermo_vars='         KinEng         PotEng          Temp          TotEng        c_msd[3]   '
log_data=log2numpy_reader(log_name,
                            filepath,
                            thermo_vars)
time=1000000000*0.00101432490424207
timepoints=np.linspace(0,time,log_data.shape[0])
plt.plot(timepoints,log_data[:,5])
popt,pcov=curve_fit(linearfunc,timepoints,log_data[:,5])
plt.plot(timepoints,(popt[0]*(timepoints)))
plt.show()

# %%
