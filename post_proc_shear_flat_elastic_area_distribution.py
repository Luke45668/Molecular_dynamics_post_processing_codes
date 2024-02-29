#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# This script will calculate the MPCD stress tensor for a pure fluid under forward NEMD using hdf5 files and python multiprocessing
# to ensure fast analysis  
# """
# Importing packages
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import regex as re
import pandas as pd
import multiprocessing as mp
from multiprocessing import Process
import time
import h5py as h5
import seaborn as sns
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
#from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
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

# key inputs 
# no_SRD=1038230
# box_size=47
# no_SRD=506530
# box_size=37
# no_SRD=121670
# box_size=23
# no_SRD=58320
# box_size=18
no_SRD=2160
box_size=6
# no_SRD=270
# box_size=3
# no_SRD=2560
# box_size=8
no_SRD=60835
box_size=23
#nu_bar=3
#delta_t_srd=0.014872025172594354
#nu_bar=0.9 
#rho=10
#delta_t_srd=0.05674857690605889
rho=5
delta_t_srd=0.05071624521210362

box_vol=box_size**3

erate= np.array([0.001])

no_timesteps=300
# estimating number of steps  required
strain=3
delta_t_md=delta_t_srd/10
strain_rate= np.array([0.01,0.001,0.0001])
number_steps_needed= np.ceil(strain/(strain_rate*delta_t_md))
dump_freq=10
spring_stiffness =np.array([20,40,80,100])
bending_stiffness=10000
j_=100
#rho=5
realisation_index=np.array([1,2,3])

os.chdir('/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/dist_test')

area_vector_1d=np.load("area_vector_summed_1d_test_erate_"+str(erate[0])+"_bk_"+str(bending_stiffness)+"_M_"+str(rho)+"_L_"+str(box_size)+"_no_timesteps_"+str(no_timesteps)+".npy")
area_vector_3d=np.reshape(area_vector_1d[:3000],(j_,3))
# not so simple to calculate spherical coords, using mathematics convention from wiki

spherical_coordinates_area_vector=np.zeros((j_,3))

x=area_vector_3d[:,0]
y=area_vector_3d[:,1]
z=area_vector_3d[:,2]
# radial coord
spherical_coordinates_area_vector[:,0]=np.sqrt((x**2)+(y**2)+(z**2))
# theta coord 
spherical_coordinates_area_vector[:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))
# phi coord
spherical_coordinates_area_vector[:,2]=np.arccos(z/spherical_coordinates_area_vector[:,0])

# plot theta histogram
plt.hist(spherical_coordinates_area_vector[:,1],density=True, bins=int(j_/10))
plt.show()

#plot phi histogram 
plt.hist(spherical_coordinates_area_vector[:,2],density=True, bins=int(j_/10))
plt.show()

#%%

sns.displot(data=spherical_coordinates_area_vector[:,1], kind="kde") #bins=int(j_/10))
plt.xlabel('$\\theta$')
plt.tight_layout()
plt.savefig("theta_distribution_"+str(j_)+"_points.pdf",dpi=1200)
plt.show()
sns.displot(data=spherical_coordinates_area_vector[:,2],  kind="kde")#, bins=int(j_/10))
plt.xlabel('$\phi$')
plt.tight_layout()
plt.savefig("phi_distribution_"+str(j_)+"_points.pdf",dpi=1200)
plt.show()

        

# %%
