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
# plt.rcParams['text.usetex'] = True
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
from scipy.stats import chi2 
from stat_tests import *


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

os.chdir('/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/Simulation_run_folder/dist_test_run_2_'+str(j_))

area_vector_1d=np.load("area_vector_summed_1d_test_erate_"+str(erate[0])+"_bk_"+str(bending_stiffness)+"_M_"+str(rho)+"_L_"+str(box_size)+"_no_timesteps_"+str(no_timesteps)+".npy")
# will need another dimension when there are more timesteps
area_vector_3d=np.reshape(area_vector_1d,(j_,3))
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




bin_count = int(np.ceil(np.log2((j_))) + 1)# sturges rule
# plot theta histogram
pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
frequencies_theta= np.histogram(spherical_coordinates_area_vector[:,1],bins=bin_count)[0]

# null hypothesis: data fits a uniform distribution ,no effect, random number generator in lammps works
# alternate hypothesis: data no longer fits a uniform distribution,lammps random number generator causes an effect
deg_f = bin_count-1 
sig = 0.05

chi_stat_test_uniform(sig,deg_f,frequencies_theta)

# theta histogram 

plt.hist((spherical_coordinates_area_vector[:,1]),density=True, bins=bin_count)
plt.xticks(pi_theta_ticks, pi_theta_tick_labels)
plt.xlabel('$\\theta$')
#plt.tight_layout()
plt.savefig("theta_histogram_"+str(j_)+"_points.pdf",dpi=1200)
plt.show()



#%% Producing sin histogram 
import sympy as sy

def sin_for_sym(x):

    return np.sin(x)


x = sy.Symbol("x") 
bounds=np.arange(0,np.pi+np.pi/bin_count,np.pi/bin_count )
#bounds=np.linspace(0,np.pi,bin_count )

area_under_sin_curve_interval =np.zeros(bin_count)
mid_point_of_bound=np.zeros(bin_count)

for i in range(int(bounds.size-1)):
    print(sy.integrate(sy.sin(x), (x, bounds[i], bounds[i+1])))
    area_under_sin_curve_interval[i]=sy.integrate(sy.sin(x), (x, bounds[i], bounds[i+1]))
    mid_point_of_bound[i]=(bounds[i]) +np.pi/bin_count


total_area_under_curve=np.sum(area_under_sin_curve_interval)

# have to normalise the areas so it can be scaled by any value of j_
freq_sin_from_integral=np.round(area_under_sin_curve_interval*j_/2).astype('int')





#%% phi test 

bin_count_expected = int(np.ceil(np.log2((j_))) + 1)

pi_phi_ticks=[ 0,np.pi/4, np.pi/2,3*np.pi/4,np.pi]
pi_phi_tick_labels=[ '0','π/4', 'π/2','3π/4' ,'π']
frequencies_phi= np.histogram(spherical_coordinates_area_vector[:,2],bins=bin_count)[0]

deg_f = bin_count-1 
sig = 0.05
chi_stat_test_custom(sig,deg_f,frequencies_phi,freq_sin_from_integral)

# plot phi hist



plt.hist(spherical_coordinates_area_vector[:,2],density=True, bins=bin_count)
plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
plt.xlabel('$\phi$')
#plt.tight_layout()
#plt.savefig("phi_histogram_"+str(j_)+"_points.pdf",dpi=1200)
plt.show()

        




# %%
