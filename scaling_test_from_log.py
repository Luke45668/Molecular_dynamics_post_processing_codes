##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script will calculate the MPCD stress tensor for a pure fluid under forward NEMD using hdf5 files 
"""
#%% Importing packages
import os
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import regex as re
import pandas as pd
import sigfig
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
#from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import h5py as h5 
import multiprocessing as mp
import mmap as m


path_2_post_proc_module= '/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/LAMMPS python run and analysis scripts/Analysis codes'
#os.chdir(path_2_post_proc_module)
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

box_size=34
no_SRD=int((34**3)*5)
bending_stiffness=np.array([10000])
internal_stiffness=np.array([20,60,100])
np_req=np.array([8,16,32,36])
erate= np.array([0.0005,0.001,0.002,0.005,0.01,0.1])
no_timesteps=10000
repeats=3
realisation_index=np.arange(0,repeats,1)
VP_general_name_string='vel.*'

Mom_general_name_string='mom.*'

log_general_name_string='log.*'
                         #log.H20_no466188_wall_VACF_output_no_rescale_
TP_general_name_string='temp.*'

dump_general_name_string_before='*'+str(no_timesteps)+'*before*.h5'

# multi
#filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/scaling_tests/flat_elastic/box_"+str(box_size)+"_multi"
# single cut off data 
filepath="/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/scaling_tests/flat_elastic/box_"+str(box_size)+"_single_cutoff"


log_realisation_name_info_before=VP_and_momentum_data_realisation_name_grabber(TP_general_name_string,log_general_name_string,VP_general_name_string,Mom_general_name_string,filepath,dump_general_name_string_before)
realisation_name_log=log_realisation_name_info_before[4]
count_log=log_realisation_name_info_before[5]

var_1=internal_stiffness
var_2=np_req
start_np_string="Loop time of ** on "
end_np_string=" procs for "+str(no_timesteps)

#def scaling_data_from_log(filepath,realisation_name_log,repeats,var_1,var_2):

os.chdir(filepath)
start_np_string=bytes("Loop time of",'utf-8')
end_np_string=bytes(" procs for "+str(no_timesteps),'utf-8')
start_perf_string=bytes("Performance:",'utf-8')
end_perf_string=bytes("CPU use with",'utf-8')


scaling_data_array=np.zeros((repeats,erate.size,var_1.size,var_2.size))
parallel_efficiency_data=np.zeros((repeats,erate.size,var_1.size,var_2.size))

#%%
for i in range(count_log):
    find_K= int(realisation_name_log[i].split('_')[-1])
    K_index=np.where(find_K==internal_stiffness)[0][0]
    find_j=int(realisation_name_log[i].split('_')[6])
    j_index=np.where(find_j==realisation_index)[0][0]
    find_erate=float((realisation_name_log[i].split('_')[16]))
    erate_index=np.where(find_erate==erate)[0][0]



    with open(realisation_name_log[i],'r') as f:
        log= m.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        if log.find(start_np_string) != -1:
           print('np string exist in a file')
           start_np_byte_index=log.rfind(start_np_string)

           if log.find(end_np_string) != -1:
                print("np string exists")
                end_np_byte_index=log.rfind(end_np_string)

                size_np_info=end_np_byte_index-start_np_byte_index

        if log.find(start_perf_string) != -1:
           print('perf string exist in a file')
           start_perf_byte_index=log.rfind(start_perf_string)

           if log.find(end_perf_string) != -1:
                print("perf string exists")
                end_perf_byte_index=log.rfind(end_perf_string)

                size_perf_info=end_perf_byte_index-start_perf_byte_index

        log.seek(start_np_byte_index)
        log_bytes=log.read(end_np_byte_index)
        np_info_dirty=str(log_bytes[0:size_np_info])
        np_info_clean=int(np_info_dirty[-3:-1])# splicing number of procs
        np_index=np.where(np_info_clean==np_req)[0][0]

        log.seek(start_perf_byte_index)
        log_bytes=log.read(end_perf_byte_index)
        perf_info_dirty=str(log_bytes[0:size_perf_info])
        perf_info_split=perf_info_dirty.split(' ')
        timesteps_per_sec=float(perf_info_split[3])
        dirty_parallel_eff=perf_info_split[6]
        clean_parallel_eff=float(dirty_parallel_eff[-5:-1])

        scaling_data_array[j_index,erate_index,K_index,np_index]=timesteps_per_sec
        parallel_efficiency_data[j_index,erate_index,K_index,np_index]=clean_parallel_eff
#%% 
        
scaling_data_array_real_mean=np.mean(np.mean(scaling_data_array,axis=0),axis=0)
parallel_efficiency_data_real_mean=np.mean(np.mean(parallel_efficiency_data,axis=0),axis=0)
# np.save("scaling_data_array_real_mean_multi",scaling_data_array_real_mean)
# np.save("parallel_efficiency_data_real_mean_multi",parallel_efficiency_data_real_mean)
np.save("scaling_data_array_real_mean_single",scaling_data_array_real_mean)
np.save("parallel_efficiency_data_real_mean_single",parallel_efficiency_data_real_mean)

for k in range(internal_stiffness.size):
    plt.scatter(np_req[:],scaling_data_array_real_mean[k,:], label="$K="+str(internal_stiffness[k])+"$")
    plt.xlabel("Number of processors")
    plt.ylabel("$N_{\Delta t} s^{-1}$", rotation=0,labelpad=20)
    plt.legend()
plt.show()

for k in range(internal_stiffness.size):
    plt.plot(np_req[:],parallel_efficiency_data_real_mean[k,:],label="$K="+str(internal_stiffness[k])+"$")
    plt.xlabel("Number of processors")
    plt.ylabel("\% PE", rotation=0)
    plt.legend()
plt.show()


                    
seconds_in_a_day=86400
max_number_of_timesteps_possible=2*seconds_in_a_day*np.max(scaling_data_array)
print("max number of timesteps possible in 48hrs is:",max_number_of_timesteps_possible)



#Loop time of 364.96 on 8 procs for 10000 steps with 196526 atoms
                 
#Performance: 600.324 tau/day, 27.400 timesteps/s, 5.385 Matom-step/s
#99.4% CPU use with 8 MPI tasks x 1 OpenMP threads
    

#%% making plot of both 
os.chdir("/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/scaling_tests/flat_elastic/box_"+str(box_size)+"_multi")
scaling_data_array_real_mean_multi=np.mean(np.load("scaling_data_array_real_mean_multi.npy"),axis=0)
parallel_efficiency_data_real_mean_multi=np.mean(np.load("parallel_efficiency_data_real_mean_multi.npy"),axis=0)
max_number_of_timesteps_possible_multi=scaling_data_array_real_mean_multi*86400*2

os.chdir("/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/scaling_tests/flat_elastic/box_"+str(box_size)+"_single_cutoff")
scaling_data_array_real_mean_single=np.mean(np.load("scaling_data_array_real_mean_single.npy"),axis=0)
parallel_efficiency_data_real_mean_single=np.mean(np.load("parallel_efficiency_data_real_mean_single.npy"),axis=0)
max_number_of_timesteps_possible_single=scaling_data_array_real_mean_single*86400*2

plt.scatter(np_req,scaling_data_array_real_mean_multi,label="Multi",marker='x')
plt.scatter(np_req,scaling_data_array_real_mean_single,label="Single")
plt.xlabel("Number of processors")
plt.ylabel("$N_{\Delta t} s^{-1}$", rotation=0,labelpad=20)
plt.legend()
plt.tight_layout()
plt.savefig("/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/scaling_tests/flat_elastic/box_size_"+str(box_size)+"_timesteps_per_second_vs_num_proc.png",dpi=1200 )
plt.show()


plt.scatter(np_req,parallel_efficiency_data_real_mean_multi,label="Multi", marker='x')
plt.scatter(np_req,parallel_efficiency_data_real_mean_single,label="Single")
plt.xlabel("Number of processors")
plt.ylabel("% PE", rotation=0,labelpad=20)
plt.tight_layout()
plt.legend()
plt.savefig("/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/scaling_tests/flat_elastic/box_size_"+str(box_size)+"_parallel_eff_vs_num_proc.png",dpi=1200 )
plt.show()

plt.bar(np_req,max_number_of_timesteps_possible_multi,label="Multi")
plt.bar(np_req,max_number_of_timesteps_possible_single,label="Single")
plt.xlabel("Number of processors")
plt.ylabel("Max $N_{\Delta t}$", rotation=0,labelpad=20)
plt.tight_layout()
plt.legend()
plt.savefig("/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/scaling_tests/flat_elastic/box_size_"+str(box_size)+"_max_timesteps_vs_num_proc.png",dpi=1200 )
plt.show()
# %%
