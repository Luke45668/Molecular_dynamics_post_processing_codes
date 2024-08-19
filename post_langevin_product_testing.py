##!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
This file processes the log files from brownian dynamics simulations 

after an MPCD simulation. 
"""
#%% Importing packages
import os
from random import sample
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import regex as re
import pandas as pd
import sigfig
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = (8,6 )
plt.rcParams.update({'font.size': 16})
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
import scipy.stats
from datetime import datetime
import mmap
import h5py as h5
from scipy.optimize import curve_fit

path_2_post_proc_module= '/Users/luke_dev/Documents/MPCD_post_processing_codes/'
os.chdir(path_2_post_proc_module)
import seaborn as sns
from log2numpy import *
from dump2numpy import *
import glob 
from post_MPCD_MP_processing_module import *
import pickle as pck
from post_langevin_module import *

linestyle_tuple = ['-', 
 '--', 
 '-.', ':', 
 'None', ' ', '', 'solid', 
 'dashed', 'dashdot', 'dotted']

#%% 


damp=np.array([ 0.035, 0.035 ])
K=np.array([ 40     , 50   ,
            ])

erate=np.flip(np.array([1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.175,0.15,0.125,0.1,0.08,
                0.06,0.04,0.01,0.005,0]))

no_timesteps=np.flip(np.array([ 3944000,  4382000,  4929000,  5634000,  6573000,  7887000,
         9859000, 13145000, 19718000,  2253000,  2629000,  3155000,
         3944000,  4929000,  6573000,  9859000, 39435000,
        78870000, 10000000]))

e_in=0
e_end=erate.size
n_plates=100

strain_total=100

path_2_log_files="/Users/luke_dev/Documents/MYRIAD_lammps_runs/langevin_runs/100_particle/run_667325/saved_tuples"



thermo_vars='         KinEng         PotEng         Press         c_myTemp        c_bias         TotEng    '
j_=5

eq_spring_length=3*np.sqrt(3)/2
mass_pol=5 
damp_ratio=mass_pol/damp


#%% save tuples
label='damp_'+str(damp)+'_K_'+str(K)+'_'

os.chdir(path_2_log_files)
#os.mkdir("tuple_results")
#os.chdir("tuple_results")

def batch_load_tuples(label,tuple_name):

    with open(label+tuple_name, 'rb') as f:
         load_in= pck.load(f)

    return load_in

erate_velocity_batch_tuple=()
spring_force_positon_tensor_batch_tuple=()
COM_velocity_batch_tuple=()
conform_tensor_batch_tuple=()
log_file_batch_tuple=()
area_vector_spherical_batch_tuple=()
interest_vectors_batch_tuple=()


# loading all data into one 
for i in range(K.size):
    label='damp_'+str(damp[i])+'_K_'+str(K[i])+'_'

    spring_force_positon_tensor_batch_tuple= spring_force_positon_tensor_batch_tuple+(batch_load_tuples(label,
                                                            "spring_force_positon_tensor_tuple.pickle"),)
    # erate_velocity_batch_tuple=erate_velocity_batch_tuple+(batch_load_tuples(label,
    #                                                         "erate_velocity_tuple.pickle"),)
    # COM_velocity_batch_tuple=COM_velocity_batch_tuple+(batch_load_tuples(label,
    #                                                         "COM_velocity_tuple.pickle"),)
    # conform_tensor_batch_tuple=conform_tensor_batch_tuple+(batch_load_tuples(label,
    #                                                         "conform_tensor_tuple.pickle"),)
    log_file_batch_tuple=log_file_batch_tuple+(batch_load_tuples(label,
                                                            "log_file_tuple.pickle"),)
    area_vector_spherical_batch_tuple=area_vector_spherical_batch_tuple+(batch_load_tuples(label,"area_vector_tuple.pickle"),)
    
    interest_vectors_batch_tuple=interest_vectors_batch_tuple+(batch_load_tuples(label,
                                                                                 "interest_vectors_tuple.pickle"),)

    


     

 #%% strain points for temperatuee data 
strainplot_tuple=()

for i in range(erate.size):
    
    strain_plotting_points= np.linspace(0,strain_total,log_file_batch_tuple[0][i].shape[0])

    strainplot_tuple=strainplot_tuple+(strain_plotting_points,)  
    print(strainplot_tuple[i].size)

def strain_plotting_points(total_strain,points_per_iv):
     #points_per_iv= number of points for the variable measured against strain 
     strain_unit=total_strain/points_per_iv
     strain_plotting_points=np.arange(0,total_strain,strain_unit)
     return  strain_plotting_points



folder="temperature_plots"
folder_check_or_create(path_2_log_files,folder)
column=5
final_temp=np.zeros((erate.size))
mean_temp_array=np.zeros((K.size,erate.size))
for j in range(K.size):
    for i in range(erate.size):
        
        # plt.plot(strainplot_tuple[i][:],log_file_batch_tuple[j][i][:,column])
        # final_temp[i]=log_file_batch_tuple[j][i][-1,column]
        
        mean_temp_array[j,i]=np.mean(log_file_batch_tuple[j][i][1000:,column])
      
        #plt.axhline(np.mean(log_file_batch_tuple[j][i][:,column]))
    #     plt.ylabel("$T$", rotation=0)
    #     plt.xlabel("$\gamma$")
    

    # #   plt.savefig("temp_vs_strain_damp_"+str(damp)+"_gdot_"+str(erate[i])+"_.pdf",dpi=1200,bbox_inches='tight')
    #     plt.show()

#

marker=['x','o','+','^',"1","X","d","*","P","v"]
for j in range(K.size):
    plt.scatter(erate,mean_temp_array[j,:],label="$K="+str(K[j])+"$" ,marker=marker[j])
    plt.ylabel("$T$", rotation=0)
    plt.xlabel('$\dot{\gamma}$')
    plt.xscale('log')
   # plt.yscale('log')
plt.axhline(1,label="$T_{0}=1$")
plt.legend()
plt.tight_layout()
#plt.savefig(path_2_log_files+"/plots/temp_vs_erate.pdf",dpi=1200,bbox_inches='tight')


plt.show()


#%% look at internal stresses
sns.set_palette('colorblind')
plt.rcParams["figure.figsize"] = (5.5,4 )
plt.rcParams.update({'font.size': 14})
SIZE_DEFAULT = 14
SIZE_LARGE = 16
#plt.rcParams['text.usetex'] = True
# plt.rc("font", family="Roboto")  # controls default font
# plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

legfont=12
folder="stress_tensor_plots"
marker=['x','+','^',"1","X","d","*","P","v"]
aftcut=1
cut=0.3
folder_check_or_create(path_2_log_files,folder)
labels_stress=np.array(["$\sigma_{xx}$",
               "$\sigma_{yy}$",
               "$\sigma_{zz}$",
               "$\sigma_{xz}$",
               "$\sigma_{xy}$",
               "$\sigma_{yz}$"])



#compute stress tensor 
y_ticks_stress=[-10,0,20,40,60,80]
stress_tensor=np.zeros((K.size,e_end,6))
stress_tensor_std=np.zeros((K.size,e_end,6))
n_1=np.zeros((K.size,e_end))
n_1_error=np.zeros((K.size,e_end))
n_2=np.zeros((K.size,e_end))
n_2_error=np.zeros((K.size,e_end))
for j in range(K.size):
    stress_tensor[j],stress_tensor_std[j]= stress_tensor_averaging(e_end,
                            labels_stress,
                            cut,
                            aftcut,
                           spring_force_positon_tensor_batch_tuple[j],j_)
    n_1[j],n_1_error[j]=compute_n_stress_diff(stress_tensor[j], 
                          stress_tensor_std[j],
                          0,2,
                          j_,n_plates,
                         )

    n_2[j],n_2_error[j]=compute_n_stress_diff(stress_tensor[j], 
                          stress_tensor_std[j],
                          2,1,
                          j_,n_plates,
                          )
  
for j in range(K.size):    
    for i in range(6):

        plotting_stress_vs_strain( spring_force_positon_tensor_batch_tuple[j],
                                e_in,e_end,j_,
                                strain_total,cut,aftcut,i,labels_stress[i],erate)
    plt.legend(fontsize=legfont) 
    plt.tight_layout()
    plt.savefig(path_2_log_files+"/plots/"+str(K[j])+"_SS_grad_plots.pdf",dpi=1200,bbox_inches='tight')       
   
    plt.show()



for j in range(K.size): 
    plot_stress_tensor(0,3,
                       stress_tensor[j],
                       stress_tensor_std[j],
                       j_,n_plates, labels_stress,marker,0,erate,e_end,linestyle_tuple[j])
    plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
plt.legend(fontsize=legfont) 
plt.yticks(y_ticks_stress)
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/stress_tensor_0_3_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.show()

for j in range(K.size): 
    plot_stress_tensor(3,6,
                       stress_tensor[j],
                       stress_tensor_std[j],
                       j_,n_plates, labels_stress,marker,0,erate,e_end,linestyle_tuple[j])
    plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
plt.legend(fontsize=legfont) 
plt.yticks(y_ticks_stress)
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/_stress_tensor_3_6_plots.pdf",dpi=1200,bbox_inches='tight') 
plt.show()

# collapse plot
for j in range(K.size): 
    plot_stress_tensor(0,3,
                       stress_tensor[j]/K[j],
                       stress_tensor_std[j]/K[j],
                       j_,n_plates, labels_stress,marker,0,erate,e_end,linestyle_tuple[j])
    plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
plt.ylabel("$\sigma_{\\alpha\\beta}/K$",rotation=0,labelpad=25)
plt.legend(fontsize=legfont) 
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/stress_tensor_K_scaled_0_3_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()

for j in range(K.size): 
    plot_stress_tensor(3,6,
                       stress_tensor[j]/K[j],
                       stress_tensor_std[j]/K[j],
                       j_,n_plates, labels_stress,marker,0,erate,e_end,linestyle_tuple[j])
    plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
plt.ylabel("$\sigma_{\\alpha\\beta}/K$",rotation=0,labelpad=25)
plt.legend(fontsize=legfont) 
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/stress_tensor_K_scaled_3_6_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()



# now plot n1 and n2 
#probably need to turn this into a a function 
n_y_ticks=[-10,0,20,40,60,80]
cutoff=0
for j in range(K.size):
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
  

    plt.errorbar(erate[cutoff:e_end], n_1[j,cutoff:e_end], yerr =n_1_error[j,cutoff:e_end],
                  ls='none',label="$N_{1},K="+str(K[j])+"$",marker=marker[j] )
    popt,cov_matrix_n1=curve_fit(linearthru0,erate[cutoff:e_end], n_1[j,cutoff:e_end])
    difference=np.sqrt(np.sum((n_1[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

    plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),ls=linestyle_tuple[j])#,
            # label="$N_{1,fit,K="+str(K[j])+"},m="+str(sigfig.round(popt[0],sigfigs=2))+\
            #     ",\\varepsilon="+str(sigfig.round(difference,sigfigs=2))+"$", ls=linestyle_tuple[1])

    #plt.xscale('log')
    #plt.show()
    print(difference)

    plt.errorbar(erate[cutoff:e_end], n_2[j,cutoff:e_end], yerr =n_2_error[j,cutoff:e_end],
                  ls='none',label="$N_{2},K="+str(K[j])+"$",marker=marker[j] )
    popt,cov_matrix_n2=curve_fit(linearthru0,erate[cutoff:e_end], n_2[j,cutoff:e_end])
    difference=np.sqrt(np.sum((n_2[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

    plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),ls=linestyle_tuple[j])#,
    #         label="$N_{2,fit,K="+str(K[j])+"},m="+str(sigfig.round(popt[0],sigfigs=2))+\
    #         ",\\varepsilon="+str(sigfig.round(difference,sigfigs=2))+"$",ls=linestyle_tuple[2])
    # #
plt.legend(fontsize=legfont)
#plt.xscale('log')
plt.xlabel("$\dot{\gamma}$")
plt.ylabel("$N_{\\alpha}$",rotation=0)
plt.yticks(n_y_ticks)
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/N1_N2_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
print(difference)

#collapse N1 and N2 /K
cutoff=0
for j in range(K.size):
    #plt.plot(0,0,marker='none',ls=linestyle_tuple[j],color='grey',label="$K="+str(K[j])+"$")
  

    plt.errorbar(erate[cutoff:e_end], n_1[j,cutoff:e_end]/K[j], yerr =n_1_error[j,cutoff:e_end]/K[j],
                  ls='none',label="$N_{1},K="+str(K[j])+"$",marker=marker[j] )
    # popt,cov_matrix_n1=curve_fit(linearthru0,erate[cutoff:e_end], n_1[j,cutoff:e_end])
    # difference=np.sqrt(np.sum((n_1[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

    # plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),
    #         label="$N_{1,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
    #             ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$", ls=linestyle_tuple[j])

    #plt.xscale('log')
    #plt.show()
    print(difference)

    plt.errorbar(erate[cutoff:e_end], n_2[j,cutoff:e_end]/K[j], yerr =n_2_error[j,cutoff:e_end]/K[j],
                  ls='none',label="$N_{2},K="+str(K[j])+"$",marker=marker[j] )
    # popt,cov_matrix_n2=curve_fit(linearthru0,erate[cutoff:e_end], n_2[j,cutoff:e_end])
    # difference=np.sqrt(np.sum((n_2[j,cutoff:e_end]-(popt[0]*(erate[cutoff:e_end])))**2)/(e_end))

    # plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end])),
    #         label="$N_{2,fit},m="+str(sigfig.round(popt[0],sigfigs=3))+\
    #         ",\\varepsilon="+str(sigfig.round(difference,sigfigs=3))+"$",ls=linestyle_tuple[j])
    #
    plt.legend(fontsize=legfont)
    plt.ylabel("$N_{\\alpha}/K$",rotation=0)
    plt.xscale('log')
    plt.xlabel("$\dot{\gamma}$")
plt.tight_layout()
plt.savefig(path_2_log_files+"/plots/N1_N2_scaled_K_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()
print(difference)





# now do N1 N2 comparison plots

n2_factor=[-13,-16]
for j in range(K.size):

    #sns.set_palette('icefire')
    plt.scatter(erate,n_1[j],label="$N_{1}$", marker=marker[j])
    plt.scatter(erate,n2_factor[j]*n_2[j],label="$"+str(n2_factor[j])+"N_{2}$", marker=marker[j+2])
    plt.xlabel("$\dot{\gamma}$")
    plt.ylabel("$N_{\\alpha}$",rotation=0)
    #plt.ylabel("$\\frac{N_{1}}{N_{2}}$", rotation=0)
    plt.legend()
    #plt.xscale('log')
    plt.legend(fontsize=legfont) 
plt.yticks(n_y_ticks)
plt.tight_layout()

plt.savefig(path_2_log_files+"/plots/N1_N2_multi_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show()



cutoff=1
for j in range(K.size):
    xz_stress= stress_tensor[j,cutoff:,3]
    xz_stress_std=stress_tensor_std[j,:,3]/np.sqrt(j_*n_plates)
    #powerlaw
    plt.errorbar(erate[cutoff:e_end], xz_stress/erate[cutoff:e_end], yerr =xz_stress_std[cutoff:],
                  ls='none',label="$\eta,K="+str(K[j])+"$",marker=marker[j] )
    popt,cov_matrix_xz=curve_fit(powerlaw,erate[cutoff:e_end], xz_stress/erate[cutoff:e_end])
    y=xz_stress/erate[cutoff:e_end]
    y_pred=popt[0]*(erate[cutoff:e_end]**(popt[1]))
    difference=np.sqrt(np.sum((y-y_pred)**2)/e_end-cutoff)
    plt.plot(erate[cutoff:e_end],(popt[0]*(erate[cutoff:e_end]**(popt[1]))),
            label="$\eta_{fit},a="+str(sigfig.round(popt[0],sigfigs=3))+",n="+str(sigfig.round(popt[1],sigfigs=3))+

            ",\\varepsilon=\pm"+str(sigfig.round(difference,sigfigs=3))+"$")

    plt.legend(fontsize=legfont) 
    plt.ylabel("$\eta$", rotation=0,labelpad=10)
    plt.xlabel("$\dot{\gamma}$")
plt.tight_layout()
    # plt.xscale('log')
    # plt.yscale('log')
plt.savefig(path_2_log_files+"/plots/eta_vs_gdot_plots.pdf",dpi=1200,bbox_inches='tight')
plt.show() 

#%% area vector plots 
sns.set_palette('colorblind')
plt.rcParams["figure.figsize"] = (6,4 )
plt.rcParams.update({'font.size': 14})
SIZE_DEFAULT = 14
SIZE_LARGE = 16
legfont=9
#plt.rcParams['text.usetex'] = True
# plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels

pi_theta_ticks=[ -np.pi, -np.pi/2, 0, np.pi/2,np.pi]
pi_theta_tick_labels=['-π','-π/2','0', 'π/2', 'π'] 
phi_y_ticks=[0,0.2,0.4,0.6,0.8,1.0,1.2]
pi_phi_ticks=[ 0,np.pi/4, np.pi/2]
pi_phi_tick_labels=[ '0','π/4', 'π/2']
theta_y_ticks=[0,0.02,0.04,0.06,0.08,0.1]
skip_array=np.arange(0,e_end,3)
spherical_coords_tuple=()
# fig = plt.figure(constrained_layout=True)
# spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
for j in range(K.size):
    spherical_coords_tuple=()
    for i in range(e_in,e_end):
        
        area_vector_ray=area_vector_spherical_batch_tuple[j][i]
        # detect all z coords less than 0 and multiply all 3 coords by -1
        area_vector_ray[area_vector_ray[:,:,:,2]<0]*=-1
        spherical_coords_array=np.zeros((j_,area_vector_ray.shape[1],n_plates,3))
        x=area_vector_ray[:,:,:,0]
        y=area_vector_ray[:,:,:,1]
        z=area_vector_ray[:,:,:,2]


        # radial coord
        spherical_coords_array[:,:,:,0]=np.sqrt((x**2)+(y**2)+(z**2))
        #  theta coord 
        spherical_coords_array[:,:,:,1]=np.sign(y)*np.arccos(x/(np.sqrt((x**2)+(y**2))))
        # phi coord
        spherical_coords_array[:,:,:,2]=np.arccos(z/spherical_coords_array[:,:,:,0])

        spherical_coords_tuple=spherical_coords_tuple+(spherical_coords_array,)

   
    for l in range(skip_array.size):
    #for j in range(j_):


        l=skip_array[l]
        
        
        # sns.displot( data=np.ravel(spherical_coords_tuple[i][:,200000,:,1]),
        #             label ="$\dot{\gamma}="+str(erate[i])+"$", kde=True)
        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j],:,1]),
        #             label="output_range:"+str(skip_array_2[j]))
        data=np.ravel( spherical_coords_tuple[l][:,-500:,:,1])
        periodic_data=np.array([data-2*np.pi,data,data+2*np.pi])  

        sns.kdeplot( data=np.ravel(periodic_data),
                    label ="$\dot{\gamma}="+str(erate[l],)+"$")#bw_adjust=0.1
        
        # mean_data=np.mean(spherical_coords_tuple[0][:,-1,:,1],axis=0)      
        #plt.hist(np.ravel(spherical_coords_tuple[i][:,-100,:,1]))
        # bw adjust effects the degree of smoothing , <1 smoothes less
    plt.xlabel("$\Theta$")
    plt.xticks(pi_theta_ticks,pi_theta_tick_labels)
    plt.yticks(theta_y_ticks)
    plt.xlim(-np.pi,np.pi)
    plt.ylabel('Density')
    plt.legend(fontsize=legfont) 
    plt.savefig(path_2_log_files+"/plots/theta_dist_K_"+str(K[j])+"_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()

   
    for l in range(skip_array.size):
    #for j in range(skip_array_2.size):
        l=skip_array[l]
        

        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,skip_array_2[j],:,2]),
        #              label="output_range:"+str(skip_array_2[j]))
        # sns.kdeplot( data=np.ravel(spherical_coords_tuple[i][:,-1,:,2]),
        #              label ="$\dot{\gamma}="+str(erate[i])+"$")
        data=np.ravel(spherical_coords_tuple[l][:,-500:,:,2])
        periodic_data=np.array([data,np.pi+data])  
        sns.kdeplot( data=np.ravel(periodic_data),
                      label ="$\dot{\gamma}="+str(erate[l])+"$")
                   
        #plt.hist(np.ravel(spherical_coords_tuple[i][:,-1,:,2]))

    plt.xlabel("$\Phi$")
    plt.xticks(pi_phi_ticks,pi_phi_tick_labels)
    plt.yticks(phi_y_ticks)
    plt.ylabel('Density')
    plt.legend(fontsize=legfont,loc='upper right') 
    plt.xlim(0,np.pi/2)
    plt.savefig(path_2_log_files+"/plots/phi_dist_K_"+str(K[j])+"_.pdf",dpi=1200,bbox_inches='tight')
    plt.show()


    

  


extension_ticks=[0,1,2,3,4]
for j in range(K.size):
    for i in range(skip_array.size):
        i=skip_array[i]
    # for i in range(e_in,e_end):

        # sns.kdeplot(eq_spring_length-np.ravel(interest_vectors_tuple[i][:,:,2:5]),
        #              label ="$K="+str(K)+"$")
                    #label ="$\dot{\gamma}="+str(erate[i])+"$")
        sns.kdeplot(eq_spring_length+0.125-np.ravel(interest_vectors_batch_tuple[j][i][:,:,2:5]),
                    label ="$\dot{\gamma}="+str(erate[i])+"$")
        
    plt.xlabel("$\Delta x$")

    plt.ylabel('Density')
    plt.legend(fontsize=legfont,loc='upper right') 
    plt.yticks(extension_ticks)

    plt.legend(fontsize=11)
    plt.savefig(path_2_log_files+"/plots/deltax_dist_K_"+str(K[j])+"_.pdf",dpi=1200,bbox_inches='tight')
    #plt.xlim(-3,2)
    plt.show()

# %%
