#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 25 2023

This script will contain all the petersen graph plotting functions 

@author: lukedebono
"""

import os
import numpy as np

import matplotlib.pyplot as plt
import regex as re
import pandas as pd
import sigfig
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
from mpl_toolkits import mplot3d
from matplotlib.gridspec import GridSpec
#import seaborn as sns
import math as m
import scipy.stats
from datetime import datetime

#%%
def sc_vs_mfp_to_collision_cell_ratio(length_multiplier,number_of_lengthscales,fluid_name,number_of_test_points,mean_free_path_to_box_ratio_neg,sc_neg_soln,mean_free_path_to_box_ratio_pos,sc_pos_soln,Solvent_bead_SRD_box_density_cp_1):
    fig=plt.figure(figsize=(18,7))
    gs=GridSpec(nrows=1,ncols=2)
    ax1= fig.add_subplot(gs[0,0],projection='3d') 
    ax2= fig.add_subplot(gs[0,1],projection='3d') 

    fig.suptitle(fluid_name+': $Sc\ vs\ \\frac{\lambda}{\Delta x}$',size='large', wrap=True)

    #ax1= fig.add_subplot(gs[0]) 
    for k in range(0,number_of_lengthscales):
        for z in range(0,number_of_test_points):
            
            ax1.plot(mean_free_path_to_box_ratio_neg[k][z,:],length_multiplier[k,:],sc_neg_soln[k][z,:],marker='x',label='M+={}'.format(Solvent_bead_SRD_box_density_cp_1[0,z]))
            
            
            ax2.plot(mean_free_path_to_box_ratio_pos[k][z,:],length_multiplier[k,:],sc_neg_soln[k][z,:],marker='o')
            
           # ax1.set_xscale('log')
           # ax1.set_yscale('log')
            ax1.set_xlabel('$\\frac{\lambda}{\Delta x}\  [-]$', rotation='horizontal',ha='right',size='large')
            ax1.set_ylabel('$\ell\ [m]$')
            ax1.set_zlabel( '$Sc\ [-]$', rotation='horizontal',ha='right',size='large')
            ax1.grid('on')
            ax2.set_xlabel('$\\frac{\lambda}{\Delta x}\  [-]$', rotation='horizontal',ha='right',size='large')
            ax2.set_ylabel('$\ell\ [m]$')
            ax2.set_zlabel( '$Sc\ [-]$', rotation='horizontal',ha='right',size='large')
            ax2.grid('on')
            
    plt.show()
    
def sc_vs_collision_cell_to_lengthscale(fluid_name,number_of_test_points,box_size_to_lengthscale,sc_neg_soln,sc_pos_soln,Solvent_bead_SRD_box_density_cp_1):
    # Sc vs box size/lengthscale
    fig=plt.figure(figsize=(10,6))
    gs=GridSpec(nrows=1,ncols=1)

    fig.suptitle(fluid_name+': $Sc\ vs$ $\\frac{\Delta x}{\\bar{\ell}}\\ $',size='large', wrap=True)

    ax1= fig.add_subplot(gs[0]) 
    for z in range(0,number_of_test_points):
        
        ax1.plot(box_size_to_lengthscale[0,:],sc_neg_soln[z,:],label='M+={}'.format(Solvent_bead_SRD_box_density_cp_1[0,z]))
        
        #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
        ax1.plot(box_size_to_lengthscale[0,:],sc_pos_soln[z,:])
        
        ax1.set_xscale('linear')
        ax1.set_yscale('log')
        ax1.set_xlabel('$\\frac{\Delta x}{\\bar{\ell}}\  [-]$', rotation='horizontal',ha='right',size='large')
        ax1.set_ylabel( '$Sc\ [-]$', rotation='horizontal',ha='right',size='large')
        ax1.grid('on')
        
    plt.show()

def sc_vs_mfp_to_collision_cell(mean_free_path_to_box_ratio_pos,mean_free_path_to_box_ratio_neg,fluid_name,number_of_test_points,sc_neg_soln,sc_pos_soln,Solvent_bead_SRD_box_density_cp_1):
    # Sc vs box size/lengthscale
    fig=plt.figure(figsize=(15,6))
    gs=GridSpec(nrows=1,ncols=1)
    fontsize=20

    #fig.suptitle(fluid_name+': $Sc\ vs$ $\\frac{\Delta x}{\\bar{\ell}}\\ $',size='large', wrap=True)

    ax1= fig.add_subplot(gs[0]) 
    for z in range(0,number_of_test_points):
        
        ax1.plot(mean_free_path_to_box_ratio_neg[z,:],sc_neg_soln[z,:],label='$M$={}'.format(sigfig.round(Solvent_bead_SRD_box_density_cp_1[0,z]),sigfigs=2),marker='x')
        
        #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
       # ax1.plot(mean_free_path_to_box_ratio_pos[z,:],sc_pos_soln[z,:],label='M+={}'.format(Solvent_bead_SRD_box_density_cp_1[0,z]))
        
        
        ax1.set_xscale('linear')
        ax1.set_yscale('log')
        ax1.set_xlabel('$Kn\  [-]$', rotation='horizontal',ha='right',fontsize=fontsize)
        ax1.set_ylabel( '$Sc\ [-]$', rotation='horizontal',ha='right',fontsize=fontsize)
        ax1.grid('on')
        ax1.legend(loc='right',bbox_to_anchor=(1.075, 0.5))
        
    plt.show()
        
def mfp_to_collision_cell_vs_collision_cell_to_lengthscale(fluid_name,number_of_test_points,box_size_to_lengthscale,mean_free_path_to_box_ratio_neg):
    #box size to length_scale  vs lammda to box size
    fig=plt.figure(figsize=(10,6))
    gs=GridSpec(nrows=1,ncols=1)

    fig.suptitle(fluid_name+': $\\frac{\lambda}{\Delta x}\ $ vs $\\frac{\Delta x}{\\bar{\\ell}}\ $',size='large', wrap=True)

    ax1= fig.add_subplot(gs[0]) 
    for z in range(0,number_of_test_points):
        
        ax1.plot(box_size_to_lengthscale[0,:],mean_free_path_to_box_ratio_neg[z,:])#label='M+={}'.format(Solvent_bead_SRD_box_density_cp_1[0,z]))
        
        #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
        #ax1.plot(box_size_to_lengthscale[0,:],mean_free_path_to_box_ratio_pos[z,:])
        
        ax1.set_xscale('linear')
        ax1.set_yscale('linear')
        ax1.set_ylabel('$\\frac{\lambda}{\Delta x}\ [-]$', rotation='horizontal',ha='right',size='large')
        #ax1.set_ylabel('$\\frac{\lambda{\Delta x}\ [-]$')
        ax1.set_xlabel('$\\frac{\Delta x}{\\bar{\ell}}\ [-]$', rotation='horizontal',ha='right',size='large')
        ax1.grid('on')
        
    plt.show()


    
def SRD_timestep_vs_collsion_cell(fluid_name,number_of_test_points,box_size_to_lengthscale,Solvent_bead_SRD_box_density_cp_1,SRD_step_pos_nd,SRD_step_neg_nd):
        fig=plt.figure(figsize=(10,6))
        gs=GridSpec(nrows=1,ncols=1)

        fig.suptitle(fluid_name+': $\\frac{\Delta t_{SRD}}{\\bar{\\tau}}\ $ vs $\\frac{\Delta x}{\\bar{\ell}}\ $',size='large', wrap=True)

        ax1= fig.add_subplot(gs[0]) 
        
        for z in range(0,number_of_test_points):
            
            ax1.plot(box_size_to_lengthscale[0,:],SRD_step_pos_nd[z,:],label='M+={}'.format(Solvent_bead_SRD_box_density_cp_1[0,z]))
            
            #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
            ax1.plot(box_size_to_lengthscale[0,:],SRD_step_neg_nd[z,:])
            
            ax1.set_xscale('linear')
            ax1.set_yscale('log')
            ax1.set_ylabel('$\\frac{\Delta t_{SRD}}{\\bar{\\tau}}\ [-] $', rotation='horizontal',ha='right',size='large')
            ax1.set_xlabel( '$\\frac{\Delta x}{\\bar{\ell}}\ [-]$', rotation='horizontal',ha='right',size='large')
            ax1.grid('on')
            
        plt.show()
        
        
        
        
def mfp_to_collsion_cell_vs_SRD_MD_ratio_vs_Sc(fluid_name,number_of_test_points,mean_free_path_to_box_ratio_neg,mean_free_path_to_box_ratio_pos,SRD_MD_ratio_neg,SRD_MD_ratio_pos,sc_neg_soln,sc_pos_soln):
    
        fig=plt.figure(figsize=(18,7)) #width x height
        gs=GridSpec(nrows=1,ncols=2)

        fig.suptitle(fluid_name+':  $\\frac{\lambda}{\Delta x}\ $ vs $\\frac{\Delta t_{SRD}}{\Delta t_{MD}}$ vs  $Sc$',size='x-large', wrap=True)
        #$\\frac{\lambda}{\Delta x}\ $
        ax1= fig.add_subplot(gs[0,0],projection='3d') 
        ax2= fig.add_subplot(gs[0,1],projection='3d') 
        for z in range(0,number_of_test_points):
            
            ax1.plot(mean_free_path_to_box_ratio_neg[z,:],SRD_MD_ratio_neg[z,:],sc_neg_soln[z,:],marker ='x')
            ax2.plot(mean_free_path_to_box_ratio_pos[z,:],SRD_MD_ratio_pos[z,:],sc_pos_soln[z,:],marker ='o')
            
            #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
        # ax1.plot(mean_free_path_to_box_ratio_pos[z,:],SRD_MD_ratio_pos[z,:],sc_pos_soln[z,:])
            
            #ax1.set_xscale('log')
            # ax1.set_yscale('log')
            # ax1.set_zscale('log')
            ax1.set_xlabel('$\\frac{\lambda}{\Delta x}\ $', rotation='horizontal',ha='right',size='large')
            ax1.set_ylabel( '$\\frac{\Delta t_{SRD}}{\Delta t_{MD}}$', rotation='vertical',ha='right',size='large')
            ax1.set_zlabel( '$Sc$', rotation='horizontal',ha='right',size='large')
            #ax2.grid('on')
            ax2.set_xlabel('$\\frac{\lambda}{\Delta x}\ $', rotation='horizontal',ha='right',size='large')
            ax2.set_ylabel( '$\\frac{\Delta t_{SRD}}{\Delta t_{MD}}$', rotation='vertical',ha='right',size='large')
            ax2.set_zlabel( '$Sc$', rotation='horizontal',ha='right',size='large')
            #ax2.grid('on')
            
            
        plt.show()

def SRD_MD_ratio_vs_ell(lengthscale_parameter,fluid_name,number_of_test_points,SRD_MD_ratio_neg,SRD_MD_ratio_pos):
    
        fig=plt.figure(figsize=(18,7)) #width x height
        gs=GridSpec(nrows=1,ncols=2)

        fig.suptitle(fluid_name+':  $\\frac{\Delta t_{SRD}}{\Delta t_{MD}}$ vs ${\ell}\$ vs  $Sc$',size='x-large', wrap=True)
        #$\\frac{\lambda}{\Delta x}\ $
        ax1= fig.add_subplot(gs[0,0]) 
        #ax2= fig.add_subplot(gs[0,1]) 
        for z in range(0,number_of_test_points):
            
            ax1.plot(lengthscale_parameter[z,:],SRD_MD_ratio_neg[z,:],marker ='x')
            #ax2.plot(lengthscale_parameter[z,:],SRD_MD_ratio_pos[z,:],marker ='o')
            
            #ax1.legend(Solvent_bead_SRD_box_density_cp_1[0,z])
        # ax1.plot(mean_free_path_to_box_ratio_pos[z,:],SRD_MD_ratio_pos[z,:],sc_pos_soln[z,:])
            
            #ax1.set_xscale('log')
            # ax1.set_yscale('log')
            # ax1.set_zscale('log')
            ax1.set_xlabel('${\ell}\ [] $', rotation='horizontal',ha='right',size='large')
            ax1.set_ylabel( '$\\frac{\Delta t_{SRD}}{\Delta t_{MD}}$', rotation='vertical',ha='right',size='large')
            
            #ax2.grid('on')
            #ax2.set_xlabel('${\ell}\ [] $', rotation='horizontal',ha='right',size='large')
            #ax2.set_ylabel( '$\\frac{\Delta t_{SRD}}{\Delta t_{MD}}$', rotation='vertical',ha='right',size='large')
            
            #ax2.grid('on')
            
            
        plt.show()

# %%


# %%
