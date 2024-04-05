# -*- coding: utf-8 -*-
"""
This file will read log.lammps files and output a run time in seconds 
"""
#%%
from math import log
import os
import mmap
import re 
import numpy as np


# realisation_name='log.H20_solid484692_inc_mom_output_no_rescale_323752_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_30_SN_1_rparticle_10.0'


def run_time_reader(realisation_name):
   format_string = "%H:%M:%S"
   run_data_start= ("Total wall time: ")
   run_data_start_bytes=bytes(run_data_start,'utf-8')

   with open(realisation_name) as f:
      read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ ) 

      run_data_seek=read_data.find(run_data_start_bytes)  
      read_data.seek(run_data_seek) 

      
      run_time_bytes=read_data.read()
      
      read_data.close()
   run_time=str(run_time_bytes)[18:26].replace(" ", "")
   epoch_time = datetime(1900, 1, 1)
   run_time=datetime.strptime( run_time, format_string)
   delta = (run_time - epoch_time)
   run_time=int(delta.total_seconds())
   
   return run_time

# %%
