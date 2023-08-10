# -*- coding: utf-8 -*-
"""
This file will read log.lammps files and output a numpy array containing each thermo column. 
"""
#%%
from math import log
import os
import mmap
import re 
import numpy as np


# realisation_name='log.H20_solid484692_inc_mom_output_no_rescale_323752_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_30_SN_1_rparticle_10.0'


def log2numpy_reader(realisation_name,Path_2_log,thermo_vars):

   #log_data= log2numpy(Path_2_log,thermo_vars,realisation_name)


   os.chdir(Path_2_log)
            
         
   # Searching for thermo output start and end line

   Thermo_data_start_line= ("   Step" + thermo_vars)
   Thermo_data_end_line =("Loop time of") 
   SRD_temp_lambda_start_line = "SRD temperature & lamda ="
   SRD_temp_lambda_end_line = "SRD max distance & max velocity"# re.compile(b"Loop time of [+-]?[0-9]+\.[0-9]+ on . procs for "+bytes(total_no_timesteps,'utf-8')) 

   #Loop time of 316.462 on 6 procs for 300000 steps with 27653 atoms

      
   warning_start = "WARNING: Fix srd particle moved outside valid domain"
   #warning_end = "(../fix_srd.cpp:2492)\\n"
   # convert string to bytes for faster searching 
   Thermo_data_start_line_bytes=bytes(Thermo_data_start_line,'utf-8')
   Thermo_data_end_line_bytes =bytes(Thermo_data_end_line,'utf-8')
   SRD_temp_lambda_end_bytes = bytes(SRD_temp_lambda_end_line,'utf-8')
   SRD_temp_lambda_start_bytes = bytes(SRD_temp_lambda_start_line,'utf-8')
         
         # find begining of data run 
   with open(realisation_name) as f:
      read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) 
      
      if read_data.find(Thermo_data_start_line_bytes) != -1:
         print('true')
         thermo_byte_start=read_data.find(Thermo_data_start_line_bytes)
         
      else:
         print('could not find thermo variables')
      
      
      # find end of run         
      
      if read_data.rfind(Thermo_data_end_line_bytes) != -1:
         print('true')
         thermo_byte_end =read_data.rfind(Thermo_data_end_line_bytes)      #Thermo_data_end_line_bytes.search(read_data)#read_data.find(Thermo_data_end_line_bytes)
      else:
         print('could not find end of run')
         # correct 
      if read_data.find(SRD_temp_lambda_start_bytes)!= -1:
         print('true')
         SRD_temp_lambda_start = read_data.find(SRD_temp_lambda_start_bytes)
      else: 
         print('Could not find SRD temperature')
      
      if read_data.find(SRD_temp_lambda_end_bytes)!= -1:
         print('true')
         SRD_temp_lambda_end = read_data.find(SRD_temp_lambda_end_bytes)
      else:     
            print("Could not find SRD temperature ending")
      
         
         
         
         
         
         
         # Splicing out the thermo data and closing the file
   with open(realisation_name) as f:
      read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ ) 

         
      read_data.seek(thermo_byte_start) 

      size_of_data =thermo_byte_end-thermo_byte_start
      log_array_bytes=read_data.read(thermo_byte_end)
      log_array_bytes_trim=log_array_bytes[0:size_of_data]
      read_data.seek(SRD_temp_lambda_start)
      SRD_temp_lambda_byte_array= read_data.read(SRD_temp_lambda_end)
      size_of_temp_lambda_data = SRD_temp_lambda_end-SRD_temp_lambda_start
      SRD_temp_lambda_byte_array_trim = SRD_temp_lambda_byte_array[0:size_of_temp_lambda_data]

      read_data.close()
            
      # comparison    
      #    log_string_before_warning_removal =  str(log_array_bytes_trim)
      # Convert byte array to string

   log_string= str(log_array_bytes_trim)
   SRD_temp_lambda_string = str(SRD_temp_lambda_byte_array_trim)

   #Count SRD warnings   
   warn_count  = log_string.count(warning_start) 
   #print('Number of Warnings:',warn_count)
   newline_count= log_string.count('\\n')

   empty=''
   warning_regex_pattern= re.compile(r'WARNING:\sFix\ssrd\sparticle\smoved\soutside\svalid\sdomain\\n\s\sparticle\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\son\sproc\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\sat\stimestep\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\\n\s\sxnew\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\\n\s\ssrdlo\/hi\sx\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\\n\s\ssrdlo\/hi\sy\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\\n\s\ssrdlo\/hi\sz\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\\n\s\(\.\.\/fix\_srd\.cpp:[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\)')
   final_warning_regex_pattern=re.compile(r'WARNING:\sToo\smany\swarnings:\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\svs\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\.\sAll\sfuture\swarnings\swill\sbe\ssuppressed\s\(\.\.\/thermo\.cpp:[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\)')
   #WARNING: Too many warnings: 101 vs 100. All future warnings will be suppressed (../thermo.cpp:465)
   newline_regex_pattern= re.compile(r'\\n')
   # matches a floating point number [+-]?([0-9]*[.])?[0-9]+
   warning_regex_pattern.search(log_string)
   # remove warnings
   count_warnings_with_regex=re.findall(warning_regex_pattern,log_string)
   
   log_string=re.sub(warning_regex_pattern,empty,log_string,count=warn_count)
   log_string=re.sub( final_warning_regex_pattern,empty,log_string)
   log_string=re.sub(newline_regex_pattern,empty,log_string)
   # log_array=log_string.split()
   warn_count_after  = log_string.count(warning_start) 
   #print('Number of Warnings after:',warn_count_after)
   #print(log_string)
   newline_count_after = log_string.count('\\n') 

   if warn_count_after==0:
      if newline_count_after==0:
       print('All standard warnings removed')
      else:
       print('Warning removal failed, reapplying regex.')
       log_string=re.sub(warning_regex_pattern,empty,log_string)
       log_string=re.sub(newline_regex_pattern,empty,log_string)
  
   
   log_string_array=log_string[:-1]
   step = "b'   Step"
   thermo_vars = step+thermo_vars
   thermo_vars_length = len(thermo_vars)
   log_string_array =log_string_array[thermo_vars_length:]

   log_string_array= log_string_array.split()
  # print(log_string_array)
   log_float_array = [0.00]*len(log_string_array)

   for i in range(len(log_string_array)):
      log_float_array[i] = float(log_string_array[i])


   log_file = np.array(log_float_array, dtype=np.float64)
   dim_log_file =np.shape(log_file)[0]
   col_log_file = len(Thermo_data_start_line.split())
   new_dim_log_file = int(np.divide(dim_log_file,col_log_file))
   log_file = np.reshape(log_file,(new_dim_log_file,col_log_file))
   
   return log_file

#%%
# Path_2_log='/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/T_1_phi_0.0005_solid_inc_data/H20_data/run_484692/'
# Path_2_log='/Volumes/Backup Plus 1/PhD_/Rouse Model simulations/Using LAMMPS imac/MYRIAD_LAMMPS_runs/T_1_phi_0.005_solid_inc_data/H20_data/run_208958'
# thermo_vars='         KinEng          Temp          TotEng    '
# count_log=30
# realisation_name_log=['log.H20_solid484692_inc_mom_output_no_rescale_145067_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_900_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_1527_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_600_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_158656_0.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_900_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_164541_1.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_15_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_168323_1.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_7_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_235058_0.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_1200_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_235669_0.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_3_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_282269_1.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_150_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_2828_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_300_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_323752_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_30_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_360133_0.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_60_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_396289_1.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_900_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_401357_1.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_3_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_427055_0.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_30_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_457422_1.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_60_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_513905_0.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_15_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_576909_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_15_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_601593_0.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_600_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_648931_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_60_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_702443_0.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_300_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_705546_1.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_600_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_725990_1.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_1200_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_728764_1.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_30_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_732115_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_1200_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_784890_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_3_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_789046_0.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_7_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_892601_0.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_150_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_921263_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_7_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_936130_2.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_150_SN_1_rparticle_10.0', 'log.H20_solid484692_inc_mom_output_no_rescale_941336_1.0_45733_255.88777235700172_0.001_3473_1000_10000_2500000_T_1.0_lbda_1.8151416549641417_SR_300_SN_1_rparticle_10.0']
# realisation_name_log=['log.H20_solid208958_inc_mom_output_no_rescale_128122_0.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_900_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_147887_0.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_1200_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_176283_2.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_3_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_185780_1.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_600_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_18828_0.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_600_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_201852_1.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_300_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_249979_1.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_60_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_287224_1.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_7_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_356862_0.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_3_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_358910_1.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_30_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_363679_1.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_3_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_369660_0.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_15_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_380975_2.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_30_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_388954_2.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_7_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_43376_2.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_600_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_434409_2.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_1200_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_438643_1.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_1200_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_441788_1.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_150_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_499804_1.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_900_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_656638_0.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_60_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_699201_0.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_150_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_707088_0.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_300_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_732945_0.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_7_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_739104_2.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_900_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_784611_2.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_150_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_788865_2.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_300_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_829140_2.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_15_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_911704_2.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_60_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_98160_0.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_30_SN_1_rparticle_10.0', 'log.H20_solid208958_inc_mom_output_no_rescale_988851_1.0_9112_118.77258268303078_0.001_1781_1000_10000_2000000_T_1.0_lbda_1.3166259218664098_SR_15_SN_1_rparticle_10.0']
# count=0
# for i in range(0,30):
#     print(realisation_name_log[i])
#     count=count+1
#     print(count)
#     log2numpy_reader(realisation_name_log[i],Path_2_log,thermo_vars)
#%%
