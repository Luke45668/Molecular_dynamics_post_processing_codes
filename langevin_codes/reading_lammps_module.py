# -*- coding: utf-8 -*-
"""
This module contains all the codes I've produced to read lammps output files 

"""

import os
import mmap
import re 
import numpy as np
import pandas as pd


# this function reads a dumpfile by reading the whole thing into a text file, its 
# definitely over complicated.
def dump2numpy_f(dump_start_line,Path_2_dump,dump_realisation_name,number_of_particles_per_dump):
       
       
        dump_start_line_bytes = bytes(dump_start_line,'utf-8')

        os.chdir(Path_2_dump) #+simulation_file+"/" +filename


        with open(dump_realisation_name) as f:
                    read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) 
                    if read_data.find(dump_start_line_bytes) != -1:
                        #print('Start of run found')
                        dump_byte_start=read_data.find(dump_start_line_bytes)
                        end_pattern = re.compile(b"\d\n$")
                        #end_pattern = re.compile(b"^\s$")
                        end_position_search = end_pattern.search(read_data)  
                    
                
                    else:
                        print('could not find dump variables')
            

        # find end of run        
        
                    if end_pattern.search(read_data) != -1:
                        #print('End of run found')
                        #print(end_position_search)
                        dump_byte_end= end_position_search.span(0)[1]
                    else:
                        print('could not find end of run')
            
   
        read_data.seek(dump_byte_start) #setting relative psotion of the file 
        size_of_data =dump_byte_end-dump_byte_start
        dump_array_bytes=read_data.read(dump_byte_end)

        #finding all the matches and putting their indicies into lists 

        timestep_start_pattern = re.compile(dump_start_line_bytes) #b"ITEM: ATOMS id type x y z vx vy vz mass"
        timestep_end_pattern = re.compile(b"ITEM: TIMESTEP")

        dumpstart = timestep_start_pattern.finditer(dump_array_bytes)
        dumpend = timestep_end_pattern.finditer(dump_array_bytes)
        count=0
        dump_start_timestep=[]
        dump_end_timestep=[]


        for matches in dumpstart:
                count=count+1 # this is counted n times as we include the first occurence 
                #print(matches)
                dump_start_timestep.append(matches.span(0)[1])
                

        count1=0
        for match in dumpend:
                count1=count1+1 # this is counted n-1 times as we dont include the first occurence
                #print(match)
                dump_end_timestep.append(match.span(0)[0])


        #Splicing and storing the dumps into a list containing the dump at each timestep  
        dump_one_timestep=[]    
       
            
        for i in range(0,count-1):
                dump_one_timestep.append(dump_array_bytes[dump_start_timestep[i]:dump_end_timestep[i]])
            
        # getting last dump not marked by ITEM: TIMESTEP      
        dump_one_timestep.append(dump_array_bytes[dump_start_timestep[count-1]:])
        #dump_one_timestep_tuple=dump_one_timestep_tuple+(str(dump_array_bytes[dump_start_timestep[count-1]:]),)
        dump_one_timestep=str(dump_one_timestep)

        newline_regex_pattern= re.compile(r'\\n')
        remove_b_regex_pattern= re.compile(r'b\'')
        remove_left_bracket_regex_pattern=re.compile(r'\[')
        remove_right_bracket_regex_pattern=re.compile(r'\]')

        remove_stray_comma_regex_pattern=re.compile(r'\,')
        remove_stray_appost_regex_pattern = re.compile(r'\'')
        empty=' '


        dump_one_timestep=re.sub(newline_regex_pattern,empty,dump_one_timestep)

        dump_one_timestep=re.sub(remove_b_regex_pattern,empty,dump_one_timestep)
        dump_one_timestep=re.sub(remove_left_bracket_regex_pattern,empty,dump_one_timestep)
        dump_one_timestep=re.sub(remove_right_bracket_regex_pattern,empty,dump_one_timestep)
        dump_one_timestep=re.sub(remove_stray_comma_regex_pattern,empty,dump_one_timestep)
        dump_one_timestep=re.sub(remove_stray_appost_regex_pattern ,empty,dump_one_timestep)
            # needs to remove all the line markers and the b'
            
        dump_one_timestep=dump_one_timestep.split()
        for i in range(0,count): 
           # print(i)
            dump_one_timestep[i]=float(dump_one_timestep[i])
            
        
        #print(len(dump_one_timestep))
        # particle_sorted_array_1=[]  #np.zeros((1,len(dump_start_line.split()) -2)) #*number_of_particles_per_dump))    
        # particle_sorted_array_2=[] 
        # total_number_of_readouts_for_run=count
        #print(total_number_of_readouts_for_run)
        
        number_cols_dump = (len(dump_start_line.split()) -2) #*number_of_particles_per_dump  # minus one to remove item: 
        number_rows_dump= count *number_of_particles_per_dump 

        dump_file_unsorted=np.reshape(np.array(dump_one_timestep),(number_rows_dump,number_cols_dump))
        #particle_sorted_array=np.array([[]])
        # need to extend this loop for more particles and generalise it if you want N particles
        # j=0
        # # particle 1
        # for i in range(0,number_rows_dump):
                
                
        #         if dump_file_unsorted[i,0]=='1.0':
        #                 #insert_point==[j+1,0:11]
        #                 particle_sorted_array_1.append(dump_file_unsorted[i,:])
        #                 #print(np.append(dump_file_unsorted[i,:],particle_sorted_array_1[0,0:11],axis=0))
        #                 #print(dump_file_unsorted[i,:])
        #         elif dump_file_unsorted[i,0]=='1':   
        #                 #insert_point=[j+1,0:11]
        #                 particle_sorted_array_1.append(dump_file_unsorted[i,:])
        #                 #print(dump_file_unsorted[i,:])
        #         else:
        #              continue
        # #particle 2
        # for i in range(0,number_rows_dump):
                
                
        #         if dump_file_unsorted[i,0]=='2.0':
        #                 #insert_point==[j+1,0:11]
        #                 particle_sorted_array_2.append(dump_file_unsorted[i,:])
        #                 #print(np.append(dump_file_unsorted[i,:],particle_sorted_array_1[0,0:11],axis=0))
        #                 #print(dump_file_unsorted[i,:])
        #         elif dump_file_unsorted[i,0]=='2':   
        #                 #insert_point=[j+1,0:11]
        #                 particle_sorted_array_2.append(dump_file_unsorted[i,:])
        #                 #print(dump_file_unsorted[i,:])
        #         else:
        #              continue
        
        # dump_file_1=np.stack(particle_sorted_array_1)
        # dump_file_2=np.stack(particle_sorted_array_2)
       
       
        
        return dump_file_unsorted


# this function will read the output of compute bond/local from a dump file, though I think 
# will work with any dump file, with the right rows x cols per dump and skip spacing 

def dump2numpy_tensor_1tstep(dump_start_line,
                      Path_2_dump,dump_realisation_name,
                      number_of_particles_per_dump,lines_per_dump, cols_per_dump):
       
       
        

        os.chdir(Path_2_dump) 

        

        with open(dump_realisation_name, 'r') as file:
            

            lines = file.readlines() # read in list of lines 
    
            
            counter = Counter(lines) # make counter object for the list 
            
    
            n_outs=int(counter["ITEM: TIMESTEP\n"]) #find number of timestep outputs in counter

            dump_outarray=np.zeros((n_outs,lines_per_dump,cols_per_dump))
            
            # define size of skip array

            skip_spacing=lines_per_dump+9

            # +9 may need to be made variable if file output chnages
            skip_array=np.arange(1,len(lines),skip_spacing)

            for i in range(n_outs):
                k=skip_array[i]
                start=k-1
                end=start+skip_spacing
                timestep_list=lines[start:end]
                data_list=timestep_list[9:]
                data=np.zeros((lines_per_dump,cols_per_dump))
                # get rid of \n markers
                for j in range(len(data_list)):
                    data[j,:]=data_list[j].split(" ")[0:cols_per_dump]
            
                dump_outarray[i,:,:]=data


        return dump_outarray



#This function will read a log file by splicing out the whole string of data, 
# however, it may fail if you use the same variables of the non-eq run as the eq run. 
# this one could definitely be improved 

def log2numpy_reader(realisation_name,Path_2_log,thermo_vars):

   

   os.chdir(Path_2_log)
            
   # Searching for thermo output start and end line

   Thermo_data_start_line= ("   Step" + thermo_vars)
   Thermo_data_end_line =("Loop time of") 
   
   warning_start = "WARNING: Fix srd particle moved outside valid domain"
  
   Thermo_data_start_line_bytes=bytes(Thermo_data_start_line,'utf-8')
   Thermo_data_end_line_bytes =bytes(Thermo_data_end_line,'utf-8')

         
    # find begining of data run 
   with open(realisation_name) as f:
      read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) 
      
      if read_data.find(Thermo_data_start_line_bytes) != -1:
        #print('true')
         thermo_byte_start=read_data.find(Thermo_data_start_line_bytes)
         
      else:
         print('could not find thermo variables')
      
      
      # find end of run         

      if read_data.rfind(Thermo_data_end_line_bytes) != -1:
        #print('true')
         thermo_byte_end =read_data.rfind(Thermo_data_end_line_bytes)     
      else:
         print('could not find end of run')
         # correct 
     
              
         
   # Splicing out the thermo data and closing the file
   with open(realisation_name) as f:
      read_data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ ) 

         
      read_data.seek(thermo_byte_start) 

      size_of_data =thermo_byte_end-thermo_byte_start
      
      log_array_bytes=read_data.read(thermo_byte_end)
      log_array_bytes_trim=log_array_bytes[0:size_of_data]
     
      read_data.close()
            
    

   log_string= str(log_array_bytes_trim)
  

  
   warn_count  = log_string.count(warning_start) 
   
   newline_count= log_string.count('\\n')

   empty=''
   warning_regex_pattern= re.compile(r'WARNING:\sFix\ssrd\sparticle\smoved\soutside\svalid\sdomain\\n\s\sparticle\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\son\sproc\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\sat\stimestep\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\\n\s\sxnew\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\\n\s\ssrdlo\/hi\sx\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\\n\s\ssrdlo\/hi\sy\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\\n\s\ssrdlo\/hi\sz\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\\n\s\(\.\.\/fix\_srd\.cpp:[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\)')
   final_warning_regex_pattern=re.compile(r'WARNING:\sToo\smany\swarnings:\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\svs\s[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\.\sAll\sfuture\swarnings\swill\sbe\ssuppressed\s\(\.\.\/thermo\.cpp:[-+]?([0-9]*[.])?[0-9]+([eE][-+]?\d+)?\)')
  
   newline_regex_pattern= re.compile(r'\\n')
   # regex matches a floating point number [+-]?([0-9]*[.])?[0-9]+
   warning_regex_pattern.search(log_string)
   
   
   log_string=re.sub(warning_regex_pattern,empty,log_string,count=warn_count)
   log_string=re.sub( final_warning_regex_pattern,empty,log_string)
   log_string=re.sub(newline_regex_pattern,empty,log_string)
   
   
   log_string_array=log_string[:-1]
   step = "b'   Step"
   thermo_vars = step+thermo_vars
   thermo_vars_length = len(thermo_vars)
   log_string_array =log_string_array[thermo_vars_length:]

   log_string_array= log_string_array.split()

   log_float_array = [0.00]*len(log_string_array)

   for i in range(len(log_string_array)):
      log_float_array[i] = float(log_string_array[i])


   log_file = np.array(log_float_array, dtype=np.float64)
   dim_log_file =np.shape(log_file)[0]
   col_log_file = len(Thermo_data_start_line.split())
   new_dim_log_file = int(np.divide(dim_log_file,col_log_file))
   log_file = np.reshape(log_file,(new_dim_log_file,col_log_file))
   
   return log_file