import numpy as np
import os 
import matplotlib.pyplot as plt
from collections import Counter
from numpy.linalg import inv as matinv

def stress_tensor_averaging(e_end,
                            labels_stress,
                            trunc1,
                            trunc2,
                            spring_force_positon_tensor_tuple,j_):
    stress_tensor=np.zeros((e_end,6))
    stress_tensor_std=np.zeros((e_end,6))
    stress_tensor_reals=np.zeros((e_end,j_,6))
    stress_tensor_std_reals=np.zeros((e_end,j_,6))
    for l in range(6):
        for i in range(e_end):
            for j in range(j_):
                cutoff=int(np.round(trunc1*spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0]))
                aftercutoff=int(np.round(trunc2*spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0]))
                # print(spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0])
                # print(cutoff)
                # print(aftercutoff)
                data=np.ravel(spring_force_positon_tensor_tuple[i][j,cutoff:aftercutoff,:,l])
                stress_tensor_reals[i,j,l]=np.mean(data)
                stress_tensor_std_reals[i,j,l]=np.std(data)
                stress_tensor=np.mean(stress_tensor_reals, axis=1)
                stress_tensor_std=np.mean(stress_tensor_std_reals, axis=1)
    return stress_tensor,stress_tensor_std

def stress_tensor_averaging_var_trunc(e_end,
                            labels_stress,
                            trunc1,
                            trunc2,
                            spring_force_positon_tensor_tuple,j_):
    stress_tensor=np.zeros((e_end,6))
    stress_tensor_std=np.zeros((e_end,6))
    stress_tensor_reals=np.zeros((e_end,j_,6))
    stress_tensor_std_reals=np.zeros((e_end,j_,6))
    for l in range(6):
        for i in range(e_end):
            for j in range(j_):
                cutoff=int(np.round(trunc1[i]*spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0]))
                aftercutoff=int(np.round(trunc2[i]*spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0]))
                # print(spring_force_positon_tensor_tuple[i][j,:,:,l].shape[0])
                # print(cutoff)
                # print(aftercutoff)
                data=np.ravel(spring_force_positon_tensor_tuple[i][j,cutoff:aftercutoff,:,l])
                stress_tensor_reals[i,j,l]=np.mean(data)
                stress_tensor_std_reals[i,j,l]=np.std(data)
                stress_tensor=np.mean(stress_tensor_reals, axis=1)
                stress_tensor_std=np.mean(stress_tensor_std_reals, axis=1)
    return stress_tensor,stress_tensor_std


def plotting_stress_vs_strain(spring_force_positon_tensor_tuple,
                              i_1,i_2,j_,
                              strain_total,cut,aftcut,stress_component,label_stress,erate):
    

    mean_grad_l=[] 
    for i in range(i_1,i_2):
        #for j in range(j_):
            cutoff=int(np.round(cut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            aftcutoff=int(np.round(aftcut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))


            strain_plot=np.linspace(cut*strain_total,aftcut*strain_total,spring_force_positon_tensor_tuple[i][0,cutoff:aftcutoff,:,stress_component].shape[0])
            cutoff=int(np.round(cut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            aftcutoff=int(np.round(aftcut*spring_force_positon_tensor_tuple[i][0,:,:,stress_component].shape[0]))
            stress=np.mean(spring_force_positon_tensor_tuple[i][:,:,:,stress_component],axis=0)
            stress=stress[cutoff:aftcutoff]
            gradient_vec=np.gradient(np.mean(stress,axis=1))
            mean_grad=np.mean(gradient_vec)
            mean_grad_l.append(mean_grad)
            #print(stress.shape)
            # plt.plot(strain_plot,np.mean(stress,axis=1))
            # plt.ylabel(labels_stress[stress_component],rotation=0)
            # plt.xlabel("$\gamma$")
            # plt.plot(strain_plot,gradient_vec, label="$\\frac{dy}{dx}="+str(mean_grad)+"$")

            #plt.legend()
            #plt.show()

    plt.scatter(erate,mean_grad_l, label=label_stress)
    plt.xlabel("$\dot{\gamma}$")
    plt.ylabel("$\\frac{d\\bar{\sigma}_{\\alpha\\beta}}{dt}$", rotation=0,labelpad=20)
    #plt.show()

def plot_stress_tensor(t_0,t_1,
                       stress_tensor,
                       stress_tensor_std,
                       j_,n_plates, labels_stress,marker,cutoff,erate,e_end,ls_pick):
    for l in range(t_0,t_1):
          plt.errorbar(erate[cutoff:e_end], stress_tensor[cutoff:,l], yerr =stress_tensor_std[cutoff:,l]/np.sqrt(j_*n_plates), ls=ls_pick,label=labels_stress[l],marker=marker[l] )
          plt.xlabel("$\dot{\gamma}$")
          plt.ylabel("$\sigma_{\\alpha\\beta}$",rotation=0,labelpad=20)
    plt.legend()      
    #plt.show()


def compute_n_stress_diff(stress_tensor, 
                          stress_tensor_std,
                          i1,i2,
                          j_,n_plates,
                          ):
    n_diff=stress_tensor[:,i1]- stress_tensor[:,i2]
    n_diff_error=np.sqrt(stress_tensor_std[:,i1]**2 +stress_tensor_std[:,i2]**2)/np.sqrt(j_*n_plates)
   
   
        
    return n_diff,n_diff_error


def compute_plot_n_stress_diff(stress_tensor, 
                          stress_tensor_std,
                          i1,i2,
                          j_,n_plates,
                          erate,e_end,
                          ylab):
    n_diff=stress_tensor[:,i1]- stress_tensor[:,i2]
    n_diff_error=np.sqrt(stress_tensor_std[:,i1]**2 +stress_tensor_std[:,i2]**2)/np.sqrt(j_*n_plates)
    #plt.errorbar(erate[:e_end], n_diff, yerr =n_diff_error, ls='none' )
    plt.scatter(erate[:e_end],n_diff )
    plt.ylabel(ylab, rotation=0, labelpad=20)
    plt.xlabel("$\dot{\gamma}$")
    plt.legend()  
    plt.show() 
        
    return n_diff,n_diff_error


def folder_check_or_create(filepath,folder):
     os.chdir(filepath)
     # combine file name with wd path
     check_path=filepath+"/"+folder
     print((check_path))
     if os.path.exists(check_path) == 1:
          print("file exists, proceed")
          os.chdir(check_path)
     else:
          print("file does not exist, making new directory")
          os.chdir(filepath)
          os.mkdir(folder)
          os.chdir(filepath+"/"+folder)
  
def window_averaging(i,window_size,input_tuple,array_size,outdim1,outdim3):
    
    output_array_final=np.zeros((outdim1,array_size[i],outdim3))
    for k in range(outdim1):
        input_array=input_tuple[i][k,:,:]
        df = pd.DataFrame(input_array)
        output_dataframe=df.rolling(window_size,axis=0).mean()
        output_array_temp=output_dataframe.to_numpy()

        #print(output_array_temp)
        non_nan_size=int(np.count_nonzero(~np.isnan(output_array_temp))/outdim3)
        print("non_nan_size", non_nan_size)
        output_array_final[k,:,:]=output_array_temp

    return output_array_final, non_nan_size


 
def quadfunc(x, a):

    return a*(x**2)

def linearfunc(x,a,b):
    return (a*x)+b 

def linearthru0(x,a):
     return a*x 

def powerlaw(x,a,n):
    return a*(x**(n))



class realisation():
     def __init__(self,realisation_full_str,data_set,realisation_index_):
          self.realisation_full_str= realisation_full_str
          self.data_set= data_set
          self.realisation_index_=realisation_index_
     def __repr__(self):
        return '({},{},{})'.format(self.realisation_full_str,self.data_set,self.realisation_index_)
realisations_for_sorting_after_srd=[]
realisation_split_index=6
erate_index=15

def org_names(split_list_for_sorting,unsorted_list,first_sort_index,second_sort_index):
    for i in unsorted_list:
          realisation_index_=int(i.split('_')[first_sort_index])
          data_set =i.split('_')[second_sort_index]
          split_list_for_sorting.append(realisation(i,data_set,realisation_index_))


    realisation_name_sorted=sorted(split_list_for_sorting,
                                                key=lambda x: ( x.data_set,x.realisation_index_))
    realisation_name_sorted_final=[]
    for i in realisation_name_sorted:
          realisation_name_sorted_final.append(i.realisation_full_str)
    
    return realisation_name_sorted_final



def dump2numpy_tensor_1tstep(dump_start_line,
                      Path_2_dump,dump_realisation_name,
                      number_of_particles_per_dump,lines_per_dump, cols_per_dump):
       
       
        

        os.chdir(Path_2_dump) #+simulation_file+"/" +filename

        

        with open(dump_realisation_name, 'r') as file:
            

            lines = file.readlines()
            
            counter = Counter(lines)
            
            #print(counter.most_common(3))
            n_outs=int(counter["ITEM: TIMESTEP\n"])
            dump_outarray=np.zeros((n_outs,lines_per_dump,cols_per_dump))
            #print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])
            skip_spacing=lines_per_dump+9
            skip_array=np.arange(1,len(lines),skip_spacing)
            for i in range(n_outs):
                k=skip_array[i]
                # timestep_list=[]
                start=k-1
                end=start+skip_spacing
                timestep_list=lines[start:end]
                data_list=timestep_list[9:]
                #print(data_list[0])
                #print(len(data_list))
                data=np.zeros((lines_per_dump,cols_per_dump))
                for j in range(len(data_list)):
                    data[j,:]=data_list[j].split(" ")[0:cols_per_dump]
            
                dump_outarray[i,:,:]=data


        return dump_outarray


def dump2numpy_box_coords_1tstep(Path_2_dump,dump_realisation_name,
                     lines_per_dump):
       
       
        

        os.chdir(Path_2_dump) #+simulation_file+"/" +filename

        

        with open(dump_realisation_name, 'r') as file:
            

            lines = file.readlines()
            
            counter = Counter(lines)
            
            #print(counter.most_common(3))
            n_outs=int(counter["ITEM: TIMESTEP\n"])
            #dump_outarray=np.zeros((n_outs,lines_per_dump,cols_per_dump))
            #print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])
            skip_spacing=lines_per_dump+9
            #print(skip_spacing)
            skip_array=np.arange(1,len(lines),skip_spacing)
            data_list=[]
            for i in range(n_outs):
                k=skip_array[i]
               
                # timestep_list=[]
                start=k-1
                end=start+skip_spacing
                timestep_list=lines[start:end]
                

                data_list.append(timestep_list[:9])
                #print(data_list)

                #print(data_list[0])
                #print(len(data_list))
                # data=np.zeros((lines_per_dump,cols_per_dump))
                # for j in range(len(data_list)):
                #     data[j,:]=data_list[j].split(" ")[0:cols_per_dump]
            
                #dump_outarray[i,:,:]=data


        return data_list


def cfg2numpy_coords(Path_2_dump,dump_realisation_name,
                      number_of_particles_per_dump,cols_per_dump):
       
       
        

        os.chdir(Path_2_dump) #+simulation_file+"/" +filename

        

        with open(dump_realisation_name, 'r') as file:
            

            lines = file.readlines()
            box_vec_lines=lines[2:11] 
            #print(box_vec_lines[0])
            box_vec_array=np.zeros((9))
            
            for i in range(9):
                 
                 box_vec_array[i]=box_vec_lines[i].split(" ")[2]
                 #print(coord)
                 

            box_vec_array=np.reshape(box_vec_array,(3,3))

            lines=lines[15:] #remove intial file lines 

            for i in range(len(lines)-1):
                #print(lines)
                try:
                   lines.remove("C \n")
                   lines.remove("5.000000 \n")
                
                except:
                    continue 
                
                # print(lines)
                # print(i)
                
            
            
            dump_outarray=np.zeros((number_of_particles_per_dump,cols_per_dump))
            #print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])
             
            for i in range(len(lines)):
                dump_outarray[i,:]=lines[i].split(" ")[0:cols_per_dump]
                #print(lines[i].split(" ")[0:cols_per_dump])
                



        return dump_outarray,box_vec_array

# def cfg2numpy_coords_plate(Path_2_dump,dump_realisation_name,
#                       number_of_particles_per_dump,cols_per_dump):
       
       
        

#         os.chdir(Path_2_dump) #+simulation_file+"/" +filename

        

#         with open(dump_realisation_name, 'r') as file:
            

#             lines = file.readlines()
#             box_vec_lines=lines[2:11] 
#             #print(box_vec_lines[0])
#             box_vec_array=np.zeros((9))
            
#             for i in range(9):
                 
#                  box_vec_array[i]=box_vec_lines[i].split(" ")[2]
#                  #print(coord)
                 

#             box_vec_array=np.reshape(box_vec_array,(3,3))

#             lines=lines[15:] #remove intial file lines 

#             for i in range(len(lines)-1):
#                 #print(lines)
#                 try:
#                    lines.remove("C \n")
#                    lines.remove("5.000000 \n")
                
#                 except:
#                     continue 
                
#                 # print(lines)
#                 # print(i)
                
            
            
#             dump_outarray=np.zeros((number_of_particles_per_dump,cols_per_dump))
#             #print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])
             
#             for i in range(len(lines)):
#                 dump_outarray[i,:]=lines[i].split(" ")[0:cols_per_dump]
#                 #print(lines[i].split(" ")[0:cols_per_dump])
                



#         return dump_outarray,box_vec_array


def cfg2numpy_coords(Path_2_dump,dump_realisation_name,
                      number_of_particles_per_dump,cols_per_dump):
       
       
        

        os.chdir(Path_2_dump) #+simulation_file+"/" +filename

        

        with open(dump_realisation_name, 'r') as file:
            

            lines = file.readlines()
            box_vec_lines=lines[2:11] 
            #print(box_vec_lines[0])
            box_vec_array=np.zeros((9))
            
            for i in range(9):
                 
                 box_vec_array[i]=box_vec_lines[i].split(" ")[2]
               
                 

            box_vec_array=np.reshape(box_vec_array,(3,3))

            lines=lines[15:] #remove intial file lines 

            # print(len(lines))
            # print(lines)
            
            # list filtering 
            lines= list(filter(("C \n").__ne__, lines))
            lines= list(filter(("5.000000 \n").__ne__, lines))
            lines= list(filter(("0.050000 \n").__ne__, lines))
           

            # for i in range(len(lines)-1):
                


            #     try:
            #        lines.remove("C \n")
            #        lines.remove("5.000000 \n")
        
            #        lines.remove('0.050000 \n')
                
            #     except:
            #         continue 
                
            #     # print(lines)
            #     # print(i)

            # print(lines[596])
                
            
            
            dump_outarray=np.zeros((number_of_particles_per_dump,cols_per_dump))
            #print(counter["ITEM: ENTRIES c_spring_f_d[1] c_spring_f_d[2] c_spring_f_d[3] c_spring_f_d[4] c_spring_f_d[5] c_spring_f_d[6]\n"])
            #print(lines)
            for i in range(len(lines)):
                # print(i)
                # print(lines[i])
                dump_outarray[i,:]=lines[i].split(" ")[0:cols_per_dump]
                #print(lines[i].split(" ")[0:cols_per_dump])
                



        return dump_outarray,box_vec_array

def q_matrix_transform(box_vector_array_cfg,box_vec_array_upper_tri,
                      full_cfg_coord_after_sorted,posvel_from_dump_sing_sorted,
                       n_particles,n_plates,db_forces,db_dirns,k):
    
    # inverting lammps frame box vectors 
    inv_box_vec_array=matinv(box_vec_array_upper_tri) 

    # multiply  Q= FL^{-1}
    Q_matrix=np.matmul(box_vector_array_cfg.T,inv_box_vec_array) 
    
    unscaled_cfg=np.zeros((n_particles,3))
    transform_dump_coords=np.zeros((n_particles,3))  
    transform_force_dump=np.zeros((n_plates,6))


    
    for m in range(n_particles):
        # convert scaled cfg coords to unscaled coords by multiplication by box vector from cfg
        unscaled_cfg[m]=np.matmul(box_vector_array_cfg.T,full_cfg_coord_after_sorted[m][0:3])
        # transform dump coords by q matrix 
        transform_dump_coords[m]=np.matmul(Q_matrix,posvel_from_dump_sing_sorted[m][2:5])

    
    # compared unscaled cfg to transformed dump, they should match 
    # this part needs more work
    comparison=np.abs(unscaled_cfg-transform_dump_coords)


    #apply Q matrix transform to force components and direction components 
    for m in range(n_plates):
        transform_force_dump[m,0:3]=np.matmul(Q_matrix,db_forces[k,m])
        transform_force_dump[m,3:6]=np.matmul(Q_matrix,db_dirns[k,m])



   


    # if np.any(comparison>1e-3):
    #     #print("comparison incorrect")
    #      break


    
    return transform_dump_coords,transform_force_dump

def q_matrix_transform_plate(box_vector_array_cfg,box_vec_array_upper_tri,
                      full_cfg_coord_after_sorted,posvel_from_dump_sing_sorted,
                       n_particles,n_plates,db_forces,db_dirns,k):
    
                    # inverting lammps frame box vectors 
                    inv_box_vec_array=matinv(box_vec_array_upper_tri) 

                    # multiply  Q= FL^{-1}
                    Q_matrix=np.matmul(box_vector_array_cfg.T,inv_box_vec_array) 
                    
                    unscaled_cfg=np.zeros((n_particles,3))
                    transform_dump_coords=np.zeros((n_particles,3))  
                    transform_force_dump=np.zeros((n_plates*3,6))
                    transform_dump_velocities=np.zeros((n_particles,3)) 


                    
                    for m in range(n_particles):
                        # convert scaled cfg coords to unscaled coords by multiplication by box vector from cfg
                        unscaled_cfg[m]=np.matmul(box_vector_array_cfg.T,full_cfg_coord_after_sorted[m][0:3])
                        # transform dump coords by q matrix 
                        transform_dump_coords[m]=np.matmul(Q_matrix,posvel_from_dump_sing_sorted[m][2:5])
                        transform_dump_velocities[m]=np.matmul(Q_matrix,posvel_from_dump_sing_sorted[m][5:8])

                    
                    # compared unscaled cfg to transformed dump, they should match 
                    # this part needs more work 
                    # print(full_cfg_coord_after_sorted[0][0:3])
                    # comparison=np.abs(unscaled_cfg-transform_dump_coords)
                    # print(np.max(comparison))
                    # plt.plot(comparison)
                    # plt.show()
                    # plt.plot(np.ravel(unscaled_cfg))
                    # plt.show()
                    # plt.plot(np.ravel(transform_dump_coords))
                    # plt.show()



                    #apply Q matrix transform to force components and direction components 
                    for m in range(n_plates*3):
                        transform_force_dump[m,0:3]=np.matmul(Q_matrix,db_forces[k,m])
                        transform_force_dump[m,3:6]=np.matmul(Q_matrix,db_dirns[k,m])



                    
                    return transform_dump_coords,transform_force_dump,transform_dump_velocities
