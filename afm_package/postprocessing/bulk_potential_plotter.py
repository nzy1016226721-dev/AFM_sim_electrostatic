# -*- coding: utf-8 -*-
"""
@author: Eric Friesen Waldner, September 23, 2026 
"""
import os
import glob
import numpy as np
import sys
import re
from matplotlib import pyplot as plt
from natsort import natsorted


def file_selection():
   """
   Function for generating lists of numbers seperated by commas to give to the potential plotting function
   For the use case where one wishes to generate plots for every n-th .npy file from a list
   """ 
   print("Enter the first file #, the last file #, and the interval, seperated by commas.")
   _input = input("e.g. 0,50,2 to return a list from 0 to 50 (inclusive) of every even number: ").strip()
   start, stop, step = _input.strip().split(',') #Splits the string using commas as a delimiter
   start, stop, step = int(start), int(stop), int(step) #np.arange needs integers
   arr = np.arange(start, stop+1, step)  #+1 to make the stopping point inclusive
   comma_str = ','.join(map(str, arr)) #turns the arr [1 2 3] into 1,2,3
   print(f"Plotting Files {comma_str}")
   return comma_str


def list_to_array(num_str):
    """
    The function takes a list of numbers delineated with commas and dashes,
    eg. of the form 1,2,3,6-9,12,... and returns an array of the given numbers,
    e.g. [1,2,3,6,7,8,9,12]. 
    Credit to Stack Overflow user Tim Roberts for the code:
    https://stackoverflow.com/questions/73043130/parse-a-string-of-mixed-comma-and-hyphen-separated-digits-to-return-a-sorted-lis     
    """
    
    num = []
    for part in num_str.split(','):
        p1 = part.split('-')
        if len(p1) == 1:
            num.append(int(p1[0]))
        else:
            num.extend(list(range(int(p1[0]),int(p1[1])+1)))
    return num


def list_to_array_float(num_str):
    """
    Same as list_to_array() but for the use case of the user inputting non-int single values
    Also catches mistakes in the input and defaluts to 0.5 when blank.
    """
    if num_str.strip() == "":  #Default case is 50nm, one might wish the default was L/2
        return [50]
    # Reject anything that isn't a digit, comma, period, dash, or whitespace
    if not re.fullmatch(r'[0-9.,\-\s]*', num_str):
        print("Invalid input. Restart Postprocessing")
        sys.exit()
    num = []
    
    for part in num_str.split(','):
        p1 = part.split('-')
        if len(p1) == 1:
            num.append(float(p1[0]))
        else:
            num.extend(list(range(float(p1[0]),float(p1[1])+1))) #This only works for ints
            #One could change it to step through with np.arange somehow, such an edge case i won't waste my time
    return num


def folder_finder():
    
    #Asks the user what folder they would like to look for .npy files in
    main_afm_folder = os.path.dirname(os.path.dirname(__file__)) #Returns the path to the folder run_all.py is in
    npy_directory = input("Enter the path to the folder containing the .npy files to be plotted (default: outputs): ").strip()
    if not npy_directory:  #If the user doesn't type anything 
        npy_directory = os.path.join(main_afm_folder, "outputs")
    if not os.path.isdir(npy_directory): #If the directory specified does not exist
        print(f" '{npy_directory}' folder not found")
    return(npy_directory)


def npy_file_sorter(npy_directory):
    
    #Finds all files ending in .npy in a folder and return them sorted alphabetically
    npy_files = natsorted(glob.glob(os.path.join(npy_directory, '*.npy')))  #This will sort numerically like: afm_1, afm_10, afm_2, etc.
    if not npy_files: #if the folder is empty
        print("No .npy files found.")
        return
    print("\nAvailable .npy files:")
    for i, f in enumerate(npy_files):  #The for loop prints every file found to show the user
        print(f"  {i}. {os.path.basename(f)}")
    return(npy_files)


def title_maker_line_plot(filename, choice):  
    """
    If the user wants a automatic title this function uses regular expressions 
        to find the name of the config file for the default title.
    If the user wants a custom title it prompts them to enter a string
    """
    if choice == "n" or choice == "": #Use regex to find the simulation name from the config filename
        pattern = re.compile(r".*?ID_\d+_(?P<sim_name>.+?)\.json_.*?")
        m = re.match(pattern, filename)
        sim_name_or_custom_name = m.groupdict()["sim_name"] #Save it to be used for the default filename
    elif choice == "y":
        sim_name_or_custom_name = input("Input the desired custom plot title: ")
    else: 
        print("Invalid choice, try again:")
        return(title_maker_line_plot)
    return(sim_name_or_custom_name)


def regex_pattern_for_npys(filename):
    """
    re.match pulls out the size of the simulation box, and the resolution of the grid, 
      i.e. # of rows and columns in the .npy file, so we can relate "50nm" to 
           an exact row of the .npy file
    Note that this assumes the simulation size and resolution will be the same for all files analyzed
    """
    pattern = re.compile(
        r".*?Lx_(?P<Lx>[\d.]+)nm_Ly_(?P<Ly>[\d.]+)nm_Lz_(?P<Lz>[\d.]+)nm_.*?"
        r"Nx_(?P<Nx>\d+)_Ny_(?P<Ny>\d+)_Nz_(?P<Nz>\d+)_.*?"
        r"Bias_(?P<Bias>[\d.]+)V")
    m = re.match(pattern, filename)
    Lx, Ly, Lz = int(m.groupdict()["Lx"]), int(m.groupdict()["Ly"]), int(m.groupdict()["Lz"])
    Nx, Ny, Nz = int(m.groupdict()["Nx"]), int(m.groupdict()["Ny"]), int(m.groupdict()["Nz"]) #I need the simulation resolution
    Bias = m.groupdict()["Bias"]
    return(Lx, Ly, Lz, Nx, Ny, Nz, Bias)


def line_plotter(potentials_to_plot, axis_choice, positions, save_plot):
    """
    Plots lines of potential given the path to the .npy file, and given 
        the selected axes through simulation space on which to plot the lines,
        and given the locations of the lines in simulation space.
    This will plot all the lines in the same spot from all of the simulation outputs selected
        together in one plot.
    """
         
    #Pull relevant parameters out of a representative filename
    representative_file = os.path.basename(potentials_to_plot[0])  #representative_file will be resused for the plot titling function later
    Lx, Ly, Lz, Nx, Ny, Nz, Bias = regex_pattern_for_npys(representative_file)
   
    #Prepares the colours for the plot
    n_files = len(potentials_to_plot)
    colours = plt.cm.plasma(np.linspace(0, 1, n_files)) #makes a list of colours, one for each potential line

    #Asks user if they just want to plot over a specific range on the chosen axis,
    #  say, from 50 to 60nm in z only.
    restricted_range = input(f"Do you want to plot over a limited range in {axis_choice}? y/n (default: n): ").strip()
    if restricted_range == "y":
        lower_bound = float(input("Input the lower bound: "))
        upper_bound = float(input("Input the upper bound: "))
    #Takes the list of input x and/or y and/or z coordinates of the desired lines and
    #  pairs them in every possible way. e.g. x = 0.1, 0.2 and y = 0.3, 0.4 will give 4 pairs
    #   (0.1,0.3), (0.1,0.4), (0.2,0.3), (0.2,0.4)
    position_pairs =  np.array(np.meshgrid(positions[0], positions[1])).T.reshape(-1, 2)
    print(f"The positions to be plotted are: {position_pairs}") 
    for i in position_pairs:
        #The list of all possible pairs of line locations in the simulation space
        #The line of potential from each file will be plotted together for
        fig, ax = plt.subplots() #Defined now before the loop begins to create the plots
                                 #  so that all the lines for a given position are plotted together
                                 #  i.e. that each position is plotted separately.
        
        for j in enumerate(potentials_to_plot): #Choose the j-th file in the list of paths provided
            phi_file = np.load(j[1])  #Its a tuple with # and file path, so always load the second part
            
            line_colour = colours[j[0]] #The colour of an individual line in the plot is defined here
            
            #Now the legend will be purely defined by tip height from sample in nm
            file_name = os.path.basename(j[1]) #Gets just the filename for creating the legend title
            #Pulls out the tip height from the bottom of the simulation space using regular expressions
            tip_height_pattern = re.compile(r".*?TipHeightfromBottom_(?P<tip_height>[\d.]+nm)_.*?")
            m = re.match(tip_height_pattern, file_name)
            tip_height = m.groupdict()["tip_height"] #Saves this future label for the legend
            
    
            #The code will take the position pairs generated by the user input
            #  and convert them from real values into the corresponding/closest
            #  part of the output potential .npy file matrix,
            #  by dividing the input position by length and multiplying by resolution
            if axis_choice == "x":
                iy = round((i[0]/Ly)*(Ny))
                iz = round((i[1]/Lz)*(Nz))
                line_xlabel = 'x (nm)'
                line_title_pos = f'at y = {i[0]:.2f} nm, z = {i[1]:.2f} nm'
                if restricted_range == "y":
                    i_lower = round((lower_bound/Lx)*(Nx))
                    i_upper = round((upper_bound/Lx)*(Nx))
                    new_Nx = round((Nx/Lx)*(upper_bound-lower_bound)) #Need only as many points as there are over the new axis
                    phi_line = phi_file[i_lower:i_upper, iy, iz]
                    x_axis = np.linspace(lower_bound, upper_bound, new_Nx)
                else:
                    x_axis = np.linspace(0,Lx,Nx) #The x-axis will be in nm, from the size of the simulation
                    phi_line = phi_file[:, iy, iz]
                    
                ax.plot(x_axis, phi_line, color = line_colour, lw =2, label= tip_height)
                #This will plot the sliced potential values along the axis chosen in a unique colour, labelled with tip height in nm
           
            if axis_choice == "y":
                ix = round((i[0]/Lx)*(Nx))
                iz = round((i[1]/Lz)*(Nz))
                line_xlabel = 'y (nm)'
                line_title_pos = f'at x = {i[0]:.2f} nm, z = {i[1]:.2f} nm'
                if restricted_range == "y":
                    i_lower = round((lower_bound/Ly)*(Ny))
                    i_upper = round((upper_bound/Ly)*(Ny))
                    new_Nx = round((Ny/Ly)*(upper_bound-lower_bound)) #Need only as many points as there are over the new axis
                    phi_line = phi_file[ix, i_lower:i_upper, iz]
                    x_axis = np.linspace(lower_bound, upper_bound, new_Nx)
                else:
                    x_axis = np.linspace(0,Ly,Ny) #The x-axis will be in nm, from the size of the simulation
                    phi_line = phi_file[ix, :, iz]
 
                ax.plot(x_axis, phi_line, color = line_colour, lw = 2, label= tip_height)
                
            if axis_choice == "z":
                ix = round((i[0]/Lx)*(Nx))
                iy = round((i[1]/Ly)*(Ny))
                line_xlabel = 'z (nm)'
                line_title_pos = f'at x = {i[0]:.2f} nm, y = {i[1]:.2f} nm'
                if restricted_range == "y":
                    i_lower = round((lower_bound/Lz)*(Nz))
                    i_upper = round((upper_bound/Lz)*(Nz))
                    new_Nx = i_upper-i_lower #Need only as many points as there are over the new axis
                    phi_line = phi_file[ix, iy, i_lower:i_upper]
                    x_axis = np.linspace(lower_bound, upper_bound, new_Nx)
                else:
                    x_axis = np.linspace(0,Lz,Nz) #The x-axis will be in nm, from the size of the simulation
                    phi_line = phi_file[ix, iy, :]
                    
                ax.plot(x_axis, phi_line, color = line_colour, lw = 2, label= tip_height)
            
            
        ax.set_xlabel(line_xlabel)
        ax.set_ylabel('Potential (V)') 
        
        #Now generate a default title or let the user input a custom title
        print("Would you like a custom title?")
        print("example default title: Potential of 'config name' along z at x = 50.00 nm, y = 50.00 nm, Tip Bias Applied: 1.00 V")
        title_choice = input("y/n (default: n): ").strip()
        sim_name_or_custom_name = title_maker_line_plot(representative_file, title_choice) #if/else function, allows user to modify the title
        fig_title = f'{sim_name_or_custom_name}' #Saves something meaningful for the saved figures title
        
        if title_choice == "y":
            
            ax.set_title(f'{sim_name_or_custom_name}') #This makes whatever the user inputs into the whole title
        else: #Generate the default title
            ax.set_title(f' Potential of {sim_name_or_custom_name} along {axis_choice}' 
                     f' {line_title_pos}, Tip Bias Applied: {Bias} V', wrap = True)
        ax.legend(title = "Tip z-position")

        if len(ax.get_legend_handles_labels()[1]) > 12: #If there are too many lines for the legend
            handles, labels = ax.get_legend_handles_labels()
            #Takes the first last, and 12 total evenly spaced line labels and only shows those in the legend
            desired_label_index = np.linspace(0, len(labels)-1, 12) 
            #Does some rounding to the nearest integer
            desired_label_index =np.round([x for x in desired_label_index])
            desired_label_index = [int(x) for x in desired_label_index]
            #Makes a new list of handles and labels for ax.legend() to plot
            desired_handles = [handles[i] for i in desired_label_index]
            desired_labels = [labels[i] for i in desired_label_index]
            ax.legend(desired_handles, desired_labels, title = "Tip z-position", loc='center left', bbox_to_anchor=(1, 0.5))
        
        ax.grid(True, alpha=0.5) #Saves the plot as long as the internal file structure of the package is unaltered
        if save_plot == True:
            outer_folder = os.path.dirname(os.path.dirname(__file__)) #The afm_package folder
            save_plots_folder = os.path.join(outer_folder, "outputs", "saved_plots") #goes down to the desired folder
            fname = os.path.join(save_plots_folder, fig_title)
            plt.savefig(fname)

        plt.show()
        return
       
 
def plane_plotter(potentials_to_plot, axis_choice, plane_positions, save_plot):
    
    #Pull relevant parameters out of a representative filename
    representative_file = os.path.basename(potentials_to_plot[0])  #representative_file will be resused for the plot titling function later
    Lx, Ly, Lz, Nx, Ny, Nz, Bias = regex_pattern_for_npys(representative_file)
    
    
    print(f"Slices will be plotted along {axis_choice} at {plane_positions} nm") 
    for j in enumerate(potentials_to_plot): #Choose the j-th file in the list of paths provided
        phi_file = np.load(j[1])
        
        file_name = os.path.basename(j[1]) #Gets just the filename for creating the legend title
        #Pulls out the tip height from the bottom of the simulation space using regular expressions
        tip_height_pattern = re.compile(r".*?TipHeightfromBottom_(?P<tip_height>[\d.]+)nm_.*?")
        m = re.match(tip_height_pattern, file_name)
        tip_height = m.groupdict()["tip_height"] #Saves this future label for the plot

        for i in plane_positions: #The list of all desired slice heights in the chosen axis of the simulation space
            #The slice of potential from each file will be plotted, at each height
            fig, ax = plt.subplots() #Create one imshow() plot of the slice at each height:
                
            #The code will now take the position of each slice generated by the user input
            #  and convert them from real values into the corresponding/closest
            #  part of the output potential .npy file matrix,
            #  by dividing the input position by length and multiplying by resolution
            if axis_choice == "xy":
                iz = int((i/Lz)*(Nz))
                phi_slice = phi_file[:,:,iz]
                slice_extent = [0, Lx, 0, Ly]
                slice_title_pos = f'z = {iz:.2f} nm'
                line_xlabel = 'x (nm)'
                line_ylabel = 'y (nm)'
                
            if axis_choice == "yz":
                ix = int((i/Lx)*(Nx))
                phi_slice = phi_file[ix,:,:]
                slice_extent = [0, Ly, 0, Lz]
                slice_title_pos = f'x = {ix:.2f} nm'
                line_xlabel = 'y (nm)'
                line_ylabel = 'z (nm)'
                
            if axis_choice == "xz":
                iy = int((i/Ly)*(Ny))
                phi_slice = phi_file[:,iy,:]
                slice_extent = [0, Lx, 0, Lz]
                slice_title_pos = f'y = {iy:.2f} nm'
                line_xlabel = 'x (nm)'
                line_ylabel = 'z (nm)'

            ax.set_xlabel(line_xlabel)
            ax.set_ylabel(line_ylabel)  
            
            #Now generate a default title or let the user input a custom title
            print("Would you like a custom or default title?")
            print("example default title: Potential of 'config name' along z at x = 50.00 nm, y = 50.00 nm, Tip Bias Applied: 1.00 V")
            title_choice = input("y/n (default: n): ").strip()
            sim_name_or_custom_name = title_maker_line_plot(representative_file, title_choice) #if/else function, allows user to modify the title
            fig_title = f'{sim_name_or_custom_name}' #Saves something meaningful for the file name if the plot is saved
            if title_choice == "y":
                ax.set_title(f'{sim_name_or_custom_name}')  #This makes the user input the whole title
            else: #Generate the default title
                ax.set_title(f' Potential of {sim_name_or_custom_name} along the {axis_choice} plane' 
                         f' at {slice_title_pos}, with Tip Height: {tip_height} nm, Tip Bias Applied: {Bias} V', wrap=True)
            
            
            image = plt.imshow(phi_slice.T, origin='lower', interpolation = "none", extent=slice_extent,
                                  cmap='RdBu_r', aspect='equal') #aspect makes all of the pixels square
            plt.colorbar(image, label='Potential (V)')
          
            
            if save_plot == True:
                outer_folder = os.path.dirname(os.path.dirname(__file__)) #The afm_package folder
                save_plots_folder = os.path.join(outer_folder, "outputs", "saved_plots") #goes down to the desired folder
                fname = os.path.join(save_plots_folder, fig_title)
                plt.savefig(fname)
            plt.show()
    return


def bulk_potential_plots():
    
    #Asks the user what folder they would like to look for .npy files in
    npy_directory = folder_finder()
    #Next finds all files ending in .npy and return them sorted alphabetically
    npy_files = npy_file_sorter(npy_directory)
        
    #Now gives the user an option to select every n-th file
    interval_opt_in = input("Would you like to plot every few .npy files on some interval? y/n (default: n): ").strip()
    if interval_opt_in == "y":
        comma_str = file_selection()
        list_of_npys = list_to_array(comma_str) #turns 1,3,2 6-8 into [1 3 2 6 7 8]
        potentials_to_plot = []
        for i in list_of_npys:
            potentials_to_plot.append(npy_files[i]) #Saves to a new array the selected file paths

    else:
        choice = input("Select file number(s) (or press Enter for all): ").strip()
        if choice.isdigit():
            nth_file = int(choice)
            if 0 <= nth_file < len(npy_files): #Double check the number entered is valid
                potentials_to_plot = [npy_files[nth_file]]
                print(f"Plotting {potentials_to_plot}")
        if choice== "":
            print("Plotting all .npy files.")
            potentials_to_plot = npy_files
        
        elif isinstance(choice, str): #If the user input contains symbols other than just digits, such as commas or dashes
             list_of_npys = list_to_array(choice) #turns 1,2,3, 6-8,... into [1 2 3 6 7 8]
             potentials_to_plot = []
             for i in list_of_npys:
                 potentials_to_plot.append(npy_files[i])
             #print("Selected Files:"
                  # f"{potentials_to_plot}") 
        else:
            print("Invalid Input. Start Again")
            return bulk_potential_plots()


    #Now we have either a single potential file to plot, or a list of .npy files to plot

    #Next the user selects either plane or line plots, and the desired axis
    def plane_or_line_and_axis_choice():
            
        plane_or_line = input("Do you want to plot 'planes' ('p') or 'lines' ('l') of potential (default: lines): ").strip()
        
        if plane_or_line == "planes" or plane_or_line == "p":
            axis_choice = input("What axis would you like to slice? 'xz', 'yz', or 'xy' (default: xy): ").strip()
            if axis_choice == "xz":
                print("Plotting in xz.")
            elif axis_choice == "yz":
                print("Plotting in yz.")
            elif axis_choice == "xy" or axis_choice == "":
                axis_choice = "Plotting in xy."
                axis_choice = "xy"
            else:
                print("Invalid choice, plotting in xy.")
                axis_choice = "xy"
                
        elif plane_or_line == "lines" or plane_or_line == "l" or plane_or_line == "":
            print("plotting lines")
            axis_choice = input("Along which axis would you like to plot a line of potential? (default: z): ").strip()
            if axis_choice == "x":
                axis_choice = "x"
            elif axis_choice == "y":
                axis_choice = "y"
            elif axis_choice == "z" or axis_choice == "":
                axis_choice = "z"    
            else: 
                print("Invalid choice")
                axis_choice = "z"
        else: 
            print("Invalid choice, try again:")
            return plane_or_line_and_axis_choice() #run the function again
        return(axis_choice)
    
    #Now call the function defined above to get the desired axis_choice and positions
    axis_choice = plane_or_line_and_axis_choice()
    print(f"Plotting in {axis_choice}")

    
    #Now asks the user where on the axes chosen they would like to take slices or lines
    if axis_choice == "xz" or axis_choice == "yz" or axis_choice == "xy": #For the plane plots:
        print("plotting planes")
        print(f"Where along the not chosen axis in the simulation space would you like to plot the plane slice(s) of potential in {axis_choice}? (default: 50 nm)")
        plane_positions = input("Input a list of values in nm (e.g. take a slice in xy for z values (10, 20, 30) nm) (default 50 nm): ").strip()
        plane_positions = list_to_array_float(plane_positions)
        
    else:  #And for the line plotting case:
        if axis_choice == "x":
            print("Where along the y-axis would you like to plot the lines of potential? (default 50 nm)")
            line_positions_y = input("Input a list of values in nm e.g. 10,20,40,... ").strip()
            line_positions_y = list_to_array_float(line_positions_y) #Convert user input string of ints or floats into array 
            print("Where along the z-axis (in fractional coordinates) would you like to plot the lines of potential? (default 50 nm)")
            print("Note that every possible pair of (y,z) values from the lists given will be plotted")
            line_positions_z = input("Input a list values in nm e.g. 10, 21.5, 50.2,... ").strip()  
            line_positions_z = list_to_array_float(line_positions_z)
            
            line_positions = np.array([line_positions_y, line_positions_z])
            
        if axis_choice == "y":
           print("Where along the y-axis would you like to plot the lines of potential? (default 50 nm)")
           line_positions_x = input("Input a list of values in nm e.g. 10,20,40,... ").strip()
           line_positions_x = list_to_array_float(line_positions_x) 
           print("Where along the z-axis would you like to plot the lines of potential? (default 50 nm)")
           print("Note that every possible pair of (x,z) values from the lists given will be plotted")
           line_positions_z = input("Input a list values in nm e.g. 10, 21.5, 50.2,... ").strip()
           line_positions_z = list_to_array_float(line_positions_z)
           
           line_positions = np.array([line_positions_x, line_positions_z])
           
        if axis_choice == "z":
            print("Where along the z-axis would you like to plot the lines of potential? (default 50 nm)")
            line_positions_x = input("Input a list of values in nm e.g. 10,20,40,... ").strip()
            print("Where along the y-axis would you like to plot the lines of potential? (default 50 nm)")
            print("Note that every possible pair of (x,y) values from the lists given will be plotted")
            line_positions_y = input("Input a list values in nm e.g. 10, 21.5, 50.2,... ").strip()
            #Default case is handled by the list_to_array_float() function. Not a great solution
            line_positions_x = list_to_array_float(line_positions_x) 
            line_positions_y = list_to_array_float(line_positions_y)
            
            line_positions = np.array([line_positions_x, line_positions_y])
       
        
   #Now the plotting function itself can be called given the paths to the npy files and the 
   #     locations in the data the user desires to plot
    save_plot = input("Would you like to save the plot(s) generated? y/n (default: n): ")
    if save_plot == "y":
        save_plot = True
    else:
        save_plot = False
            
    if axis_choice == "xy" or axis_choice ==  "yz" or axis_choice == "xz":
         plane_plotter(potentials_to_plot, axis_choice, plane_positions, save_plot)
         
    if axis_choice == "x" or axis_choice ==  "y" or axis_choice ==  "z":
        line_plotter(potentials_to_plot, axis_choice, line_positions, save_plot)

    return


