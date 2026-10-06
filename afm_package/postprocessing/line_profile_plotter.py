# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:05:39 2026

@author: ericf
""" 

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython
from matplotlib.backends.qt_compat import QtCore

from postprocessing.single_plane_plotter import bulk_potential_slices


def line_profile_plotter():
    #Put the entire .py file inside a function so I can call it in run_all.py
    
    #This line changes the matplotlib backend to allow for interactive plots, so the user can draw lines
    get_ipython().run_line_magic('matplotlib', 'qt')
    

    class LineBuilder:
        #Code taken and modified from Matplotlib Release 3.4.3 John Hunter, Darren Dale, Eric Firing, Michael Droettboom, and the matplotlib development team
        #Refer to Section 3.2 for details on events
        def __init__(self, line):
            self.line = line
            self.xs = list(line.get_xdata())  #The (x,y) position (in the coordinates of the plot) of the click
            self.ys = list(line.get_ydata())
            self.line_counter = 0  #Used later for keeping track of whether a click was the first or second click of a new line
            self.cid = line.figure.canvas.mpl_connect('button_release_event', self) #Matplotlib event that triggers and generates xdata, ydata, etc. upon button release
    
    
        
        def line_plotter(self, x_pos1, x_pos2, y_pos1, y_pos2):
            """
            This function is going to take the two (x,y) values that make the start and end of the line,
            and from them define a chain of connected points from which to plot the potentials
            """
            x1 = x_pos1
            x2 = x_pos2
            y1 = y_pos1
            y2 = y_pos2
            
            potential_values = [] #Preparing an array to hold the potential values along the line

            
            scaling = 1  #Oversample points on the line drawn by this factor
            #Length of the simulation in nm divided by the number of points along one axis i.e. "density"
            step = (length/(density*scaling)) #This line assumes that the simulation space is perfectly cubic
            
            #Gets the total length of the line so the total # of points to sample can be determined
            x_length = (x2 - x1)**2 
            y_length = (y2 - y1)**2
            line_length = np.sqrt(x_length + y_length)
            
            total_points = np.arange(0, line_length, step) #Needs to be an integer, so instead of rounding line_length/step I'm making an array
            x_points = np.linspace(x1, x2, len(total_points))
            y_points = np.linspace(y1, y2, len(total_points))
    	
            for i in range(len(total_points)):
                #Each iteration of the for loop saves the potential at each (x,y) point on the line
                col = int((x_points[i] - left) / (right - left) * ncols) # <- AI solution to relating x[i] to a point on the plot
                #row = int((y_points[i] - top) / (bottom - top) * nrows)  #If the imshow plot is origin = 'upper'
                row = int((y_points[i] - bottom) / (top - bottom) * nrows) #If the imshow plot is origin = 'lower'
                
                potential_values.append(potential_array[row,col]) 
    
    
            fig2, ax2 = plt.subplots() #Creates the line plot of the potential
            ax2.plot(total_points, potential_values)
            ax2.set(xlabel='Drawn Line Length (nm)', ylabel= 'Potential (V)',
                   title='Potential Over the Selected Line') #This could be more robust
            ax2.grid()
            return
                
            
        def __call__(self, event): 
            if event.inaxes!=self.line.axes: return #Do nothing if the click is outside the plot
            self.xs.append(event.xdata) #Save the xdata of where the click was made
            self.ys.append(event.ydata) #Save the ydata of where the click was made
            self.line.set_data(self.xs, self.ys) 
            self.line.figure.canvas.draw() #Draws a line over the potential slice
            self.line_counter += 1 #Increase the previously created line_counter to keep track of what click one is on
            if self.line_counter % 2 == 0: #If it is the second clck
                #Uses the last two clicks to give positional arguments to the line_plotter function 
                self.line_plotter(self.xs[0], self.xs[1], self.ys[0], self.ys[1])
                
                self.line.remove()  #Delete the last line drawn
                self.xs = [] #Delete the x values saved in the list
                self.ys = [] #Delete the y values saved in the list
                self.line, = ax.plot([],[]) #Start again with an empty line to populate and plot
    
    #Call the function that creates the figure, returns the figure length and # of points along an axis ("density")
    fig, length, density = bulk_potential_slices() 
    
    
    ax = fig.axes[0]   #ax is defined as the first set of axes in the figure
    im = ax.images[0]  #im is defined as the first image on the first axes, the underlying imshow dataset of the potentials
    
    
    potential_array = im.get_array() #Gets the information needed about the slice
    left, right, bottom, top = im.get_extent()  #Gets the size of the slice
    nrows, ncols = potential_array.shape[:2]  #Defines the rows and columns of the slice
    
    
    line, = plt.plot([], []) # Initializes empty line, to be filled by the class
    LineBuilder(line)
    
    #When this function is called the matplotlib backend goes from interactive back to non-interactive
    def switch_to_inline(*args):
        get_ipython().events.unregister('pre_run_cell', switch_to_inline) #First stop the event from triggering infinitely
        get_ipython().run_line_magic('matplotlib', 'inline')
    
    def check_figures():
        if not plt.get_fignums():  #Checks to see if all figures are closed. If they are it continues.
            timer.stop()  #If every figure has been closed it stops the timer
            #At the predetermined event 'pre_run_cell' this calls the switch_to_inline function
            get_ipython().events.register('pre_run_cell', switch_to_inline) 
    
    
    timer = QtCore.QTimer() 
    timer.setInterval(500)  # 500 millisecond interval
    timer.timeout.connect(check_figures) # On this interval run the check_figures function
    timer.start()  # Starts the timer and starts running the _check_figures function every 500 ms




    return
