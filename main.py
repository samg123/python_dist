#Torsional Wave Visualisation Program 1.0 by Sam Greenwood (27/9/16)
#
#This program offers 3 ways to visualise torsional wave propogation in the
#earths core as movies. 4 external functions are provided with this program:
#3 for the different visualisations and one for drawing cylinders. Of the
# visualisations, the first 2 show a 2D equatorial slice and the 3rd
# shows a full 3D representation of the core. 'scatterplot.m' generates a
# random set of data points across the equatorial slice then advects them
# acording to the velocity at that radius. 'cylinders_2D.m' and
# 'cylinders_3D.m' approximate the core to a user defined number of
# rotating concentic cylinders. All three types of animation output a series
# of sequentially numbered png images saved into a folder named 'output'.
# This program is also set up to encode them into a movie by using ffmpeg
# on a unix system. If ffmpeg is not installed or the system is not based
# on unix commands then the Movie Encoding section of this code is not
# needed. It is left to the user to decide how they would like the frames
# encoded into a movie.
# If you would like to be notified of any further updates to this code or
# have any questions, feel free to contact me at ee12sg@leeds.ac.uk
#
#
# Data formats:
#
# Velocity data needs to be a matrix of mxn (rowsxcolumns) where m=number
# of radial points and n=number of time steps, hence giving the velocity
# for every radius at each time step. Define the name of this matrix as
# 'vel'.
# Time data needs to be an array of length 1xn (same as in vel) with the
# value of time at each time step. This will be used to calculate the
# advection for all time as well as be displayed on the plot. Define the
# name of this array as 'time'.
#
#######################################################################################
# User Defined Variables
#######################################################################################
# Vfile = name of file containing the velocity data
# Tfile = name of file containing the time data
# start = index of time array to start from
# step = number of time steps each image file represents (to avoid potentially creating
#         potentially 1000's of images)
# intro_anim= Set to 1 to also produce introduction animation, set to 0 to
#             skip it (only applies to 3D cylinders).
# tmpl = name template for output files. Files will be saved with the
#        defined tmpl, then with a number index following e.g. tmpl_0001.png,
#        tmpl_0002.png etc. All will be saved into a folder called 'output'.
# mov_out = name of movie file to encode. This must contain the desired
#           extension. By default this code is set up to use ffmpeg on a unix system
#           to encode the movie and so any file types supported by ffmpeg can be used.
#           If ffmpeg is not installed then the 2 lines of code in the Movie Encoding
#           section of this program are not needed.

vfile = 'file.txt'
tfile = 'file2.txt'
start=0
step=15
intro_anim=1
tmpl = 'scatter_grace'
mov_out = 'scatter_grace.mp4'
######################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.mp1_toolkits.mplot3d import Axes3D
from fctns import *

#Load the data
vel = np.genfromtxt(vfile)
time = np.genfromtxt(tfile)


#Check with user directory output is ok to use

if os.path.isdir("output"):
    an = input('output folder already exists. Is it ok to write files to it? (y/n) %\n CAUTION: MATLAB WILL OVERWRITE FILES OF THE SAME NAME %\n');
    assert an == 'y', "Rename existing folder or rename folder this program will use to store output"
else:
    os.makedirs('./output')


#####################################################################################
#Load the colour scheme
cs = cscheme()

#Ask user for choice of plot and run relavant program.
choice = input('Which type of plot would you like? (enter 1/2/3) %\n 1. Scatter Plot %\n 2. 2D cylinder approximation %\n 3. 3D cylinder approximation %\n ');
assert choice == 1 or choice == 2 or choice ==3, "Enter 1/2/3"
if choice > 1:
    n = input('how many cylinders would you like to approximate to?: %\n')

    if choice == 2:
        cylinders_2D(n,time,vel,cs,tmpl,start,step)
    elif choice == 3:
        cylinders_3D(n,time,vel,cs,tmpl,start,step,intro_anim)

elif choice == 1:
    scatterplot(time,vel,cs,tmpl,start,step)
