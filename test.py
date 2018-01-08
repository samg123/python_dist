v.shape

a = np.array([[1,2,3],[4,5,6]])
a
np.reshape(a,(3,2))
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg' #If using ffmpeg to encode, file path to executible must be specified here.
import matplotlib.animation as anim
from scipy.interpolate.interpolate import interp2d

# File parameters
save = 0                  #Switch to save the animation (1=save, 0=don't save)
data_file = '/Users/Sam/Documents/University/internship2/python_dist/cox_etal_2014.txt'  #Data file
out_file = 'mymovie.mp4'  #Filename for output movie (must contain desired file extension)
nframes=200               #Number of frames in movie
fps = 20                  #Frames per second
dpi = 300                 #Dots per inch resolution

#Create function to interpolate data file to pass into flow()
v = np.genfromtxt(data_file,delimiter=',')
n_r, n_t = v.shape
t = np.linspace(0,n_t-1,n_t)
r_data = np.linspace(0,1,n_r)
f = interp2d(t,r_data,v)

d_max = np.max(np.round(np.max(v),decimals=1),np.round(np.min(v),decimals=1))
d_min = -d_max
# Variables passed into flow() function [loop counter, any other variables user desires].
# Must contain the first variable (loop counter) as it is hard coded lower down. Any others are
# optional to the user however flow() must be editted to correctly unpack these variables.
variables = [0,f,n_t,nframes]

# Plot parameters
title = 'test'                #Title of plot
cmap = 'jet'                  #Colourmap used in plot
resolution = 100              #Number of points in radius and theta.
levels = np.linspace(d_min,d_max,60) #Contour levels
c_ticks = np.linspace(levels[0],levels[-1],5)  #Tick values for colourbar.

def flow(R,THETA,variables):
    '''
    flow(R,THETA,variables)

    Function to evaluate the flow velocity at radius R and angle THETA. THETA either
    represents azimuth or co-latitude depending if the plot is intended as an equitorial
    or meridional view. 'variables' comprises of a list of variables that may need to be
    passed into the function to evaluate it.

    Inputs:
        R: 2D array of shape (resolution,resolution).
        THETA: 2D array of shape (resolution,resolution).
        variables: list

    Output:
        u: 2D array of shape (resolution,resolution).
    '''
    # Unpack variables list
    i, f, n_t, n_frames = variables

    time = (i/n_frames)*n_t
    # Calculate u
    u_temp = f(time,R[0,:])

    U = np.zeros((resolution,resolution))
    for j in range(resolution):
        U[j,:] = u_temp[:,0]

    return U

def flow_plot(ax, THETA, R, z, *args,**kwargs):
    '''
    flow_plot(ax, THETA, R, z, *args,**kwargs)

    Function to plot the flow onto the given axes.

    Inputs:
        ax: matplotlib.pyplot axis class to plot flow onto.
        THETA: 2D array of shape (resolution,resolution)
        R: 2D array of shape (resolution,resolution)

    Outputs:
        matplotlib.pyplot plot class
    '''
    # Return the plot. Can change this to include arguments/keyword arguments not hard coded in.
    # Default is set to contourf but this can be changed to contour without any loss of functionality.
    return ax.contourf(THETA,R,z, *args, **kwargs)

###############################################################################################
# No more editting should need to take place from here on out.
###############################################################################################
###############################################################################################


# Set up radius and theta domains.
r = np.linspace(0,1,resolution)
theta = np.linspace(0,2*np.pi,resolution)
R, THETA = np.meshgrid(r,theta)

# Create the figure and axes
fig, ax=plt.subplots(1,1,subplot_kw=dict(projection='polar'))
ax.set_xticks([])
ax.set_yticks([])
ax.set_theta_offset(0.5*np.pi)
ax.set_theta_direction('clockwise')

# Plot the initial flow, colourbar and title
p = [flow_plot(ax,THETA,R,flow(R,THETA,variables),levels, cmap=cmap )]
cbar = plt.colorbar(p[0],ax=ax, ticks=c_ticks)
plt.title(title)

# Define the function to update the plot that FuncAnimation uses.
def update(i):
    #Remove the existing contours
    for tp in p[0].collections:
        tp.remove()
    #Update loop counter in 'variables'
    variables[0]=i
    #Update the contours
    p[0] = flow_plot(ax,THETA,R,flow(R,THETA,variables),levels, cmap=cmap)
    return p[0].collections

# Create the animimation (blit=True does not display the plot for some reason?)
ani = anim.FuncAnimation(fig, update, frames=nframes,
                                         interval=10, blit=False, repeat=True)

# If the user wants to save the animation, then do so
if save == 1:
    print('Saving file '+out_file+' at '+str(dpi)+'dpi and '+str(fps)+'fps')
    ani.save(out_file,dpi=dpi,fps=fps)
    print('SAVED')

# Display the animation
plt.show()
