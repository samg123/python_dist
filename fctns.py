#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate.interpolate import interp2d

import numpy as np

################################################################
def make_cylinder(ri,ro,h):

    h=h/2
    size = 100

    theta = np.linspace(0, 2.*np.pi,size)
    r = np.linspace(ri,ri,2)
    theta, r = np.meshgrid(theta, r)
    x1 = r*np.cos(theta)
    y1 = r*np.sin(theta)
    z1 = np.ones(size)*np.atleast_2d(np.array([-h,h])).T

    theta = np.linspace(0, 2.*np.pi,size)
    r = np.linspace(ro,ro,2)
    theta, r = np.meshgrid(theta, r)
    x2 = r*np.cos(theta)
    y2 = r*np.sin(theta)
    z2 = np.ones(size)*np.atleast_2d(np.array([-h,h])).T

    x = np.row_stack([x1,np.flipud(x2)])
    y = np.row_stack([y1,np.flipud(y2)])
    z = np.row_stack([z1,np.flipud(z2)])

    return x,y,z
###############################################################
def pol2cart(r,theta):
    s = r * np.exp( 1j * theta )
    return np.column_stack((s.real,s.imag))
###############################################################
def cart2pol(x,y):
    d = x+(1j*y)
    return np.column_stack((np.abs(d), np.angle(d)))
###############################################################
def cscheme(*args):

    p = 1
    c = np.ones([61,3])

    for i in range(30):

        C = i**p/30**p
        c[i,:] = [C,C,1]

        C = 1-((i+1)**(1/p)/30**(1/p))
        c[31+i,:] = [1,C,C]

    return c
###############################################################
def displacements(vel,idx,dt):
    disp = np.zeros(vel.shape[0])

    for i in range(vel.shape[0]):
        disp[i] = np.sum(vel[i,:idx+1])*dt

    return disp
###############################################################
def data_setup(nframes,resolution,vel):

    r_data = np.linspace(0,1,vel.shape[0])
    t_data = np.linspace(0,1,vel.shape[1])
    f = interp2d(t_data,r_data,vel)

    r = np.linspace(0,1,resolution+1)
    r = r[:-1]+r[1]/2
    t = np.linspace(0,1,nframes)
    VEL = f(t,r)

    return VEL
###############################################################

ri = 0.5
ro = 1
h = 1
x,y,z = make_cylinder(ri,ro,h)

fig = plt.figure()
ax=Axes3D(fig)
ax.plot_surface(x,y,z,color='b',linewidth=0)
ax.set_axis_off()
plt.savefig('test',dpi=300,format='png')
plt.show()

# def displacements2():
#
#     start = start - 1
#     s = vel.shape[0] - start
#     avg_vel = np.zeros([n,vel.shape[1]])
#     f = (s/n)-1
#
#     for j in range(vel.shape[1]):
#
#         lower = 1;
#
#         for i in range(n):
#
#             upper = int(lower+f)
#             avg_vel[i,j] = np.mean(vel[lower:upper,j])
#             lower = upper
#
#
#     D = np.zeros([n,nframes])
#     f = ((time.size-start)/nframes)-1
#
#     for i in range(n):
#
#         lower = 1
#
#         for j in range(nframes):
#
#             upper = int(lower+f);
#             D[i,j]=  np.mean(avg_vel[i,lower:upper])
#             lower = upper
#
#     return D
#
###############################################################
# def cylinders_2D(n,time,V,colourscheme,tmpl,start,nframes):
#
#     tmpl = ['./output/',tmpl,'_%04d'];
#     name_counter =start;
#
#     #Setup circle data points at the tangent cylinder and CMB
#     icb = np.ones([151,2])*1221/3480;
#     cmb = np.ones([151,2])
#
#     for i in range(150):
#         icb[i,0] = 2*(np.pi/150)*i
#         cmb[i,1] = 2*(pi/150)*i
#
#     icbx,icby = pol2cart(icb[:,0],icb[:,1])
#     cmbx,cmby = pol2cart(cmb[:,1],cmb[:,2])
#
#     x = range(1,n+1)
#     f = np.sum(x)/n_tex
#     x = np.ceil(x/f)
#
#     c = 0
#     data = np.zeros([n_tex,2])
#     for i in range(x.size):
#         for j in range(x[i]):
#             data[c,0] = 2*np.pi*((j+1)/x(i))
#             data[c,1] = (i+1)/n - 0.5/n
#             c = c+1
#
#     #Start the loop
#
#     tt = np.linspace(time[start],max(time),nframes)
#     step = (max(time)-time[start])/nframes
#     fraction = np.round(V*30/abs(V).max())
#
#     theta=np.linspace(0,2*pi)
#
#     for t in range(nframes):
#         for i in range(data.shape[0]):
#             for j in range(n):
#                 flag=0
#                 rad = (j+1)/n - 0.5/n
#
#                 if data[i,1] == rad and t > 0 :
#
#                     data[i,0] = data[i,0] + V[j,t]*step
#                     flag=1
#
#                 elif data[i,1] == rad and t == 0:
#
#                     data[i,0] = data[i,0] + V[j,t]*step
#                     data[i,2] = 30
#                     flag=1
#
#
#                 if flag == 1:
#                     break
#
#

#     #Set up figure
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111, projection='3d')
#     ax1.set_zlim(0,10)
#     ax1.view_init(36, 26)
#     plt.show()
#
#         colormap(colourscheme)
#         c=colorbar;
#         caxis(cbar_range)
#         title(c,ct)
#
#         set(gca,'XTick',ticks,'XTickLabel',lables)
#         xlabel(x_axis,'FontSize',fs)
#         set(gca,'YTick',ticks,'YTickLabel',lables)
#         ylabel(y_axis,'FontSize',fs)
#
#         axis square
#
#
#
#
#
#  % Define the colour for each cylinder and draw it along with the
#  % texture
#     for i = 1:n
#
#        rin = (i-1)/n; rout = i/n;
#
#
#        colour = colourscheme(31+fraction(i,t),:);
#
#        patch([rout*cos(theta),rin*cos(theta)],[rout*sin(theta),rin*sin(theta)],colour,'linestyle','none');
#        hold on
#     end
#     [X,Y] = pol2cart(data(:,1),data(:,2));
#
#     if n > 39
#        plot(X,Y,'k.','MarkerSize',1)
#     else
#        plot(X,Y,'k.','MarkerSize',(40-n))
#     end
#
#     %Plot the ICB and CMB circles
#     plot(cx,cy,'k-','LineWidth',1)
#     hold on
#     plot(cmbx,cmby,'k-','LineWidth',1)
#
#     text = [num2str(tt(t)),titletext];
#     title(text,'FontSize',tfs)
#
#     %Save the image file
#     fnam=sprintf(tmpl,name_counter);
#     print(fnam,'-dpng')
#
#     text=['Saving frame ', num2str(name_counter),' to: ./output/'];
#     waitbar(t/nframes,bar,text)
#
#     close(h)
#     name_counter = name_counter + 1;
# end
# close(bar)
# end
