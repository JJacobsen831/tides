#floats
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 07:38:06 2022

@author: jjacob2
"""
###################################################################
#Import Packages
###################################################################
import os
os.chdir('/home/jjacob2/python/Ideal_Ridge/critical_ridge/')
import sys
sys.path.append('/home/jjacob2/python/Ideal_Ridge/wave_forcing/')
sys.path.append('/home/jjacob2/python/NPZD/')
sys.path.append('/home/jjacob2/python/Ideal_Ridge/crticial_ridge/')

from netCDF4 import Dataset as nc4
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import critical_tools as cr

###################################################################
#Input 
###################################################################
#slicing
idx = {'time' : slice(149,None),
       'lat' : 0,
       'lon' : slice(328,672),
       'ulon': slice(328,673)}
offset = 150.75

###################################################################
#Files & Directories
###################################################################
root = '/home/jjacob2/runs/npzd_coupled/dimensional/ridge_step/'
n_file = 'subcrit_step/output02/roms_his_slice.nc'

#load file
ncfile = nc4(root+n_file, 'r')


###################################################################
#Read data
###################################################################
hr04 = {}
hr04['grid'] = cr.grid(root+n_file, ncfile, idx, offset)
hr04['grid']['hours'] = hr04['grid']['time']/3600
hr04['gridw']= cr.grid(root+n_file, ncfile, idx, offset, 'w')
hr04['u'] = ncfile.variables['u'][idx['time'],:, 
                                    idx['lat'],idx['ulon']]*3600
hr04['w'] = ncfile.variables['w'][idx['time'],:, 
                                    idx['lat'],idx['lon']]*3600
hr04['s'] = cr.tracer('dye_01', ncfile, idx)
hr04['kv'] = ncfile.variables['AKt'][1,2,0,1]*3600

###################################################################
#Grid: Shift to rho points
###################################################################
#shift nutrient to u and w points
pad = np.concatenate((hr04['s'][:,:,0:1],
                      hr04['s'][:,:,:], 
                      hr04['s'][:,:,-2:-1]), axis = 2)
nute_u = 0.5*(pad[:,:,:-1]+pad[:,:,1:])
pad = np.concatenate((hr04['s'][:,0:1,:],
                      hr04['s'][:,:,:], 
                      hr04['s'][:,-2:-1,:]), axis = 1)
nute_w = 0.5*(pad[:,:-1,:]+ pad[:,1:,:])

###################################################################
#Computations: Tracer Fluxes
###################################################################
#mean product
uN = np.mean(nute_u*hr04['u'], axis = 0)
wN = np.mean(nute_w*hr04['w'], axis = 0)

#gradient in x
xgrad = np.diff(uN, axis = 1)/1500
xgrad_t = np.diff(nute_u*hr04['u'], axis = 2)/1500

#gradient in z
zgrad = np.diff(wN, axis =0) \
        /np.diff(hr04['gridw']['depth'], axis = 0)
zgrad_t = np.empty(xgrad_t.shape)
zdiff = np.empty(xgrad_t.shape)
for t in range(xgrad_t.shape[0]) :
    zgrad_t[t,:,:] = np.diff(nute_w[t,:,:]*hr04['w'][t,:,:], 
                             axis = 0) \
                                /np.diff(hr04['gridw']['depth'], axis = 0)
    #second derivative
    fh = nute_w[t,2:,:]
    f_h= nute_w[t,:-2,:]
    fc = nute_w[t,1:-1,:]
    h0 = np.diff(hr04['gridw']['depth'], axis = 0)
    h = 0.5*(h0[1:,:]+h0[:-1,:])
    
    d2r_dz2 = (fh - 2*fc + f_h)/h**2
    
    #shift to rho points
    pad = np.concatenate((d2r_dz2[0:1,:], d2r_dz2, d2r_dz2[-2:-1,:]),
                         axis = 0)
    d2rdz2 = 0.5*(pad[1:,:] + pad[:-1])
    
    #diffusion
    zdiff[t,:,:] = hr04['kv']*d2rdz2
    

#zero diffusivity through top and bottom
zdiff[:,0,:] = 0
zdiff[:,-1,:] = 0

###################################################################
#Computation: Divergence
###################################################################
#divergence of uN
div = xgrad+zgrad
div_tt = xgrad_t + zgrad_t

#shift to "dt" points
div_t = 0.5*(div_tt[:-1,:,:] + div_tt[1:,:,:])
xdiv_t = 0.5*(xgrad_t[:-1,:,:] + xgrad_t[1:,:,:])
zdiv_t = 0.5*(zgrad_t[:-1,:,:] + zgrad_t[1:,:,:])
zdiff_t = 0.5*(zdiff[:-1,:,:] + zdiff[1:,:,:])
time = 0.5*(hr04['grid']['hours'][:-1] + hr04['grid']['hours'][1:])

#average tracer
rhobar = np.mean(hr04['s'], axis = 0)

#time rate of change of tracer
drho_dt = np.diff(hr04['s'], axis = 0)\
                  /(np.mean(np.diff(hr04['grid']['hours'])))

###################################################################
#Computation: Interpolate to target depth
###################################################################
#set points to extract time series
hr04['grid']['dist'][0,170]
#156 = -25.5 km outbeam; 164 = -13.5 km inbeam [Critical Step]
#164 = -13.5 km outbeam; 170 = -4.5 km inbeam [Subcritical Step, beam]
idist = 164, 170
rho = np.empty((drho_dt.shape[0], len(idist)))
drdt = np.empty((drho_dt.shape[0], len(idist)))
zdif= np.empty((drho_dt.shape[0], len(idist)))
afd = np.empty((drho_dt.shape[0], len(idist)))
xdiv =np.empty((drho_dt.shape[0], len(idist)))
zdiv = np.empty((drho_dt.shape[0], len(idist)))
for i in range(len(idist)) :
    for t in range(drho_dt.shape[0]) :
        rho[t,i] = np.interp(-75,hr04['grid']['depth'][:,idist[i]], 
                              hr04['s'][t,:,idist[i]])
        drdt[t,i] = np.interp(-75,hr04['grid']['depth'][:,idist[i]], 
                              drho_dt[t,:,idist[i]])
        zdif[t,i] = np.interp(-75,hr04['grid']['depth'][:,idist[i]], 
                              zdiff_t[t,:,idist[i]])
        afd[t,i] = np.interp(-75,hr04['grid']['depth'][:,idist[i]], 
                              div_t[t,:,idist[i]])
        xdiv[t,i] = np.interp(-75,hr04['grid']['depth'][:,idist[i]], 
                              xdiv_t[t,:,idist[i]])
        zdiv[t,i] = np.interp(-75,hr04['grid']['depth'][:,idist[i]], 
                              zdiv_t[t,:,idist[i]])

###################################################################
#Plot parameters
###################################################################
xlim = [-150, 150]
xstep = 25
pparam = {}
pparam['xlim'] = xlim
pparam['xticks'] = np.arange(pparam['xlim'][0], 
                             pparam['xlim'][1]+xstep,
                             xstep)
pparam['ylim_up'] = [-200,0]
pparam['cmap_up'] = cmocean.cm.dense
pparam['vmin_up'] =  0
pparam['vmax_up'] = 6
pparam['vlev_up'] = np.arange(pparam['vmin_up'],
                              pparam['vmax_up']+0.5,
                              0.5)
pparam['cmap_dn'] = cmocean.cm.balance
pparam['vmin_dn'] =  -0.2
pparam['vmax_dn'] = 0.2
pparam['vlev_dn'] = np.arange(pparam['vmin_dn'],
                              pparam['vmax_dn']+0.02,
                              0.02)

###################################################################
#Plotting
###################################################################
fig, (ax1, ax2) = plt.subplots(2,1, constrained_layout=True,
                              sharex = True,
                              figsize = (7,7), 
                              gridspec_kw = {'height_ratios': [1,1]})
#Density structure
cf1 = ax1.contourf(hr04['grid']['dist'],hr04['grid']['depth'],
                   rhobar,
                   cmap = pparam['cmap_up'],
                   vmin = pparam['vmin_up'],
                   vmax = pparam['vmax_up'],
                   levels = pparam['vlev_up'])
ax1.set_xticks(pparam['xticks'])
ax1.set_xlim(pparam['xlim'])
ax1.set_ylim(pparam['ylim_up'])
ax1.set_ylabel('Depth [m]')
ax1.grid()
ax1.plot(hr04['grid']['dist'][0,idist[0]], -75, 'b+')
ax1.plot(hr04['grid']['dist'][0,idist[1]], -75, 'b+')
#ax1.plot(hr04['grid']['dist'][0,idist[2]], -75, 'b+')
#ax1.plot(hr04['grid']['dist'][0,idist[3]], -75, 'b+')
#ax1.text(-3.5, 10, '$\overline{ \rho }$')
ax1.patch.set_facecolor('silver')

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size = '2%', pad = 0.05)
cb1 =fig.colorbar(cf1,cax = cax, orientation='vertical')#,
                  #ticks = [1.0,2.0,3.0])
cb1.ax.yaxis.set_tick_params(color = 'black')
plt.setp(plt.getp(cb1.ax.axes,'yticklabels'), color = 'black')

#convergence
cf2 =ax2.contourf(hr04['grid']['dist'],hr04['grid']['depth'],-div,
                  cmap = pparam['cmap_dn'],
                  vmin = pparam['vmin_dn'],
                  vmax = pparam['vmax_dn'],
                  levels = pparam['vlev_dn'],
                  extend = 'both')

ax2.set_xlabel('Distance from Center [km]')
ax2.set_ylabel('Depth [m]')
ax2.set_ylim([-2000,0])
ax2.patch.set_facecolor('silver')
ax2.grid()
ax2.plot(hr04['grid']['dist'][0,idist[0]], -75, 'b+')
ax2.plot(hr04['grid']['dist'][0,idist[1]], -75, 'b+')
# ax2.plot(x[0,idist[2]], -75, 'b+')
# ax2.plot(x[0,idist[3]], -75, 'b+')
ax2.text(-3.5, 15, r'$-\nabla \cdot (\overline{\hat{u} \ NO_3})$')

divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size = '2%', pad = 0.05)
cb2 =fig.colorbar(cf2,cax = cax2, orientation='vertical')#,
                  #ticks = [1.0,2.0,3.0])
cb2.ax.yaxis.set_tick_params(color = 'black')
plt.setp(plt.getp(cb2.ax.axes,'yticklabels'), color = 'black')


#time series
ylim = [-0.75,0.75]
ystep = 0.25
fig, (ax3,ax4) = plt.subplots(2, 1, figsize = (6,6))
ax3.plot(time/12.4, drdt[:,0], color = 'black', 
         label = r'$\frac{\partial s}{\partial t}$')
ax3.plot(time/12.4, -zdif[:,0], color = 'grey',
         label = r'$-kv \frac{\partial^2 s}{\partial z^2}$')
ax3.plot(time/12.4, -afd[:,0], color = 'indianred',
         linestyle = '-.', linewidth = 1.5,
         label = r'$-\nabla \cdot (u \ s)$')
ax3.set_ylim(ylim)
ax3.set_yticks(np.arange(ylim[0],ylim[1]+ystep,ystep))
ax3.legend(bbox_to_anchor= (1,1))
ax3.set_ylabel('[$s \ m^{-3} hr^{-1}$]')
ax3.set_xticklabels([])
ax3.grid()
ax3.set_title('Outside Beam', loc = 'right')

ax4.plot(time/12.4, -afd[:,0], color = 'indianred',
         linestyle = '-.', linewidth = 1.5,
         label = r'$-\nabla \cdot (u \ s)$')
ax4.plot(time/12.4, -zdiv[:,0], color = 'royalblue',
         label = r'$-\frac{\partial}{\partial z}(w \ s)$')
ax4.plot(time/12.4, -xdiv[:,0], color = 'darkorange',
         label = r'$-\frac{\partial}{\partial x}(u \ s)$')
ax4.set_ylim(ylim)
ax4.set_yticks(np.arange(ylim[0],ylim[1]+ystep,ystep))
ax4.legend(bbox_to_anchor= (1,1))
ax4.set_xticklabels([])
ax4.set_ylabel('[$s \ m^{-3} hr^{-1}$]')
ax4.grid()
ax4.set_xlabel('Time [$nM_2$]')


fig, (ax3,ax4) = plt.subplots(2, 1, figsize = (6,6))
ax3.plot(time/12.4, drdt[:,1], color = 'black', 
         label = r'$\frac{\partial s}{\partial t}$')
ax3.plot(time/12.4, -zdif[:,1], color = 'grey',
         label = r'$-kv \frac{\partial^2 s}{\partial z^2}$')
ax3.plot(time/12.4, -afd[:,1], color = 'indianred',
         linestyle = '-.', linewidth = 1.5,
         label = r'$-\nabla \cdot (u \ s)$')
ax3.set_ylim(ylim)
ax3.set_yticks(np.arange(ylim[0],ylim[1]+ystep,ystep))
ax3.legend(bbox_to_anchor= (1,1))
ax3.set_ylabel('[$s \ m^{-3} hr^{-1}$]')
ax3.set_xticklabels([])
ax3.grid()
ax3.set_title('Within Beam', loc = 'right')

ax4.plot(time/12.4, -afd[:,1], color = 'indianred',
         linestyle = '-.', linewidth = 1.5,
         label = r'$-\nabla \cdot (u \ s)$')
ax4.plot(time/12.4, -zdiv[:,1], color = 'royalblue',
         label = r'$-\frac{\partial}{\partial z}(w \ s)$')
ax4.plot(time/12.4, -xdiv[:,1], color = 'darkorange',
         label = r'$-\frac{\partial}{\partial x}(u \ s)$')
ax4.set_ylim(ylim)
ax4.set_yticks(np.arange(ylim[0],ylim[1]+ystep,ystep))
ax4.legend(bbox_to_anchor= (1,1))

ax4.set_ylabel('[$s \ m^{-3} hr^{-1}$]')
ax4.grid()
ax4.set_xlabel('Time [$nM_2$]')
