#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 13:41:23 2022

@author: jjacob2
"""
####################################################################
#Import Packates
####################################################################
import os
os.chdir('/home/jjacob2/python/Ideal_Ridge/critical_ridge/')
import sys
sys.path.append('/home/jjacob2/python/Ideal_Ridge/wave_forcing/')
sys.path.append('/home/jjacob2/python/NPZD/')
sys.path.append('/home/jjacob2/python/Ideal_Ridge/crticial_ridge/')

from netCDF4 import Dataset as nc4
import numpy as np
from scipy.interpolate import griddata
import  matplotlib.pyplot as plt
import cmocean
import NPZD_Implicit_Lagrangian as npzd
import critical_tools as cr

####################################################################
#Input
####################################################################
#subroot
sroot = 'critical_ridge/'
title = 'Critical Ridge'
dmax = 2.0
xlim = [-150, 150]
xstep = 25
beams = 1

####################################################################
#file directories
####################################################################
root = '/home/jjacob2/runs/npzd_coupled/dimensional/ridge_step/'
n_root = '/home/jjacob2/python/Ideal_Ridge/wave_forcing/data/'

#load files
flfile = nc4(root+sroot+'output02/roms_flt_slice.nc', 'r')
hsfile = nc4(root+sroot+'output02/roms_his_slice.nc', 'r') 

####################################################################
#Local Subroutines
####################################################################
#reshape to 3D grid (time, depth, position)
def flt3D(m, tstart, idx) :
    #slice at mask
    fl0 = np.dstack((m[tstart:, slice(idx[0], idx[1])],
                     m[tstart:, slice(idx[1], idx[2])],
                     m[tstart:, slice(idx[2], idx[3])],
                     m[tstart:, slice(idx[3], idx[4])],
                     m[tstart:, slice(idx[4], idx[5])],
                     m[tstart:, slice(idx[5], idx[6])],
                     m[tstart:, slice(idx[6], idx[7])],
                     m[tstart:, slice(idx[7], idx[8])],
                     m[tstart:, slice(idx[8], idx[9])],
                     m[tstart:, slice(idx[9], idx[10])],
                     m[tstart:, slice(idx[10], idx[11])]))
    
    #arange axes to (time, depth, poistion)
    flt = np.moveaxis(fl0, 1, -1)
    
    #remove maksed break pointw
    flt = flt[:,:,1:]
    
    return flt

####################################################################
#Tidal beams
####################################################################
#compute critical slope and bounce distance
N = 1.9e-3
f = 0#1e-4
om = 2*np.pi/(3600*12.4)

m, bdist = cr.critical_slope(N, f, om, dmax)

#critical point positions
n = 3 #surface bounces
a = 10 #km

#Beam position
cr02 = {}
hmax = 1.0 #km
cr02['xfor'], cr02['zfor'], cr02['fsb'] \
    = cr.forward_beam(hmax, dmax, a, bdist, m, n)


#distance (same units as floats, m)
dist = hsfile.variables['x_rho'][0,:]/1000-750.75

#NPZD parameters
param = npzd.params(hsfile)

####################################################################
# Lagrangian position
####################################################################
x = flfile.variables['x'][:]/1000-750.75
z = flfile.variables['depth'][:]

#float release index 
tstart = 150

#index of float depth break points
izfloat = np.where(np.ma.getmask(x[tstart,:]) == True)

#reshape position
fltx = flt3D(x, tstart, izfloat[0])
fltz = flt3D(z, tstart, izfloat[0])

####################################################################
# Computation: Light on Lagrangian parcels
####################################################################
#light
light = np.exp(param['Kext']*fltz)

#mean position
xbar = np.mean(fltx, axis = 0)
zbar = np.mean(fltz, axis = 0)

#regular grid
dx = np.mean(np.diff(dist))
grid_x , grid_y = np.meshgrid(np.linspace(np.min(dist), np.max(dist), 800),
                              np.arange(-195,0, 5))
glight = np.empty((fltx.shape[0], grid_x.shape[0], grid_x.shape[1]))
for t in range(fltx.shape[0]):
    glight[t,:,:]  =griddata((fltx[t,:,:].flatten(), fltz[t,:,:].flatten()),
                             light[t,:,:].flatten(),
                             (grid_x, grid_y),
                             method = 'linear')

####################################################################
# Computation: change in light over tidal cycle
####################################################################
m2name = np.array(np.arange(4,11,1),dtype = str)
#M2 slicing
iM2 = {}
iM2['4'] = slice(0,36)
iM2['5'] = slice(37,73)
iM2['6'] = slice(74,110)
iM2['7'] = slice(111,147)
iM2['8'] = slice(148,184)
iM2['9'] = slice(185,221)
iM2['10']= slice(222,258)
delta_IR = np.empty((len(iM2),glight.shape[1], glight.shape[2]))
for i in range(glight.shape[1]) :
    for j in range(m2name.shape[0]) :
        delta_IR[j,i,:] = (np.max(glight[iM2[m2name[j]],i,:], axis = 0) - \
                          np.min(glight[iM2[m2name[j]],i,:], axis = 0)) \
                           /np.mean(glight[iM2[m2name[j]],i,:], axis = 0)

    
####################################################################
#
####################################################################
fig, (ax1, ax2)   = plt.subplots(2,1)
cf = ax1.contourf(grid_x, grid_y, np.mean(glight, axis= 0),
                   cmap = cmocean.cm.solar,
                   vmin = 0, vmax = 0.6,
                   levels = np.arange(0,0.7,0.05),
                   extend = 'max')
if (beams != 0) :
    ax1.plot(cr02['xfor'],
        cr02['zfor'], color = 'grey', 
        linewidth = 3, alpha = 0.3)
ax1.set_xlim(xlim)
ax1.set_xticks(np.arange(xlim[0],xlim[1]+xstep,xstep))
ax1.set_xticklabels([])
ax1.set_ylim([-200,-25])
ax1.set_ylabel('Depth [m]')
ax1.grid()
ax1.text(xlim[0]*0.95, -185, '$\overline{f_{IR}}$', color = 'white')
ax1.set_title(title, loc = 'right')

cf =ax2.contourf(grid_x,grid_y,np.mean(delta_IR, axis = 0),
                 cmap = cmocean.cm.solar,
                   vmin = 0, vmax = 0.6,
                   levels = np.arange(0,0.7,0.05),
                   extend = 'max')
if (beams != 0) :
    ax2.plot(cr02['xfor'],
        cr02['zfor'], color = 'grey', 
        linewidth = 3, alpha = 0.3)
ax2.set_xlim(xlim)
ax2.set_xticks(np.arange(xlim[0],xlim[1]+xstep,xstep))
ax2.set_ylim([-200,-25])
ax2.grid()
ax2.set_xlabel('Distance from Ridge Center [km]')
ax2.set_ylabel('Depth [m]')
ax2.text(xlim[0]*0.95, -185, '$\overline{\Delta \ f_{IR}}$', color = 'white')
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(cf, cax=cbar_ax, 
             label = '$f_{IR}$')


newcmap = cmocean.tools.crop_by_percent(cmocean.cm.tempo,
                                         25, which = 'max', N = None)
fig, ax = plt.subplots(1,1,
                       figsize = (6,4))
cf =ax.contourf(grid_x,grid_y,np.mean(delta_IR*100, axis = 0))#,
                # cmap = newcmap,
                # vmin = 0, vmax = 2.5,
                # levels = np.arange(0,2.75,0.25))
if (beams != 0) :
    ax.plot(cr02['xfor'],
        cr02['zfor'], color = 'grey', 
        linewidth = 3, alpha = 0.5)
ax.set_xlim(xlim)
ax.set_xticks(np.arange(xlim[0],xlim[1]+xstep,xstep))
ax.set_ylim([-175,-5])
ax.grid(alpha = 0.5)
ax.set_xlabel('Distance from Center [km]')
ax.set_ylabel('Depth [m]')
ax.set_title(title, loc = 'right')
fig.subplots_adjust(right = 0.825)
cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
fig.colorbar(cf, cax=cbar_ax, 
             label = '$\overline{f_{\Delta IR}}$  [% Local Mean Irradiance]')
