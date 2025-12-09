#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:08:33 2023

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
import  matplotlib.pyplot as plt
import pyroms_bpowell as pyroms
import cmocean
import critical_tools as cr

###################################################################
#Input
###################################################################
title = 'Subcritical Ridge'

#parameters
dmax = 2.0
xlim = [-250, 50]
xstep = 25
beams = 0

#compute critical slope and bounce distance
N = 1.9e-3
f = 0#1e-4
om = 2*np.pi/(3600*12.4)
m, bdist = cr.critical_slope(N, f, om, dmax)

#critical point positions
raypath = {}
n = 4 #surface bounces
a = 10 #km
hmax = 1.0 #km

#fluid properties
N = 1.9e-3 #BV
f = 0#1e-4 #Corilois
om = 2*np.pi/(3600*12.4) #M2 tidal frequency

###################################################################
#Files & Directories
###################################################################
root = '/home/jjacob2/runs/npzd_coupled/dimensional/ridge_step/'
file = 'subcrit_ridge/output02/roms_flt_slice.nc'
cr05 = nc4(root+file, 'r')

#load depth and position
x = cr05.variables['x'][:]/1000-750.75
z = cr05.variables['depth'][:]

###################################################################
#Local Subroutines
###################################################################
def flt3D(m, tstart, idx) :
  """
  Reshape to 3D grid (time, depth, position)
  Input: m (ndarray)
         tstart (float)
         idx (list)     : 
  """
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
    flt = np.moveaxis(fl0, 1, -1)
    
    return flt

def tideAvg(m) :
"""
  Used for Tide averaging, selects floats
"""
    a = np.mean([[m[0:36,:,1:]],
                  [m[37:73,:,1:]],
                  [m[74:110,:,1:]],
                  [m[111:147,:,1:]],
                  [m[148:184,:,1:]],
                  [m[185:221,:,1:]],
                  [m[222:258,:,1:]],
                  [m[259:295,:,1:]]], axis = 0)
    
    return a[0,:,:,:]

###################################################################
#Computations: Tidal Beams
###################################################################
#compute critical slope and bounce distance
m, bdist = cr.critical_slope(N, f, om, dmax)

#critical point positions
raypath = {}
n = 4 #surface bounces
a = 10 #km
hmax = 1.0 #km
raypath['xfor'], raypath['zfor'], raypath['fsb'] \
    = cr.forward_beam(hmax, dmax, a, bdist, m, n)

#float release index
tstart = 150

#index of float depth break points
izfloat = np.where(np.ma.getmask(x[tstart,:]) == True)

###################################################################
#Computations: Tidal averaging
###################################################################
# Reshape array
fltx = flt3D(x, tstart, izfloat[0])
fltz = flt3D(z, tstart, izfloat[0])

# Subset floats
afltx = tideAvg(fltx)
afltz = tideAvg(fltz)

#subset vertical levels
iz = slice(4,None,2)
ix = slice(1,None,7)
fx = afltx[:,iz,ix]
fz = afltz[:,iz,ix]

###################################################################
#Plotting
###################################################################
#Largrangian trajectories
fig, ax1 = plt.subplots(1,1)
if (beams != 0) :
    ax1.plot(raypath['xfor'], raypath['zfor'], 
             color = 'grey', 
             linewidth = 3, alpha = 0.8)
for i in range(fx.shape[1]) :
    ax1.plot(fx[:,i,:], fz[:,i,:],
             color = 'k')
ax1.set_xlim(xlim)
ax1.set_xticks(np.arange(xlim[0],xlim[1]+xstep,xstep))
ax1.set_xlabel('Distance from Ridge Center [km]')
ax1.set_ylim([-265, 0])
ax1.set_ylabel('Depth [m]')
ax1.set_title(title, loc = 'right')
ax1.grid()
