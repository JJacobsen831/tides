#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:08:33 2023

@author: jjacob2
"""
import os
os.chdir('/home/jjacob2/python/Ideal_Ridge/critical_ridge/')
import sys
sys.path.append('/home/jjacob2/python/Ideal_Ridge/wave_forcing/')
sys.path.append('/home/jjacob2/python/NPZD/')
sys.path.append('/home/jjacob2/python/Ideal_Ridge/crticial_ridge/')

from netCDF4 import Dataset as nc4
import numpy as np
import  matplotlib.pyplot as plt
import critical_tools as cr

#file directories
root = '/home/jjacob2/runs/npzd_coupled/dimensional/sub_step/'

#options
xlim = [-100, 100]
xstep = 25
beams = 5    #3 => hmax/H = 0.3. 4 => hmax/H = 0.4. etc
n = 2 #number of beam bounces to plot

#beam parameters
a = 30 #km
hmax = 1.0 #km
N = 1.9e-3
f = 0#1e-4
om = 2*np.pi/(3600*12.4)

#load files and compute beam ray path
if (beams == 3) :
    cr_name = root + 'scr03/output02/roms_flt.nc'
    cr_file = nc4(cr_name, 'r')
    title = 'hmax/H = 0.3, step'
    xcr = 16.5
    zcr = -1.7
    xb = -7.5
    dmax = 1.4
    
    m, bdist = cr.critical_slope(N, f, om, dmax)
    SC = cr.scrbeam_path(xcr, zcr, xb, -0.001, m, n, dmax, bdist)

if (beams == 4) :
    cr_name = root + 'scr04/output02/roms_flt.nc'
    cr_file = nc4(cr_name, 'r')
    title = 'hmax/H = 0.4, step'
    xcr = 16.5
    zcr = -1.6
    xb = -7.5
    dmax = 1.2
    
    m, bdist = cr.critical_slope(N, f, om, dmax)
    SC = cr.scrbeam_path(xcr, zcr, xb, -0.001, m, n, dmax, bdist)

if (beams == 5) :
    cr_name = root + 'scr05/output02/roms_flt.nc'
    cr_file = nc4(cr_name, 'r')
    title = 'hmax/H = 0.5, step'
    xcr = 16.5
    zcr = -1.5
    xb = -7.5
    dmax = 1.0
    
    m, bdist = cr.critical_slope(N, f, om, dmax)
    SC = cr.scrbeam_path(xcr, zcr, xb, -0.001, m, n, dmax, bdist)

if (beams == 6) :
    cr_name = root + 'scr06/output02/roms_flt.nc'
    cr_file = nc4(cr_name, 'r')
    title = 'hmax/H = 0.65, step'
    xcr = 17
    zcr = -1.4
    xb = -7.5
    dmax = 0.7
    
    m, bdist = cr.critical_slope(N, f, om, dmax)
    SC = cr.scrbeam_path(xcr, zcr, xb, -0.001, m, n, dmax, bdist)



#load depth and position
x = cr_file.variables['x'][::40,:]/1000-750.75
z = cr_file.variables['depth'][::40,:]


#float release index
tstart = 150


#index of float depth break points
izfloat = np.where(np.ma.getmask(x[tstart:,:]) == True)

#reshape to 3D grid (time, depth, position)
def flt3D(m, tstart, idx) :
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

fltx = flt3D(x, tstart, izfloat[1])
fltz = flt3D(z, tstart, izfloat[1])


def tideAvg(m) :
    a = np.mean([[m[0:36,:,:]],
                  [m[37:73,:,:]],
                  [m[74:110,:,:]],
                  [m[111:147,:,:]],
                  [m[148:184,:,:]],
                  [m[185:221,:,:]],
                  [m[222:258,:,:]],
                  [m[259:295,:,:]]], axis = 0)
    
    return a[0,:,:,:]

afltx = tideAvg(fltx)
afltz = tideAvg(fltz)

#subset vertical levels
iz = slice(4,None,2)
ix = slice(1,None,4)
fx = afltx[:,iz,ix]
fz = afltz[:,iz,ix]

#plotting
fig, ax1 = plt.subplots(1,1)
if (beams != 0) :
    #subcritical beam 
    ax1.plot(SC[0], SC[1], color = 'black',
             linestyle = ':',
             linewidth = 3, alpha = 0.3)
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
