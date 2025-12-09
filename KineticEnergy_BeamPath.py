#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:33:15 2022

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
import cmocean
import critical_tools as cr

#file directories
root = '/home/jjacob2/runs/npzd_coupled/dimensional/ridge_step/'


#file name
cr02name = root+'critical_step/output02/c4/roms_his_slice.nc' 
title = 'Critical Step'
dmax = 1.0
xlim = [-150, 150]
xstep = 25
beams = 3    #1 = subcritical. 3 = critical
bdep = -1.25 #km

#load files
cr02file = nc4(cr02name, 'r')



#indices for subsetting
#slicing
idx = {'time' : slice(149,None),
       'lat' : 0,
       'lon' : slice(328,672),
       'ulon': slice(328,673)}
offset = 150.75 #km
#load grids, velocity and density
cr02 = {}
cr02['grid'] = cr.grid(cr02name, cr02file, idx, offset)
cr02['u'], cr02['w'] = cr.velocity(cr02file, idx)
cr02['rho'] = cr.tracer('rho', cr02file, idx) + 1000



#KE
cr02['KE'] = np.mean(0.5*cr02['rho']*(cr02['u']**2+cr02['w']**2),
                    axis = 0)

#compute critical slope and bounce distance
N = 1.9e-3
f = 0#1e-4
om = 2*np.pi/(3600*12.4)

m, bdist = cr.critical_slope(N, f, om, dmax)

#critical point positions
n = 3 #surface bounces
a = 10 #km

#small ridge
hmax = 1.0 #km
if (beams == 3) :
    FT, BU, BD, CR = cr.three_beam(hmax, dmax, a, bdist, m)

if (beams  == 1) :
    SC = cr.subcritical_beam(m, bdep)

#plotting
vmin = 0
vmax = 100
step = 5

ztext = 10
fig,(ax1) = plt.subplots(1,1, sharex=True,
                                   figsize = (7,6),
                                   constrained_layout=True)
cf =ax1.contourf(cr02['grid']['dist'], cr02['grid']['depth'], cr02['KE'],
                cmap = cmocean.cm.amp,
                vmin = vmin, vmax =vmax, 
                levels = np.arange(vmin, vmax+step, step),
                extend = 'max')
ax1.grid(alpha = 0.5)
if (beams == 3) :
    #critical point with 3 beams
    ax1.scatter(CR[0], CR[1],s = 50, c = 'blue',
                marker = 'o',
                linewidth = 1, edgecolors = 'black')
    ax1.text(CR[0]-12, CR[1]-50, 'Cr')
    ax1.plot(FT[0],FT[1], color = 'black', 
             linewidth = 3, alpha = 0.3)
    ax1.text(FT[0][-1], ztext, 'FT')
    ax1.plot(BU[0], BU[1], color = 'blue',
             linestyle = '--',
             linewidth = 3, alpha = 0.3)
    ax1.text(BU[0][-1], ztext, 'BU', color = 'blue')
    ax1.plot(BD[0], BD[1], color = 'black',
             linestyle = '--',
             linewidth = 3, alpha = 0.3)
    ax1.text(BD[0][-1], ztext, 'BD')

if (beams == 1) :
    #subcritical beam 
    ax1.plot(SC[0], SC[1], color = 'black',
             linestyle = ':',
             linewidth = 3, alpha = 0.3)
    ax1.text(SC[0][0], ztext, 'SC', color = 'black')

ax1.patch.set_facecolor('silver')
ax1.set_ylim([-2000,0])
ax1.set_ylabel('Depth [m]')
ax1.set_xlim(xlim)
ax1.set_xticks(np.arange(xlim[0],xlim[1]+xstep,xstep))
ax1.set_title(title, loc = 'right')
ax1.set_xlabel('Distance from Ridge Center [km]')
fig.colorbar(cf, location = 'bottom', 
             label = r'$\langle \ K. E. \rangle$ [ J ]')


# rms07 = np.sqrt(np.mean((np.diff(cr02['w'], axis = 0)/1200)**2, axis = 0))
# fig, ax = plt.subplots(1,1)
# cf = ax.contourf(cr02['grid']['dist'], cr02['grid']['depth'],rms07)
# fig.colorbar(cf, label = 'RMS $\partial w_t$ [$m \ s^{-1}$]')
# ax.patch.set_facecolor('silver')
# ax.set_xlim([-20, 20])
# ax.set_ylabel('Depth [m]')
# ax.set_xlabel('Distance from Ridge [km]')

