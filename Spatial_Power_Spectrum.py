#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 09:02:14 2023

Map of dominant frequency and power 

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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import critical_tools as cr
import div_tools as budget
from scipy.signal import welch 

#file directories
root = '/home/jjacob2/runs/npzd_coupled/dimensional/ridge_step/'
cr05name = root+'critical_step/output02/c4_test/roms_his_slice.nc'

#plot label
title = 'Critical Step'
dmax = 1.0
xlim = [-150, 150]
xstep = 25
beams = 0    #1 = subcritical. 3 = critical
bdep = -1.25 #km
#ibud = 164, 170 #subcritical
#ibud = 156, 164 #critical
ibud = 164, 170

#load files
cr05file = nc4(cr05name, 'r')

#slicing
idx = {'time' : slice(149,None),
       'lat' : 0,
       'lon' : slice(328,672),
       'ulon': slice(328,673)}
offset = 150.75 #km

#load data
cr05 = {}
cr05['gridw']= cr.grid(cr05name, cr05file, idx, offset, 'w')
cr05['gridw']['hours'] = cr05['gridw']['time']/3600
cr05['w'] = cr05file.variables['w'][idx['time'],:, 
                                    idx['lat'],idx['lon']]*3600

#compute critical slope and bounce distance
N = 1.9e-3
f = 0#1e-4
om = 2*np.pi/(3600*12.4)
m, bdist = cr.critical_slope(N, f, om, dmax)

#critical point positions
n = 3 #surface bounces
a = 10 #km
hmax = 1 #km

if (beams == 3) :
    FT, BU, BD, CR = cr.three_beam(hmax, dmax, a, bdist, m)

if (beams  == 1) :
    SC = cr.subcritical_beam(m, bdep)

#power spectrum at each point
fs = 1/np.mean(np.diff(cr05['gridw']['hours'])) #sample frequency 
pxx = np.empty((cr05['w'].shape[1], cr05['w'].shape[2]))
f = np.empty(pxx.shape)
for i in range(pxx.shape[0]) :
    for j in range(pxx.shape[1]) :
        #power spectrum
        fr, pwr = welch(cr05['w'][:, i, j])
        
        #highest power
        mpwr = np.max(pwr)
        
        #index of highest power
        ipwr = np.where(pwr == mpwr)
        
        pxx[i,j] = mpwr
        f[i,j] = fr[ipwr]

#plot dominant frequency
ztext = 10
fig, ax1 = plt.subplots(1,1, figsize = (7,4))
cf = ax1.contourf(cr05['gridw']['dist'], 
                  cr05['gridw']['depth'],
                  f*12.4,
                  vmin = 0, vmax = 2)#,
                  #levels = np.arange(0, 4, 1))

if (beams == 3) :
    #critical point with 3 beams
    ax1.scatter(CR[0], CR[1],s = 50, c = 'blue',
                marker = 'o',
                linewidth = 1, edgecolors = 'black')
    #ax1.text(CR[0]-12, CR[1]-50, 'Cr')
    ax1.plot(FT[0],FT[1], color = 'white', 
             linewidth = 3, alpha = 0.3)
    ax1.text(FT[0][-1], ztext, 'FT')
    ax1.plot(BU[0], BU[1], color = 'blue',
             linestyle = '--',
             linewidth = 3, alpha = 0.3)
    ax1.text(BU[0][-1], ztext, 'BU', color = 'blue')
    ax1.plot(BD[0], BD[1], color = 'white',
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
ax1.set_xlim([-150, 150])
ax1.set_ylim([-200,0])
ax1.set_ylabel('Depth [m]')
ax1.set_xlabel('Distance from Ridge Center [km]')
fig.colorbar(cf, 
             label = '$f_{max}/M_2$')

        

