#floats
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 07:38:06 2022

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
cr05name = root+'critical_step/output02/hsimt_test/roms_his_slice.nc'

#plot label
title = 'Critical Step HSIMT'
dmax = 1.0
xlim = [-150, 150]
xstep = 25
beams = 3    #1 = subcritical. 3 = critical
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
#critical step
cr05 = {}
cr05['grid'] = cr.grid(cr05name, cr05file, idx, offset)
cr05['grid']['hours'] = cr05['grid']['time']/3600
cr05['gridw']= cr.grid(cr05name, cr05file, idx, offset, 'w')
cr05['u'] = cr05file.variables['u'][idx['time'],:, 
                                    idx['lat'],idx['ulon']]*3600
cr05['w'] = cr05file.variables['w'][idx['time'],:, 
                                    idx['lat'],idx['lon']]*3600
cr05['s'] = cr.tracer('dye_01', cr05file, idx)
cr05['kv'] = cr05file.variables['AKt'][1,2,0,1]*3600 #constant m2 hr-1

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
    m_M4, bdist_M4 = cr.critical_slope(N, f, 2*om, dmax)
    FTm4, BUm4, BDm4, CRm4 = cr.three_beam(hmax, dmax, a, bdist, m_M4)
    

if (beams  == 1) :
    SC = cr.subcritical_beam(m, bdep)

#//////////////////////
#divergence
cr05['div'] = budget.divergence(cr05)

#average tracer
cr05['s_bar'] = np.mean(cr05['s'], axis = 0)

##############################################
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
pparam['vmin_dn'] =  -0.4 #/2
pparam['vmax_dn'] = 0.4 #/4
pparam['vlev_dn'] = np.arange(pparam['vmin_dn'],
                              pparam['vmax_dn']+0.02,
                              0.02)
ztext = 1

#plotting
fig, (ax1, ax2) = plt.subplots(2,1, constrained_layout=True,
                              sharex = True,
                              figsize = (7,6), 
                              gridspec_kw = {'height_ratios': [1,3]})

cf1 = ax1.contourf(cr05['grid']['dist'],cr05['grid']['depth'],
                   cr05['s_bar'],
                   cmap = pparam['cmap_up'],
                   vmin = pparam['vmin_up'],
                   vmax = pparam['vmax_up'],
                   levels = pparam['vlev_up'])
#ax1.plot(cr05['grid']['dist'][0,ibud[0]], -75, 'bo')
#ax1.plot(cr05['grid']['dist'][0,ibud[1]], -75, 'bP')
ax1.set_xticks(pparam['xticks'])
ax1.set_xlim(pparam['xlim'])
ax1.set_ylim(pparam['ylim_up'])
ax1.set_ylabel('Depth [m]')
ax1.grid()
if (beams == 3) :
    #critical point with 3 beams
    ax1.plot(FT[0],FT[1], color = 'black', 
             linewidth = 3, alpha = 0.3)
    ax1.text(FT[0][-1], ztext, 'FT')
    ax1.plot(BU[0], BU[1], color = 'blue',
             linestyle = '-',
             linewidth = 3, alpha = 0.3)
    ax1.text(BU[0][-1], ztext, 'BU', color = 'blue')
    ax1.plot(BD[0], BD[1], color = 'white',
             linestyle = '-',
             linewidth = 3, alpha = 0.7)
    ax1.text(BD[0][-1], ztext, 'BD')
    ax1.scatter(CR[0], CR[1], s = 50,
                color = 'black',marker = 'X')
    
    #M4 critical point with 3 beams
    ax1.plot(FTm4[0],FTm4[1], color = 'black', 
             linestyle = '--',
             linewidth = 3, alpha = 0.3)
    #ax1.text(FTm4[0][-1], ztext, 'FT M4')
    ax1.plot(BUm4[0], BUm4[1], color = 'blue',
             linestyle = '--',
             linewidth = 3, alpha = 0.3)
    #ax1.text(BUm4[0][-1], ztext, 'BU M4', color = 'blue')
    # ax1.plot(BDm4[0], BDm4[1], color = 'white',
    #          linestyle = '--',
    #          linewidth = 3, alpha = 0.7)
    #ax1.text(BDm4[0][-1], ztext, 'BD M4')
    ax1.scatter(CRm4[0], CRm4[1], s = 50,
                color = 'black',marker = "P" )

ax1.set_title(title, loc = 'right')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size = '2%', pad = 0.05)
cb1 =fig.colorbar(cf1,cax = cax, orientation='vertical',
                  label = r'$\langle s \rangle$')
cb1.ax.yaxis.set_tick_params(color = 'black')
plt.setp(plt.getp(cb1.ax.axes,'yticklabels'), color = 'black')

cf2 =ax2.contourf(cr05['grid']['dist'],cr05['grid']['depth'],
                   -cr05['div'],
                   cmap = pparam['cmap_dn'],
                   vmin = pparam['vmin_dn'],
                   vmax = pparam['vmax_dn'],
                   levels = pparam['vlev_dn'],
                   extend = 'both')
#ax2.plot(cr05['grid']['dist'][0,ibud[0]], -75, 'bo')
#ax2.plot(cr05['grid']['dist'][0,ibud[1]], -75, 'bP')
if (beams == 3) :
    #critical point with 3 beams
    ax2.plot(FT[0],FT[1], color = 'black', 
             linewidth = 3, alpha = 0.3)
    ax2.text(FT[0][-1], ztext, 'FT')
    ax2.plot(BU[0], BU[1], color = 'blue',
             linestyle = '-',
             linewidth = 3, alpha = 0.3)
    ax2.text(BU[0][-1], ztext, 'BU', color = 'blue')
    ax2.plot(BD[0], BD[1], color = 'white',
             linestyle = '-',
             linewidth = 3, alpha = 0.7)
    ax2.text(BD[0][-1], ztext, 'BD')
    ax2.scatter(CR[0], CR[1], s = 50,
                color = 'black',marker = 'X')
    
    #M4 critical point with 3 beams
    ax2.plot(FTm4[0],FTm4[1], color = 'black', 
             linestyle = '--',
             linewidth = 3, alpha = 0.3)
    #ax1.text(FTm4[0][-1], ztext, 'FT M4')
    ax2.plot(BUm4[0], BUm4[1], color = 'blue',
             linestyle = '--',
             linewidth = 3, alpha = 0.3)
    #ax1.text(BUm4[0][-1], ztext, 'BU M4', color = 'blue')
    # ax1.plot(BDm4[0], BDm4[1], color = 'white',
    #          linestyle = '--',
    #          linewidth = 3, alpha = 0.7)
    #ax1.text(BDm4[0][-1], ztext, 'BD M4')
    ax2.scatter(CRm4[0], CRm4[1], s = 50,
                color = 'black',marker = "P" )

ax2.set_xlabel('Distance from Ridge Center [km]')
ax2.set_ylabel('Depth [m]')
ax2.set_ylim([-2000,0])
ax2.patch.set_facecolor('silver')
ax2.grid()

divider = make_axes_locatable(ax2)
cax2 = divider.append_axes('right', size = '2%', pad = 0.05)
cb2 =fig.colorbar(cf2,cax = cax2, orientation='vertical',
                  label = r'$-\nabla \cdot( \langle \hat{u} \ s \rangle)$')
cb2.ax.yaxis.set_tick_params(color = 'black')
plt.setp(plt.getp(cb2.ax.axes,'yticklabels'), color = 'black')


#
##power spectrum
#f_beam, Pxx_beam = welch(cr05['w'][:,185,ibud[0]],
#                         1/np.mean(np.diff(cr05['grid']['hours'])))
#f_step, Pxx_step = welch(cr05['w'][:,185,ibud[1]],
#                         1/np.mean(np.diff(cr05['grid']['hours'])))
#
#fig, (ax1, ax2) = plt.subplots(2,1)
#ax1.plot(cr05['grid']['hours'], cr05['w'][:,185,ibud[0]])
#ax1.set_title('Beam')
## ax1.set_ylim(0, 1.5)
#ax1.set_ylabel('w')
#ax2.semilogy(f_beam, Pxx_beam)
#ax2.text(1/12.4, 10**3, 'm2')
#ax2.text(2/12.4, 10**3, 'm4')
## ax2.set_ylim(10**-10, 10)
#ax2.set_ylabel('PSD')
#ax2.set_xlabel('frequency [1/hr]')
#
#fig, (ax1, ax2) = plt.subplots(2,1)
#ax1.plot(cr05['grid']['hours'], cr05['w'][:,185,ibud[1]])
#ax1.set_title('Over Step')
## ax1.set_ylim(0, 1.5)
#ax1.set_ylabel('w')
#ax2.semilogy(f_step, Pxx_step)
#ax2.text(1/12.4, 10**3, 'm2')
#ax2.text(2/12.4, 10**3, 'm4')
## ax2.set_ylim(10**-10, 10)
#ax2.set_ylabel('PSD')
#ax2.set_xlabel('frequency [1/hr]')
