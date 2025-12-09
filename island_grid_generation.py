"""
Created on Fri Jun 25 16:34:05 2021

Generate grid with circular island in the center

@author: jjacob2
"""
###########################################################
#Import packages
###########################################################
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset as nc4

###########################################################
#Input parameters
###########################################################
#latitude for Coriolis
lat_rho0 = 42.735

#grid lateral dimension
x_max = 120*1000 #m
dx = 0.5*1000
y_max = 100*1000 #m
dy = 0.5*1000

# grid vertical dimensions
h_max = 100 #m
z_max =100

#sponge layer
sponge_max = 100

###########################################################
#Files & Directories
###########################################################
root = '/home/jjacob2/runs/island_wave/rad_itw04/'


#new grid file
newfile = nc4(root+'island_spongeNES_LRGgrid.nc','r+')

###########################################################
#Computation: Build Grid
###########################################################
#new x coord's
new_x_rho = np.arange(-dx/2, x_max+dx, dx)
new_x_psi = np.arange(0, x_max+dx, dx)

#new y coord's
new_y_rho = np.arange(-dy/2, y_max+dy, dy)
new_y_psi = np.arange(0, y_max+dy, dy)

#repeat over other axis
x_rho0 = np.repeat(new_x_rho[np.newaxis, :], new_y_rho.shape[0], axis = 0)
x_v0 = np.repeat(new_x_rho[np.newaxis, :], new_y_psi.shape[0], axis = 0)
x_psi0 = np.repeat(new_x_psi[np.newaxis, :], new_y_psi.shape[0], axis = 0)
x_u0 = np.repeat(new_x_psi[np.newaxis, :], new_y_rho.shape[0], axis = 0)

y_rho0 = np.repeat(new_y_rho[:, np.newaxis], new_x_rho.shape[0], axis = 1)
y_u0 =  np.repeat(new_y_rho[:, np.newaxis], new_x_psi.shape[0], axis = 1)
y_psi0 = np.repeat(new_y_psi[:, np.newaxis], new_x_psi.shape[0], axis = 1)
y_v0 = np.repeat(new_y_psi[:, np.newaxis], new_x_rho.shape[0], axis = 1)


#flat bathymetry
h0 = np.ones(x_rho0.shape)*z_max

#new grid info
pm0 = (1/dx)*np.ones(x_rho0.shape)
pn0 = (1/dy)*np.ones(x_rho0.shape)
el0 = y_max
xl0 = x_max
f0 = np.ones(x_rho0.shape)\
     *2*7.2921159*10**(-5)*np.sin(lat_rho0*np.pi/180)

###########################################################
#Computation: Build Sponge
###########################################################
isponge_east = slice(new_x_rho.shape[0]-20, new_x_rho.shape[0])
east_lin = np.linspace(1, sponge_max, 20)

isponge_south = slice(0,20)
south_lin = np.linspace(sponge_max,1,20)

isponge_north = slice(new_y_rho.shape[0]-20, new_y_rho.shape[0])
north_lin = np.linspace(1, sponge_max, 20)

#create sponge 
visc = np.ones(x_rho0.shape)
#visc[:,isponge_west] = west_lin
visc[:,isponge_east] = east_lin
visc[isponge_south,:]= np.repeat(south_lin[:,np.newaxis], 
                                 repeats = visc.shape[1], axis = 1)
visc[isponge_north,:]= np.repeat(north_lin[:,np.newaxis], 
                                 repeats = visc.shape[1], axis = 1)

diff = np.ones(x_rho0.shape)
#diff[:,isponge_west] = west_lin
diff[:,isponge_east] = east_lin
diff[isponge_south,:]= np.repeat(south_lin[:,np.newaxis], 
                                  repeats = visc.shape[1], axis = 1)
diff[isponge_north,:]= np.repeat(north_lin[:,np.newaxis], 
                                 repeats = visc.shape[1], axis = 1)

###########################################################
#Build netCDF file
###########################################################
#create dimensions
newfile.createDimension('eta_rho',x_rho0.shape[0])
newfile.createDimension('xi_rho', x_rho0.shape[1])

newfile.createDimension('eta_psi',x_psi0.shape[0])
newfile.createDimension('xi_psi', x_psi0.shape[1])

newfile.createDimension('eta_v', x_v0.shape[0])
newfile.createDimension('xi_v', x_v0.shape[1])

newfile.createDimension('eta_u', x_u0.shape[0])
newfile.createDimension('xi_u', x_u0.shape[1])

#create variables
x_rho = newfile.createVariable('x_rho','f8',("eta_rho","xi_rho"))
x_rho.units = "m"

y_rho = newfile.createVariable('y_rho','f8',("eta_rho","xi_rho"))
y_rho.units = "m"

x_v = newfile.createVariable('x_v','f8',("eta_v","xi_v"))
x_v.units = "m"

y_v = newfile.createVariable('y_v','f8',("eta_v","xi_v"))
y_v.units = "m"

x_psi = newfile.createVariable('x_psi','f8',("eta_psi","xi_psi"))
x_psi.units = "m"

y_psi = newfile.createVariable('y_psi','f8',("eta_psi","xi_psi"))
y_psi.units = "m"

x_u = newfile.createVariable('x_u','f8',("eta_u","xi_u"))
x_u.units = "m"

y_u = newfile.createVariable('y_u','f8',("eta_u","xi_u"))
y_u.units = "m"

pm = newfile.createVariable('pm','f8',("eta_rho","xi_rho"))
pm.units = "1/m"
pn = newfile.createVariable('pn','f8',("eta_rho","xi_rho"))
pn.units = "1/m"
h = newfile.createVariable('h','f8',("eta_rho","xi_rho"))
h.units = "m"
f = newfile.createVariable('f','f8',("eta_rho","xi_rho"))
f.units = "1/s"
el = newfile.createVariable('el','f8')
el.units = "m"
xl = newfile.createVariable('xl','f8')
xl.units = "m"

visc_factor = newfile.createVariable('visc_factor','f8',("eta_rho","xi_rho"))
diff_factor = newfile.createVariable('diff_factor','f8',("eta_rho","xi_rho"))

###########################################################
#Put data into file
###########################################################
x_rho[:] = x_rho0
y_rho[:] = y_rho0
x_v[:] = x_v0
y_v[:] = y_v0
x_u[:] = x_u0
y_u[:] = y_u0
x_psi[:] = x_psi0
y_psi[:] = y_psi0

pm[:] = pm0
pn[:] = pn0
h[:] = h0
f[:] = f0
el[:] = el0
xl[:] = xl0

visc_factor[:] = visc
diff_factor[:] = diff

#Close file!
newfile.close()
