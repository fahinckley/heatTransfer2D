# -*- coding: utf-8 -*-
"""
2-D heat transfer simulation using Landlab
Created on Sat Mar 12 18:00:48 2016

@author: Franklin Hinckley
"""

# Load packages
import numpy as np
import matplotlib.pyplot as plt
from landlab import RasterModelGrid
from landlab.plot.imshow import imshow_node_grid
from landlab import FIXED_VALUE_BOUNDARY

# Import package for zooming on plots
import mpld3
mpld3.enable_notebook()

# Define parameters
k = 2.5         # [W/mK]
kappa = 1e-6    # [m^2/sec]
rhoCp = k/kappa
Qm = 45.0/1000    # [W/m^2]

# Determine temperature gradient at depth
dTdzmax = Qm/k # [C/m]

# Form grid
zmax = 150 # maximum depth of column [m]
dz = 0.25 # position step [m]
xmax = 25 # distance to either side of borehole [m]
xbore = 1 # width of borehole [m]
Tg = RasterModelGrid(zmax*4,((2*xmax)+xbore)*4,dz)

# Get core nodes
coreNodes = Tg.core_nodes

# Add temperature data
T = Tg.add_zeros('node', 'temperature')

# Set surface boundary condition
mT_yr = -10 # mean yearly temperature [deg C]
Ts0 = mT_yr

# Set initial condition for bulk
Tbulk = -10 # [deg C]
T[:] = Tbulk*np.ones(T.shape)

# Set initial condition for borehole
Tbore = 15 # [deg C]
bInd = xmax/dz + 1
for ii in range(len(T)):
    # Check depth
    if Tg.node_y[ii] < 100:
        # Remove full lines
        bb = ii%204
        # Check lateral position
        if (100 < bb < 105):
            # Set temperature
            T[ii] = Tbore
        
# Show intial state
plt.figure(0)
imshow_node_grid(Tg, 'temperature', grid_units = ['m','m'])
plt.gca().invert_yaxis()

# Define time vector
dt = 0.25*(dz**2/(2*kappa)) # 0.75 gives margin from what is req'd for stability
tProp = 8*(3600*24*365.25)
t = np.arange(0,tProp,dt)

# Set boundary types  
surf_nodes = range(204) # get nodes at the surface
T[surf_nodes] = Ts0*np.ones(T[surf_nodes].shape) # reset top of borehole to surface temp
Tg.status_at_node[surf_nodes] = FIXED_VALUE_BOUNDARY

Tg.set_closed_boundaries_at_grid_edges(True, False, True, False) # edges of block

#base_nodes = range(len(T)-204,len(T),1) # get nodes at the base of the column
#T[base_nodes] += dTdzmax*dz # make gradient correct
#Tg.status_at_node[base_nodes] = FIXED_GRADIENT_BOUNDARY

base_nodes = range(len(T)-204,len(T),1) # get nodes at the base of the column
Tbase = dTdzmax*zmax + Ts0
T[base_nodes] = Tbase*np.ones(T[base_nodes].shape)

# Main loop
for ii in range(2500):   
    # Compute temperature gradient
    dTdz = Tg.calculate_gradients_at_active_links(T)
    
    # Compute heat flux
    q = -k*dTdz

    # Compute temperature rate
    dqdz = Tg.calculate_flux_divergence_at_nodes(q)
    dTdt = (-1.0/(rhoCp))*dqdz

    # Update temperature
    T[coreNodes] += dTdt[coreNodes]*dt
                
# Plot final results
plt.figure(1)
imshow_node_grid(Tg, 'temperature', grid_units = ['m','m'])
plt.gca().invert_yaxis()

# Plot depth slice
plt.figure(2)
depth_slice = np.where(Tg.node_x == 5)
plt.plot(Tg.node_y[depth_slice],T[depth_slice])
plt.xlabel('Depth [m]')
plt.ylabel('Temperature [deg C]')
plt.title('Depth Slice')

# Plot slice across borehole
plt.figure(3)
bore_slice = np.where(Tg.node_y == 50)
plt.plot(Tg.node_x[bore_slice],T[bore_slice])
plt.xlabel('Lateral Position [m]')
plt.ylabel('Temperature [deg C]')
plt.title('Horizontal Slice')
