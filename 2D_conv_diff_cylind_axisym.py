# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:47:09 2021

@author: Cam
"""

# =============================================================================
# 
# Explicit Finite Difference Method Code
# Solves the 2D Temperature Convection-Diffusion Equation
# Assumes Tubular Plug-Flow-Reactor in Laminar Regime 
# Assumes hagen poiseuille velocity profile
# Heat Source-Sink Included Uses Laminar Nusselt Correlation for "h"
# Written by: Cameron Armstrong (2020)
# Institution: Virginia Commonwealth University
# 
# =============================================================================

# Required Modules
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 
import math

D = 0.0015875                                   # tubing diameter in m
zl = 30/100                                     # tubing length in m & x range
rl = D/2                                        # tubing diameter & y range
nz = 300                                        # x grid points
nr = 50                                         # y grid points
dz = zl/(nz-1)                                  # x stepsize
dr = rl/(nr-1)                                  # y stepsize
k= .12                                          # thermal conductvity W/(m*K)
p = 1750                                        # density (kg/m3)
Cp = 1172                                       # specifc heat (J/kg/K)
a = k/(p*Cp)                                    # thermal diffusivity (m2/s)
sigma = .001                                     # time step factor
dt = sigma*dz*dr/a                              # time stepsize 
Vr = math.pi*(D/2)**2*zl                        # tubing volume (m3)
Qmlm = 4                                        # volumetric flowrate (mL/min)
Q = (Qmlm*10**-6)/60                            # volumetric flowrate (m3/s)
Ac = math.pi*(D/2)**2                           # cross-sectional area (m2)
lamz = a*dt/dz**2                               # lumped coefficient
lamr = a*dt/dr**2                               # lumped coefficient
Nu = 3.66                                       # nusselt laminar flow in tube
h = Nu*k/D                                      # convective heat transfer coefficient (W/m2/K)
T0 = 130+273.15                                 # stream inlet temperature (degK)
Tw = 25+273.15                                  # wall temperature (degK)
reltol = 1e-9                                   # tolerance for convergence  

# grid formation
z = np.linspace(0, zl, nz) 
r = np.linspace(0, rl, nr) 
Z, R = np.meshgrid(z, r) 

# hagen poiseuille velocity field generation
uAvg = Q/Ac                                     # average velocity (m/s)
uMax = 2*uAvg                                   # max velocity (m/s)
u = np.zeros(nr)                                # array initilization
u[:] = np.linspace(0,rl,nr)                     # array intialization
u[:] = uMax*(1-(u[:]/(D/2))**2)                 # hagan-poiselle profile
u[-1] = 0                                       # no slip BC
u = np.array([u,]*nz)                           # velocity field
u = u.T                                         # transpose/align field
maxCFL = np.max(u*dt/dz)                        # CFL condition calc.
print('The max CFL is %s'%(maxCFL))

# comsol_center_raw = np.genfromtxt('han_comsol_2D_center.csv',delimiter=',')
# comsol_center_L = comsol_center_raw[:,0]
# comsol_center_T = comsol_center_raw[:,1]
# comsol_wall_raw = np.genfromtxt('han_comsol_2D_wall.csv',delimiter=',')
# comsol_wall_L = comsol_wall_raw[:,0]
# comsol_wall_T = comsol_wall_raw[:,1]

# main function loop
def lets_get_tubular(): 
    # array initialization
    Ttol = np.zeros((nr,nz)) 
    T = np.ones((nr, nz))*Tw  
    Tn = np.ones((nr, nz))*Tw 
    # initialize termination condition
    # compares norms of current and previous solution arrays
    termcond = (np.abs((np.linalg.norm(Ttol)-np.linalg.norm(Tn))))/np.linalg.norm(Tn)
    stepcount = 1 # step counter
    while termcond >= reltol: 
        termcond = np.abs((np.linalg.norm(Ttol)-np.linalg.norm(Tn)))/np.linalg.norm(Tn)
        Tn = T.copy()
        # FDM vectorized solution using explicit euler and CDS
        T[1:-1, 1:-1] = (Tn[1:-1,1:-1] - (u[1:-1,1:-1]*(dt/(2*dz))*(Tn[1:-1,2:]  \
                     -Tn[1:-1,:-2])) + a*dt/(2*dr)/R[1:-1,1:-1]*(Tn[2:,1:-1]-Tn[:-2,1:-1])     \
                     + lamz *(Tn[1:-1, 2:] - 2 * Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) \
                     + lamr* (Tn[2:,1:-1] - 2 * Tn[1:-1, 1:-1] + Tn[:-2, 1:-1])) \
                     - h*D*math.pi*(Tn[1:-1,1:-1]-Tw)*dt/p/Cp*zl/Vr  
        # BCs
        T[-1, :] = Tw # tubing wall temp dirichlet BC
        T[0, :] = T[1,:]
        T[:, 0] = T0 # inlet flow temp dirichlet BC
        T[:, -1] = T[:,-2] # outlet flow temp neumann BC
        Ttol=T.copy() # update solution
        stepcount += 1 # update counter
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, T[:], rstride=1, cstride=1, cmap=cm.viridis,
    #     linewidth=0, antialiased=True)
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$');
    # plt.show()
    T[:]=T[:]-273.15                            # converts back to degC
    
    # generates plots
    # top plot is 2D filled contour plot 
    # bottom plot is centerline and near-wall line data points
    fig1 = plt.subplot(211)
#    ax = fig1.gca()
#    plt.imshow(T[:])
    cont = plt.contourf(Z,R,T[:],50)
    ax = plt.gca()
    #ax.axis('scaled')
    ax.axes.get_yaxis().set_visible(True)
    plt.xlim(0,zl)
    plt.yticks(np.linspace(0,rl,2),fontsize ='7')
    plt.xlabel('Tubing Length (m)')
    cbar = plt.colorbar(cont)
    cbar.ax.set_ylabel('Temperature (degC)')
    
    centerline = 0
    wallline = nr-5
    centerline = int(centerline)
    wallline = int(wallline)
    centerT = T[centerline,:]
    wallT = T[wallline,:]
    
    fig2 = plt.subplot(212)
    plt.plot(z, centerT,label='center')
    plt.plot(z,wallT,label='wall')
    plt.legend()
    plt.ylabel('Temperature (degC)')
    plt.xlabel('Tubing Length (m)')
    
    # plt.figure(2)
    # plt.plot(comsol_center_L/0.3,comsol_center_T,label='Comsol-center')
    # #plt.plot(comsol_wall_L/0.3,comsol_wall_T,label='Comsol-wall')
    # plt.plot(z/0.3,centerT,label='Python-center')
    # #plt.plot(z/0.3,wallT,label='Python-wall')
    # plt.ylabel('Temperature (degC)')
    # plt.xlabel('Normalized Reactor Length')
    # plt.legend()
    
    plt.show()
    print('Stepcount = %s' %(stepcount))
    
if __name__ == "__main__":
    lets_get_tubular()